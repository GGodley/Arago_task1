"""Reuse-first schedulers: greedy and memory-aware.

Both rank ready commands the same way:

    matrix units   ACC first (it frees memory and finishes outputs), then
                   COMPUTE, preferring output tiles that are already open
    DMA units      SAVE first (frees memory), then "hot" LOADs that complete
                   a pair with a tile already in memory, then "cold" LOADs in
                   tile order (row, snake or blocked)

GreedyReuseScheduler issues everything that fits. Like FIFO it never evicts,
so it can fill memory with inputs and deadlock; it just reuses better.

MemoryAwareScheduler adds three rules:

    1. load gating      a LOAD must leave room behind it: 1 free slot if its
                        partner tile is already in memory (room for the
                        output), 2 if not (room for the partner and the output)
    2. open outputs     at most max(1, capacity - 3) output tiles in progress
                        at once; a COMPUTE that would open another one waits
    3. eviction         when a COMPUTE or ACC can't get a slot, evict the input
                        tile whose next use is furthest away, never one that a
                        ready COMPUTE is about to read. Never evicts for LOADs,
                        which is what causes thrashing.

Tile orders (both schedulers):

    row      output tiles row by row
    snake    rows alternate direction, so the last B column of a row is
             reused by the next row
    blocked  groups of block x block output tiles, which share block rows of
             A and block columns of B

The order is read from the translator's tile names (P_i_j_k). If names do not
follow that convention, trace order is used.
"""

from __future__ import annotations

import heapq
import itertools
import re
from typing import TYPE_CHECKING, Iterator

from ..trace import OpType
from .base import Command, Scheduler

if TYPE_CHECKING:  # pragma: no cover
    from ..simulator import Simulator

_PARTIAL = re.compile(r"^P_(\d+)_(\d+)_(\d+)$")
ORDERS = ("row", "snake", "blocked")
INF = (float("inf"),)


def output_rank(i: int, j: int, Mt: int, Nt: int, order: str, block: int) -> int:
    """Position of output tile (i, j) in the visiting order."""
    if order == "row":
        return i * Nt + j
    if order == "snake":
        return i * Nt + (j if i % 2 == 0 else Nt - 1 - j)
    if order == "blocked":
        bi, bj = i // block, j // block
        blocks_per_row = -(-Nt // block)
        h = min(block, Mt - bi * block)
        w = min(block, Nt - bj * block)
        return (bi * blocks_per_row + bj) * block * block + (i - bi * block) * w + (j - bj * block)
    raise ValueError(f"unknown order {order!r}, expected one of {ORDERS}")


class GreedyReuseScheduler(Scheduler):
    name = "greedy"
    memory_aware = False

    def __init__(self, order: str = "row", block: int = 4) -> None:
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}, expected one of {ORDERS}")
        if block < 1:
            raise ValueError("block must be >= 1")
        self.order = order
        self.block = block
        if order != "row":
            self.name = f"{self.name}-{order}"

    # ------------------------------------------------------------- setup
    def reset(self, sim: "Simulator") -> None:
        trace = sim.trace
        w = trace.workload
        Mt, Nt = int(w.get("Mt", 0) or 0), int(w.get("Nt", 0) or 0)

        self.rank: dict[int, tuple] = {}            # COMPUTE op id -> (output rank, k)
        self.out_key: dict[int, int] = {}           # COMPUTE op id -> output key
        self.close_op: dict[int, int] = {}          # output key -> SAVE op id
        self.consumers: dict[str, list[int]] = {}   # input tile -> COMPUTE op ids
        self.pending: dict[str, set[int]] = {}      # input tile -> COMPUTEs not yet issued
        self.srcs: dict[int, tuple[str, ...]] = {}

        save_of = {op.srcs[0]: op.id for op in trace.ops if op.type is OpType.SAVE}
        for op in trace.ops:
            if op.type is OpType.COMPUTE:
                m = _PARTIAL.match(op.dst or "")
                if m and Mt and Nt:
                    i, j, k = map(int, m.groups())
                    self.rank[op.id] = (output_rank(i, j, Mt, Nt, self.order, self.block), k)
                else:
                    self.rank[op.id] = (op.id, 0)
                group = sim.group_of.get(op.dst)
                key = group if group is not None else op.id
                self.out_key[op.id] = key
                final = sim.ops[group].dst if group is not None else op.dst
                self.close_op[key] = save_of[final]
                self.srcs[op.id] = op.srcs
                for s in op.srcs:
                    self.consumers.setdefault(s, []).append(op.id)
                    self.pending.setdefault(s, set()).add(op.id)
        # Rank of everything else, derived from the COMPUTEs it serves.
        self.key_rank: dict[int, tuple] = {}
        for c, key in self.out_key.items():
            r = self.rank[c]
            if key not in self.key_rank or r < self.key_rank[key]:
                self.key_rank[key] = r

        self.computes_of: dict[int, list[int]] = {}
        for c in sorted(self.out_key, key=lambda c: self.rank[c]):
            self.computes_of.setdefault(self.out_key[c], []).append(c)
        self.keys_by_rank = sorted(self.computes_of, key=lambda k: self.key_rank[k])
        self._next_key = 0
        self.issued_computes: set[int] = set()
        self.focus: set[int] = set()

        self.open: set[int] = set()                 # output keys with work started, not saved
        self.ready_loads: dict[str, Command] = {}   # tile -> pending LOAD (or reload) command
        self._heaps: dict[str, list] = {n: [] for n in ("acc", "compute", "save", "hot", "cold")}
        self._seq = itertools.count()
        self.sim = sim
        mem = sim.config.local_mem_tiles
        # Worked one step at a time, an output tile needs at most J slots:
        # running sum + next A + next B + the new partial (3 when Kt = 1).
        kt = max((len(op.srcs) for op in trace.ops if op.type is OpType.ACC), default=1)
        self.job_slots = 4 if kt > 1 else 3
        self.max_open = max(1, mem // self.job_slots)
        self.live: dict[int, int] = {}              # output key -> partial/sum tiles allocated

    # -------------------------------------------------------------- ranks
    def _load_rank(self, tile: str) -> tuple:
        ranks = [self.rank[c] for c in self.pending.get(tile, ())]
        return min(ranks) if ranks else INF

    def _push(self, heap: str, key: tuple, cmd: Command) -> None:
        heapq.heappush(self._heaps[heap], (key, next(self._seq), cmd))

    # ------------------------------------------------------------ protocol
    def on_ready(self, cmd: Command) -> None:
        k = cmd.kind
        if k is OpType.LOAD:
            self.ready_loads[cmd.dst] = cmd
            self._push("cold", self._load_rank(cmd.dst), cmd)
            if self._partner_present(cmd.dst):
                self._push("hot", self._load_rank(cmd.dst), cmd)
        elif k is OpType.COMPUTE:
            opened = 0 if self.out_key[cmd.op_id] in self.open else 1
            self._push("compute", (opened,) + self.rank[cmd.op_id], cmd)
        elif k is OpType.ACC:
            self._push("acc", self.key_rank.get(cmd.op_id, (cmd.op_id,)), cmd)
        else:  # SAVE
            self._push("save", (cmd.op_id,), cmd)

    def candidates(self, sim: "Simulator") -> Iterator[Command]:
        self._update_focus()
        matrix_heaps = ("acc", "compute")
        dma_heaps = ("save", "urgent", "hot", "cold")
        for names, unit in ((matrix_heaps, "matrix"), (dma_heaps, "dma")):
            for name in names:
                if sim.hw.pool(unit).free_count == 0:
                    break
                if sim.mem_blocked and name != "save":
                    continue
                if name == "urgent":
                    yield from self._urgent_loads(sim)
                elif name in ("hot", "cold") and not self._prefetch_ok(sim):
                    continue
                else:
                    yield from self._drain(name, sim, unit)

    def _urgent_loads(self, sim: "Simulator") -> Iterator[Command]:
        """Loads for the next step of each output in focus, in rank order (no-op for greedy)."""
        return iter(())

    def _prefetch_ok(self, sim: "Simulator") -> bool:
        """Whether loads beyond the outputs in focus may be issued (always, for greedy)."""
        return True

    def _drain(self, name: str, sim: "Simulator", unit: str) -> Iterator[Command]:
        heap = self._heaps[name]
        skipped = []
        current = None  # the item last yielded; settled even if the simulator closes us at the yield
        try:
            while heap and sim.hw.pool(unit).free_count > 0:
                if sim.mem_blocked and name != "save":
                    break
                item = heapq.heappop(heap)
                cmd = item[2]
                if cmd.issued or self._stale(name, cmd):
                    continue
                if not self._allowed(cmd, sim):
                    skipped.append(item)
                    continue
                current = item
                yield cmd
                self._settle(current, skipped)
                current = None
        finally:
            if current is not None:
                self._settle(current, skipped)
            for item in skipped:
                heapq.heappush(heap, item)

    def _settle(self, item: tuple, skipped: list) -> None:
        if item[2].issued:
            self._on_issued(item[2])
        else:
            skipped.append(item)

    def _note_acc(self, cmd: Command) -> None:
        self.live[cmd.op_id] = self.live.get(cmd.op_id, 0) - 1  # two tiles in, one out

    def _stale(self, name: str, cmd: Command) -> bool:
        if cmd.kind is OpType.LOAD:
            return self.ready_loads.get(cmd.dst) is not cmd
        return False

    def _on_issued(self, cmd: Command) -> None:
        if cmd.kind is OpType.ACC:
            self._note_acc(cmd)
        elif cmd.kind is OpType.LOAD:
            del self.ready_loads[cmd.dst]
            # Its partners are now worth loading early.
            for c in self.pending.get(cmd.dst, ()):
                for s in self.srcs[c]:
                    if s != cmd.dst and s in self.ready_loads:
                        self._push("hot", self.rank[c], self.ready_loads[s])
        elif cmd.kind is OpType.COMPUTE:
            for s in cmd.srcs:
                self.pending[s].discard(cmd.op_id)
            self.issued_computes.add(cmd.op_id)
            key = self.out_key[cmd.op_id]
            self.live[key] = self.live.get(key, 0) + 1
            if key not in self.open:
                self.open.add(key)
                self._update_focus()

    # ------------------------------------------------------------- helpers
    def _partner_present(self, tile: str) -> bool:
        mem = self.sim.hw.memory
        for c in self.pending.get(tile, ()):
            for s in self.srcs[c]:
                if s != tile and mem.is_resident(s):
                    return True
        return False

    def _update_focus(self) -> None:
        """Outputs being worked on: the open ones, plus the next one in order if there is room."""
        self._open_count()
        focus = set(self.open)
        if len(self.open) < self.max_open:
            while self._next_key < len(self.keys_by_rank):
                key = self.keys_by_rank[self._next_key]
                if key in self.open or self.close_op[key] in self.sim.done or any(
                        c in self.issued_computes for c in self.computes_of[key]):
                    self._next_key += 1
                    continue
                focus.add(key)
                break
        self.focus = focus

    def _next_compute(self, key: int) -> int | None:
        for c in self.computes_of[key]:
            if c not in self.issued_computes:
                return c
        return None

    def _urgent(self, tile: str) -> bool:
        """The tile is an input of the next step of an output in focus."""
        for c in self.pending.get(tile, ()):
            key = self.out_key[c]
            if key in self.focus and self._next_compute(key) == c:
                return True
        return False

    def _open_count(self) -> int:
        done = self.sim.done
        self.open = {k for k in self.open if self.close_op[k] not in done}
        return len(self.open)

    def _allowed(self, cmd: Command, sim: "Simulator") -> bool:
        """Extra admission rules; none for the greedy scheduler."""
        return True


class MemoryAwareScheduler(GreedyReuseScheduler):
    name = "memaware"
    memory_aware = True
    evicts = True

    def _allowed(self, cmd: Command, sim: "Simulator") -> bool:
        spare = self._spare(sim)
        if cmd.kind is OpType.LOAD:
            # Inputs for the next step of an output in focus always go (evicting
            # if needed; see _urgent_loads); anything else is prefetch and only
            # uses spare room.
            return self._urgent(cmd.dst) or spare
        if cmd.kind is OpType.COMPUTE:
            key = self.out_key[cmd.op_id]
            if key in self.open:
                # One step at a time per output, unless memory has room to spare.
                return self.live.get(key, 0) <= 1 or spare
            return self._open_count() < self.max_open
        return True

    def _spare(self, sim: "Simulator") -> bool:
        return sim.hw.memory.free - 1 >= self.job_slots * max(1, len(self.open))

    def _prefetch_ok(self, sim: "Simulator") -> bool:
        return self._spare(sim)

    def _urgent_loads(self, sim: "Simulator") -> Iterator[Command]:
        wanted = []
        for key in self.focus:
            c = self._next_compute(key)
            if c is None:
                continue
            for s in self.srcs[c]:
                cmd = self.ready_loads.get(s)
                if cmd is not None:
                    wanted.append((self.rank[c], s, cmd))
        current = None  # settled even if the simulator closes us at the yield
        try:
            for _, s, cmd in sorted(wanted, key=lambda w: w[0]):
                if sim.hw.dma.free_count == 0 or sim.mem_blocked:
                    return
                if cmd.issued or self.ready_loads.get(s) is not cmd:
                    continue
                current = cmd
                yield cmd
                if cmd.issued:
                    self._on_issued(cmd)
                current = None
        finally:
            if current is not None and current.issued:
                self._on_issued(current)

    def victim(self, sim: "Simulator", candidates: list[str], cmd: Command) -> str | None:
        if cmd.kind is OpType.SAVE or (cmd.kind is OpType.LOAD and not self._urgent(cmd.dst)):
            return None  # never evict to prefetch: that is what thrashes
        if cmd.kind is OpType.COMPUTE and not (self.out_key[cmd.op_id] in self.focus
                                               and self._next_compute(self.out_key[cmd.op_id]) == cmd.op_id):
            return None  # opportunistic extra work doesn't get to evict
        # Keep the inputs of the next step of every output in focus.
        protected = set()
        for key in self.focus:
            c = self._next_compute(key)
            if c is not None:
                protected.update(self.srcs[c])
        pool = [t for t in candidates if t not in protected]
        if not pool:
            return None
        mem = sim.hw.memory
        # Furthest next use first; among equals, the one with fewest uses left.
        return max(pool, key=lambda t: (self._load_rank(t), -mem.entry(t).remaining_uses))
