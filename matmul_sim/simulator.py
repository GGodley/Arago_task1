"""Event-driven simulator: the clock.

    simulate(trace, hw_config, scheduler) -> SimResult

Loop, one iteration per tick at which something finishes:

    1. retire every command that ends now: mark its output READY, consume
       one use of each input (freeing tiles nobody needs any more), and mark
       trace ops done so their dependents become ready
    2. ask the scheduler for candidates and issue each one that has a free
       unit, ready inputs and (if it writes a tile) a free memory slot;
       when memory is full, ask the scheduler for a victim to evict
    3. if nothing is running: either everything is done, or the run is
       deadlocked (work left, nothing can start)
    4. jump to the next tick at which a running command finishes

Commands and how they come from the trace:

    LOAD / COMPUTE / SAVE   one command per trace op, ready when its deps are done
    ACC group of n srcs     n-1 binary ACC commands created at run time: whenever
                            two tiles of the group are ready and not in use, an
                            ACC command for the group is offered; the scheduler
                            picks the pair when it is issued
    reload LOAD             created when an input tile that is still needed is
                            evicted; it brings the tile back with its remaining uses

Only tiles that came from a LOAD (inputs, which still exist in global memory)
can be evicted. Partials and sums are never dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .bounds import LowerBound, lower_bound, min_memory_tiles
from .hw_config import HWConfig
from .hw_model import HWModel
from .schedulers.base import Command, Scheduler
from .trace import OpType, Trace


@dataclass(frozen=True)
class ScheduledCmd:
    """One executed hardware command, for checks and Gantt charts."""

    start: int
    end: int
    unit: str
    slot: int
    kind: str          # LOAD | COMPUTE | ACC | SAVE
    op_id: int | None  # trace op (ACC: group op id; None for reloads)
    srcs: tuple[str, ...]
    dst: str | None
    reload: bool = False


@dataclass
class SimResult:
    workload: str
    scheduler: str
    config: HWConfig
    completed: bool
    deadlock: bool                   # stopped: work left, nothing running, nothing can start
    stalled: bool                    # stopped at max_ticks without finishing (e.g. reload thrashing)
    total_ticks: int
    total_cycles: int
    matrix_util: float
    dma_util: float
    mem_peak: int
    mem_mean_pct: float              # time-weighted mean occupancy, % of capacity
    mem_series: list[tuple[int, int]]  # (tick, tiles allocated), step series
    commands: dict[str, int]         # hardware commands executed per kind
    reloads: int
    evictions: int
    bound: LowerBound
    useful_fraction: float = 1.0
    ops_done: int = 0
    ops_total: int = 0
    stuck: list[str] = field(default_factory=list)  # a few blocked commands, on deadlock
    schedule: list[ScheduledCmd] = field(default_factory=list)
    min_mem_tiles: int = 0

    @property
    def infeasible_memory(self) -> bool:
        """Local memory is below the minimum any schedule needs (see bounds.min_memory_tiles)."""
        return self.config.local_mem_tiles < self.min_mem_tiles

    @property
    def bound_ratio(self) -> float:
        """Simulated ticks / lower bound (1.0 = optimal)."""
        return self.total_ticks / self.bound.ticks if self.bound.ticks else float("nan")

    @property
    def effective_matrix_util(self) -> float:
        """Matrix utilization excluding work spent on padding."""
        return self.matrix_util * self.useful_fraction

    def summary(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "scheduler": self.scheduler,
            "n_dma": self.config.n_dma,
            "n_matrix": self.config.n_matrix,
            "local_mem_tiles": self.config.local_mem_tiles,
            "completed": self.completed,
            "deadlock": self.deadlock,
            "stalled": self.stalled,
            "infeasible_memory": self.infeasible_memory,
            "total_ticks": self.total_ticks,
            "total_cycles": self.total_cycles,
            "lower_bound_ticks": self.bound.ticks,
            "bound_ratio": round(self.bound_ratio, 4),
            "bound_limiter": self.bound.limiter,
            "matrix_util": round(self.matrix_util, 4),
            "effective_matrix_util": round(self.effective_matrix_util, 4),
            "dma_util": round(self.dma_util, 4),
            "mem_peak": self.mem_peak,
            "mem_mean_pct": round(self.mem_mean_pct, 2),
            "reloads": self.reloads,
            "evictions": self.evictions,
            **{f"cmd_{k.lower()}": v for k, v in self.commands.items()},
        }


class SimError(RuntimeError):
    """The simulator hit an internal inconsistency."""


class Simulator:
    def __init__(self, trace: Trace, config: HWConfig, scheduler: Scheduler, max_ticks: int | None = None,
                 record: bool = True, max_ticks_factor: float | None = None) -> None:
        """record=False skips the per-command schedule log (sweeps don't need it).
        max_ticks_factor stops the run once it exceeds that multiple of the lower bound."""
        self.trace = trace
        self.config = config
        self.scheduler = scheduler
        self.hw = HWModel(config)
        self.now = 0
        self.mem_blocked = False

        ops = trace.ops
        self.ops = {op.id: op for op in ops}
        if len(self.ops) != len(ops):
            raise SimError("duplicate op ids; validate the trace first")

        # Dependency bookkeeping (ACC groups are handled through tile availability).
        self.remaining_deps: dict[int, int] = {}
        self.dependents: dict[int, list[int]] = defaultdict(list)
        for op in ops:
            self.remaining_deps[op.id] = len(op.deps)
            for d in op.deps:
                self.dependents[d].append(op.id)

        # Tile bookkeeping.
        self.uses: dict[str, int] = defaultdict(int)       # trace consumers per tile
        for op in ops:
            for s in op.srcs:
                self.uses[s] += 1
        self.producer: dict[str, int] = {op.dst: op.id for op in ops if op.dst is not None}
        self.load_op_of: dict[str, int] = {op.dst: op.id for op in ops if op.type is OpType.LOAD}
        self.group_of: dict[str, int] = {}                 # partial tile -> ACC group op id
        self.group_avail: dict[int, list[str]] = {}
        self.group_issued: dict[int, int] = {}
        self.group_enqueued: dict[int, bool] = {}
        for op in ops:
            if op.type is OpType.ACC:
                producers = {self.producer[s] for s in op.srcs}
                if set(op.deps) != producers:
                    raise SimError(f"ACC op {op.id}: deps must be exactly the producers of its srcs")
                for s in op.srcs:
                    self.group_of[s] = op.id
                self.group_avail[op.id] = []
                self.group_issued[op.id] = 0
                self.group_enqueued[op.id] = False
        self.reload_uses: dict[str, int] = {}

        # Running commands, keyed by a run-time command id.
        self._next_cmd = 0
        self.running: dict[int, Command] = {}  # cmd id -> command
        self.schedule: list[ScheduledCmd] = []
        self.record = record

        # Counters.
        self.ops_done = 0
        self.done: set[int] = set()
        self.commands: dict[str, int] = {t.value: 0 for t in OpType}
        self.reloads = 0
        self.evictions = 0

        self.bound = lower_bound(trace, config)
        if max_ticks is None and max_ticks_factor is not None:
            max_ticks = int(max_ticks_factor * self.bound.ticks)
        if max_ticks is None:
            hw_cmds = sum(trace.hw_op_counts().values())
            max_ticks = 1000 + 20 * hw_cmds * max(config.dma_ticks, config.matrix_ticks)
        self.max_ticks = max_ticks

    # ------------------------------------------------------------------ run
    def run(self) -> SimResult:
        self.scheduler.reset(self)
        for op in self.trace.ops:
            if self.remaining_deps[op.id] == 0 and op.type is not OpType.ACC:
                self._make_ready(op.id)

        total = len(self.ops)
        deadlock = False
        stalled = False
        while True:
            self._retire()
            self._schedule()
            if self.hw.idle():
                if self.ops_done != total:
                    deadlock = True
                break
            nxt = self.hw.next_event()
            if nxt is None or nxt > self.max_ticks:
                stalled = True
                break
            self.now = nxt

        completed = not deadlock and not stalled
        util = self.hw.utilization(self.now)
        mem = self.hw.memory
        series = mem.occupancy_series()
        return SimResult(
            workload=str(self.trace.workload.get("name", "unnamed")),
            scheduler=self.scheduler.name,
            config=self.config,
            completed=completed,
            deadlock=deadlock,
            stalled=stalled,
            total_ticks=self.now,
            total_cycles=self.now * self.config.tick_cycles,
            matrix_util=util["matrix"],
            dma_util=util["dma"],
            mem_peak=mem.peak,
            mem_mean_pct=_time_weighted_mean(series, self.now) / mem.capacity * 100,
            mem_series=series,
            commands=dict(self.commands),
            reloads=self.reloads,
            evictions=self.evictions,
            bound=self.bound,
            useful_fraction=float(self.trace.workload.get("useful_fraction", 1.0)),
            ops_done=self.ops_done,
            ops_total=total,
            stuck=self._stuck_report() if not completed else [],
            schedule=self.schedule,
            min_mem_tiles=min_memory_tiles(self.trace),
        )

    # ------------------------------------------------------------- readiness
    def _make_ready(self, op_id: int) -> None:
        op = self.ops[op_id]
        self.scheduler.on_ready(Command(op.type, op.id, op.id, op.srcs, op.dst))

    def _op_done(self, op_id: int) -> None:
        self.ops_done += 1
        self.done.add(op_id)
        for d in self.dependents.get(op_id, ()):
            self.remaining_deps[d] -= 1
            if self.remaining_deps[d] == 0 and self.ops[d].type is not OpType.ACC:
                self._make_ready(d)

    def _tile_available_for_group(self, tile: str, group: int) -> None:
        self.group_avail[group].append(tile)
        self._maybe_offer_acc(group)

    def _maybe_offer_acc(self, group: int) -> None:
        n = len(self.ops[group].srcs)
        if (not self.group_enqueued[group] and len(self.group_avail[group]) >= 2
                and self.group_issued[group] < n - 1):
            self.group_enqueued[group] = True
            self.scheduler.on_ready(Command(OpType.ACC, group, group))

    # ----------------------------------------------------------------- issue
    def _schedule(self) -> None:
        self.mem_blocked = False
        if self.hw.dma.free_count == 0 and self.hw.matrix.free_count == 0:
            return
        gen = self.scheduler.candidates(self)
        try:
            for cmd in gen:
                cmd.issued = self._try_issue(cmd)
                if self.hw.dma.free_count == 0 and self.hw.matrix.free_count == 0:
                    break
        finally:
            gen.close()

    def _try_issue(self, cmd: Command) -> bool:
        pool = self.hw.pool(cmd.unit)
        mem = self.hw.memory
        if pool.free_count == 0:
            return False

        if cmd.kind is OpType.ACC:
            group = cmd.op_id
            avail = self.group_avail[group]
            if len(avail) < 2:
                return False
            x, y = self.scheduler.pair(self, group, avail)
            n = len(self.ops[group].srcs)
            last = self.group_issued[group] == n - 2
            dst = self.ops[group].dst if last else f"{self.ops[group].dst}#{self.group_issued[group] + 1}"
            srcs = (x, y)
        elif cmd.is_reload:
            if mem.is_resident(cmd.dst):
                raise SimError(f"reload of resident tile {cmd.dst!r}")
            srcs, dst = (), cmd.dst
        else:
            srcs, dst = cmd.srcs, cmd.dst
            if not all(mem.is_ready(s) for s in srcs):
                return False  # an input was evicted; its reload is pending

        if cmd.needs_memory and not mem.can_allocate():
            if not self._evict_for(cmd, srcs):
                self.mem_blocked = True
                return False

        # Commit.
        if cmd.kind is OpType.ACC:
            group = cmd.op_id
            avail = self.group_avail[group]
            avail.remove(srcs[0])
            avail.remove(srcs[1])
            self.group_issued[group] += 1
            self.group_enqueued[group] = False
            cmd.srcs, cmd.dst = srcs, dst
        if dst is not None:
            if cmd.is_reload:
                uses = self.reload_uses.pop(dst)
            elif cmd.kind is OpType.ACC and dst != self.ops[cmd.op_id].dst:
                uses = 1  # intermediate sum: consumed by one later ACC
            else:
                uses = self.uses.get(dst, 0)
            mem.allocate(dst, uses, self.now)
        for s in srcs:
            mem.pin(s)
        cmd_id = self._next_cmd
        self._next_cmd += 1
        pool.start(cmd_id, self.now, self.hw.duration(cmd.unit))
        self.running[cmd_id] = cmd
        self.commands[cmd.kind.value] += 1
        if cmd.is_reload:
            self.reloads += 1
        if cmd.kind is OpType.ACC:
            self._maybe_offer_acc(cmd.op_id)  # more pairs may be ready
        return True

    def _evict_for(self, cmd: Command, protect: tuple[str, ...]) -> bool:
        """Free one slot by evicting an input tile chosen by the scheduler."""
        if not self.scheduler.evicts:
            return False
        mem = self.hw.memory
        cands = [t for t in mem.evictable() if t in self.load_op_of and t not in protect]
        if not cands:
            return False
        victim = self.scheduler.victim(self, cands, cmd)
        if victim is None:
            return False
        if victim not in cands:
            raise SimError(f"scheduler chose non-evictable tile {victim!r}")
        left = mem.evict(victim, self.now)
        self.evictions += 1
        if left > 0:
            self.reload_uses[victim] = left
            self.scheduler.on_ready(Command(OpType.LOAD, self.load_op_of[victim], None, (), victim, is_reload=True))
        return True

    # ---------------------------------------------------------------- retire
    def _retire(self) -> None:
        for pool in (self.hw.dma, self.hw.matrix):
            for slot, cmd_id, end in pool.running():
                if end == self.now:
                    pool.finish(slot, self.now)
                    cmd = self.running.pop(cmd_id)
                    start = pool.history[-1].start
                    if self.record:
                        self.schedule.append(ScheduledCmd(start, end, pool.kind, slot, cmd.kind.value, cmd.op_id,
                                                          cmd.srcs, cmd.dst, cmd.is_reload))
                    self._complete(cmd)

    def _complete(self, cmd: Command) -> None:
        mem = self.hw.memory
        for s in cmd.srcs:
            if mem.unpin(s) == 0:
                mem.free_tile(s, self.now)
        if cmd.dst is not None:
            mem.mark_ready(cmd.dst, self.now)

        if cmd.is_reload:
            return
        if cmd.kind is OpType.ACC:
            group = cmd.op_id
            if cmd.dst == self.ops[group].dst:
                self._op_done(group)
            else:
                self._tile_available_for_group(cmd.dst, group)
            return
        self._op_done(cmd.op_id)
        if cmd.kind is OpType.COMPUTE and cmd.dst in self.group_of:
            self._tile_available_for_group(cmd.dst, self.group_of[cmd.dst])

    def _stuck_report(self, limit: int = 5) -> list[str]:
        """A few ops that could run by their deps but never started, and why."""
        mem = self.hw.memory
        out = []
        for op_id, op in self.ops.items():
            if op_id in self.done or op.type is OpType.ACC or self.remaining_deps[op_id]:
                continue
            if op.dst is not None and mem.is_resident(op.dst):
                continue  # running
            missing = [s for s in op.srcs if not mem.is_ready(s)]
            if missing:
                why = f" for evicted inputs {missing}"
            elif op.type is OpType.SAVE:
                why = " for a free DMA unit"
            else:
                why = " for a free memory slot"
            out.append(f"op {op_id} {op.type.value} waiting{why}")
            if len(out) >= limit:
                break
        out.append(f"memory {self.hw.memory.used}/{self.hw.memory.capacity} tiles in use")
        return out


def _time_weighted_mean(series: list[tuple[int, int]], end: int) -> float:
    if end <= 0:
        return 0.0
    total = 0
    for (t0, v), (t1, _) in zip(series, series[1:] + [(end, 0)]):
        total += v * (min(t1, end) - t0)
    return total / end


def simulate(trace: Trace, config: HWConfig, scheduler: Scheduler, max_ticks: int | None = None,
             record: bool = True, max_ticks_factor: float | None = None) -> SimResult:
    return Simulator(trace, config, scheduler, max_ticks, record, max_ticks_factor).run()
