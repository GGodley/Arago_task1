"""In-order (FIFO) scheduler: the baseline.

Offers ready commands in trace order (lowest op id first) across both unit
pools, and issues everything that fits. It never evicts, so when memory fills
up with input tiles and no command can allocate its output, the run
deadlocks. That is intended: it is the reference the smarter schedulers must
beat, and it demonstrates the deadlock risk.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Iterator

from ..trace import OpType
from .base import Command, Scheduler

if TYPE_CHECKING:  # pragma: no cover
    from ..simulator import Simulator


class FIFOScheduler(Scheduler):
    name = "fifo"

    def reset(self, sim: "Simulator") -> None:
        # One heap per command kind, so blocked kinds can be skipped without popping.
        self._heaps: dict[OpType, list[tuple[int, int, Command]]] = {k: [] for k in OpType}
        self._seq = itertools.count()

    def priority(self, cmd: Command) -> int:
        return cmd.priority

    def on_ready(self, cmd: Command) -> None:
        heapq.heappush(self._heaps[cmd.kind], (self.priority(cmd), next(self._seq), cmd))

    def candidates(self, sim: "Simulator") -> Iterator[Command]:
        skipped: list[tuple[int, int, Command]] = []
        current = None
        try:
            while True:
                best = None
                for kind, heap in self._heaps.items():
                    if not heap:
                        continue
                    if sim.hw.pool(kind.unit).free_count == 0:
                        continue
                    if sim.mem_blocked and kind is not OpType.SAVE:
                        continue
                    if best is None or heap[0] < self._heaps[best][0]:
                        best = kind
                if best is None:
                    return
                current = heapq.heappop(self._heaps[best])
                yield current[2]
                if not current[2].issued:
                    skipped.append(current)
                current = None
        finally:
            if current is not None and not current[2].issued:
                skipped.append(current)
            for item in skipped:
                heapq.heappush(self._heaps[item[2].kind], item)

    def pending(self) -> int:
        return sum(len(h) for h in self._heaps.values())
