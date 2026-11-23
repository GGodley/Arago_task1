"""Analytical lower bound on run time, used to validate the simulator.

For any schedule, in ticks:

    matrix:  Ld + ceil(matrix_cmds / Nm) * Lm + Ld
             no matrix command can start before the first LOAD lands, and the
             last one is followed by at least one SAVE
    dma:     ceil(dma_cmds / Nd) * Ld
             every LOAD and SAVE occupies one DMA unit for Ld (reloads only add)
    path:    Ld + Lm * (1 + ceil(log2 Kt)) + Ld
             one output tile: load, compute partials in parallel, pair them up
             in a tree, save

    lower bound = max(matrix, dma, path)

Ld and Lm are the DMA and matrix latencies in ticks (4 and 1 by default).
A single 64x64x64 matmul gives max(9, 12/Nd, 9) = 9 ticks = 1152 cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hw_config import HWConfig
from .trace import Trace


@dataclass(frozen=True)
class LowerBound:
    matrix: int
    dma: int
    path: int

    @property
    def ticks(self) -> int:
        return max(self.matrix, self.dma, self.path)

    @property
    def limiter(self) -> str:
        """Which term binds: 'matrix', 'dma' or 'path'."""
        return max(("matrix", self.matrix), ("dma", self.dma), ("path", self.path), key=lambda kv: kv[1])[0]


def min_memory_tiles(trace: Trace) -> int:
    """Smallest local memory that can finish the trace, working one output tile at a time.

    3 tiles when there is no ACC (A, B, P); 4 otherwise (running sum, A, B, new partial).
    """
    has_acc = any(op.type.value == "ACC" for op in trace.ops)
    return 4 if has_acc else 3


def lower_bound(trace: Trace, hw: HWConfig) -> LowerBound:
    c = trace.hw_op_counts()
    matrix_cmds = c["COMPUTE"] + c["ACC"]
    dma_cmds = c["LOAD"] + c["SAVE"]
    ld, lm = hw.dma_ticks, hw.matrix_ticks

    kt = max((len(op.srcs) for op in trace.ops if op.type.value == "ACC"), default=1)
    matrix = (ld + math.ceil(matrix_cmds / hw.n_matrix) * lm + ld) if matrix_cmds else 0
    dma = math.ceil(dma_cmds / hw.n_dma) * ld
    path = ld + lm * (1 + math.ceil(math.log2(kt))) + ld if matrix_cmds else 2 * ld
    return LowerBound(matrix=matrix, dma=dma, path=path)
