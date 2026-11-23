"""Step 3: time from work, traffic and units (the validated closed-form model).

    T = min over tilings of max(T_mat, T_dma, T_path)

    T_mat  = 2 Ld + ceil(W / Nm) Lm            matrix units busy (plus first load, last save)
    T_dma  = (L + Sv) max(Ld / Nd, tau / F)    DMA units busy, or memory slots run out
    T_path = 2 Ld + Lm (1 + ceil(log2 k))      dependency chain: load, compute, sum tree, save

F = S - R - 1 is the memory left for tiles in flight once a tiling keeps R tiles
resident. By Little's law F slots can sustain at most F / tau transfers per tick,
so DMA units beyond F Ld / tau add nothing: memory caps useful DMA parallelism.
Validated against the simulator: median error 2.5%, 84% of runs within 25%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .counts import cdiv, tiles

LD, LM = 4, 1  # ticks per DMA transfer, per matrix op (1 tick = 128 cycles)


@dataclass(frozen=True)
class Timing:
    T: float
    t_mat: float
    t_dma: float
    t_path: float
    loads: int
    strategy: str
    R: int
    F: int

    @property
    def limiter(self) -> str:
        """Term that sets T. On a tie the dependency path wins (more units cannot help),
        then DMA."""
        if self.t_path >= self.T - 1e-9:
            return "path"
        return "dma" if self.t_dma >= self.t_mat else "matrix"


def _tilings(m: int, k: int, n: int):
    """(name, loads, resident tiles R) for block and panel tilings."""
    acc = 1 if k > 1 else 0
    for bm in range(1, m + 1):
        for bn in range(1, n + 1):
            yield f"block{bm}x{bn}", k * (m * cdiv(n, bn) + n * cdiv(m, bm)), acc * bm * bn
    for p in range(1, m + 1):
        yield f"Apanel{p}", m * k + n * k * cdiv(m, p), p * k + acc * p
    for p in range(1, n + 1):
        yield f"Bpanel{p}", n * k + m * k * cdiv(n, p), p * k + acc * p


def predict_time(M: int, K: int, N: int, nd: int, nm: int, S: int) -> Timing | None:
    """Ticks to run an MxK @ KxN matmul on Nd DMA units, Nm matrix units, S tiles."""
    m, k, n = tiles(M, K, N)
    acc = 1 if k > 1 else 0
    W, Sv = m * n * (2 * k - 1), m * n
    t_mat = 2 * LD + LM * cdiv(W, nm)
    t_path = 2 * LD + LM * (1 + (math.ceil(math.log2(k)) if k > 1 else 0))
    tau = LD + LM * (1 + acc)
    best = None
    for name, L, R in _tilings(m, k, n):
        F = S - R - 1
        if F < 2:
            continue
        t_dma = (L + Sv) * max(LD / nd, tau / F)
        T = max(t_mat, t_dma, t_path)
        if best is None or T < best.T - 1e-9 or (abs(T - best.T) < 1e-9 and L < best.loads):
            best = Timing(T, t_mat, t_dma, t_path, L, name, R, F)
    return best
