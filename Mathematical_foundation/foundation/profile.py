"""Step 5: the model's outputs over time (matrix usage, DMA usage, local memory).

Built on the steady-state overlap assumption (quasi-optimal scheduling):

    fill    [0, t_fill]        DMA loads the first resident set; matrix units start after Ld
    steady  [t_fill, T - Ld]   constant rates; memory = resident + Little's-law in-flight terms
    drain   [T - Ld, T]        the last sums are saved; memory falls to zero

Two memory curves come out:
    needed    resident + in-flight: the least memory the schedule needs (what the
              optimizer sizes S against)
    occupied  what a scheduler that prefetches into every free slot holds:
              min(S, needed + (1 - DMA busy) x loads still to come): it runs ahead
              when the DMA has spare time and falls as the remaining loads run out

T, the tiling and its loads come from foundation.predict_time. Rates are set so
the area under each usage curve equals the work counted in Step 1.
"""

from __future__ import annotations

import math
import re

import numpy as np

from .counts import cdiv, tiles
from .timing import LD, LM, predict_time


def _tiling_shape(name: str, m: int, k: int, n: int, acc: int):
    """(resident tiles, output groups, sums per group, loads before steady state)."""
    if name.startswith("block"):
        bm, bn = map(int, re.findall(r"\d+", name))
        return acc * bm * bn, cdiv(m, bm) * cdiv(n, bn), bm * bn, bm + bn
    p = int(re.findall(r"\d+", name)[0])
    if name.startswith("Apanel"):
        return p * k + acc * p, cdiv(m, p) * n, p, p * k + k
    return p * k + acc * p, cdiv(n, p) * m, p, p * k + k


def profile(M: int, K: int, N: int, nd: int, nm: int, S: int, points: int = 1500) -> dict:
    t = predict_time(M, K, N, nd, nm, S)
    if t is None:
        return None
    m, k, n = tiles(M, K, N)
    acc = 1 if k > 1 else 0
    W, COMPUTE, SAVE, L = m * n * (2 * k - 1), m * n * k, m * n, t.loads
    T = t.T
    tau = LD + LM * (1 + acc)
    nd_eff = max(1.0, min(nd, t.F * LD / tau))
    resident, _groups, _G, fill_loads = _tiling_shape(t.strategy, m, k, n, acc)
    fill_loads = min(fill_loads, L)
    t_fill = min(max(LD, LD * math.ceil(fill_loads / nd_eff)), max(LD, T - LD))
    Ts = max(1.0, T - LD - t_fill)

    # Little's law: slots = arrival rate x time each tile holds its slot
    lam_load = max(0, L - fill_loads) / Ts
    lam_comp = COMPUTE / max(1.0, T - 2 * LD)
    lam_save = SAVE / Ts
    inflight = lam_load * tau + min(lam_comp * 2 * LM, 2 * nm) + lam_save * LD
    cap = min(S, m * k + k * n + m * n)
    base = min(cap, resident + inflight)

    ts = np.linspace(0, T, points)
    mem = np.empty_like(ts)
    ramp_start = min(nd, fill_loads)
    f = ts <= t_fill
    mem[f] = ramp_start + (base - ramp_start) * ts[f] / t_fill
    s = (ts > t_fill) & (ts < T - LD)
    mem[s] = base
    d = ts >= T - LD
    mem[d] = base * (T - ts[d]) / LD
    mem = np.minimum(mem, cap)

    # prefetching scheduler: free slots fill with loads that are still to come, but only
    # as fast as spare DMA time allows (a DMA-bound job has nothing spare to run ahead with)
    dma_busy = min(1.0, (L + SAVE) * LD / (nd * T))
    loads_left = np.clip(L * (1 - (ts - LD) / max(1.0, T - 2 * LD)), 0, L)
    lead = (1 - dma_busy) * loads_left
    occ = np.minimum(cap, np.maximum(mem, np.where(ts < T - LD, base + lead, mem)))
    occ = np.minimum(occ, nd + (cap - nd) * np.minimum(1, ts / max(1.0, t_fill)) if t_fill > 0 else occ)

    # usage: matrix busy only between the first landed tile and the last save
    mat = np.where((ts >= LD) & (ts <= T - LD), W * LM / max(1.0, T - 2 * LD), 0.0)
    dma_fill = nd_eff
    rest = (L + SAVE) * LD - dma_fill * t_fill - min(nd, SAVE) * LD
    dma = np.where(f, dma_fill, np.where(d, min(nd, SAVE), max(0.0, rest) / Ts))
    return {"T": T, "cycles": T * 128, "limiter": t.limiter, "tiling": t.strategy,
            "util": W * LM / (nm * T), "dma_util": (L + SAVE) * LD / (nd * T),
            "mem_peak_pct": 100 * mem.max() / S, "mem_mean_pct": 100 * mem.mean() / S,
            "occ_peak_pct": 100 * occ.max() / S, "occ_mean_pct": 100 * occ.mean() / S,
            "t": ts, "mem_tiles": mem, "mem_pct": 100 * mem / S, "occ_pct": 100 * occ / S,
            "mat_pct": 100 * np.minimum(mat, nm) / nm, "dma_pct": 100 * np.minimum(dma, nd) / nd,
            "t_fill": t_fill, "base": base, "resident": resident, "inflight": inflight}
