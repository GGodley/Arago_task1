"""Optimizer: the cheapest (Nd, Nm, S) for a workload, a speed and a memory size.

Speed is stated as "matrix units' worth" P: the job must run as fast as P matrix
units kept busy, so the time target is

    T* = 2 Ld + ceil(W / P) Lm

For a target T* and memory S:

    Nm = ceil(W / (T* - 2 Ld))                       set by the work alone
    each tiling t with cap_t >= Nm (independent-work cap):
        Nd_t = ceil((L_t + SAVE) Ld / (T* - 2 Ld))   set by its transfers
        M_t  = R_t + in-flight (Little's law)        must be <= S
    pick the fewest Nd, then the least memory

`grid` evaluates this over a ladder of speed steps and memory steps, which is
what the dashboard's two sliders click through.
"""

from __future__ import annotations

import math

from .counts import op_counts
from .parallelism import little_memory, tilings
from .profile import profile
from .timing import LD, LM


def _work(M, K, N):
    c = op_counts(M, K, N)
    return c, c.COMPUTE + c.ACC


def plan(M: int, K: int, N: int, target: float, S: float, _ts=None):
    """Fewest units that finish within `target` ticks in S tiles; None if impossible."""
    c, W = _work(M, K, N)
    win = target - 2 * LD
    if win <= 0:
        return None
    nm = math.ceil(W / win / LM - 1e-9)
    best = None
    for t in (_ts or tilings(M, K, N)):
        if t.cap < nm:
            continue
        nd = math.ceil((t.loads + c.SAVE) * LD / win - 1e-9)
        mem = little_memory(t, c, win)
        if mem <= S and (best is None or (nd, mem) < (best[0], best[2])):
            best = (nd, t, mem)
    if best is None:
        return None
    nd, t, mem = best
    T = 2 * LD + max(math.ceil(W / nm) * LM, (t.loads + c.SAVE) * LD / nd)
    return {"Nd": nd, "Nm": nm, "tiling": t.name, "mem_needed": mem, "loads": t.loads, "T_plan": T}


def s_min(M, K, N, target, _ts=None) -> float | None:
    """Least memory that can meet the target (over tilings with enough work)."""
    c, W = _work(M, K, N)
    win = target - 2 * LD
    nm = math.ceil(W / win / LM - 1e-9)
    ms = [little_memory(t, c, win) for t in (_ts or tilings(M, K, N)) if t.cap >= nm]
    return min(ms) if ms else None


def s_sat(M, K, N, target, _ts=None) -> float | None:
    """Memory beyond which more memory saves no DMA units at this target."""
    c, W = _work(M, K, N)
    win = target - 2 * LD
    nm = math.ceil(W / win / LM - 1e-9)
    best = None
    for t in (_ts or tilings(M, K, N)):
        if t.cap < nm:
            continue
        nd = math.ceil((t.loads + c.SAVE) * LD / win - 1e-9)
        mem = little_memory(t, c, win)
        if best is None or (nd, mem) < best:
            best = (nd, mem)
    return best[1] if best else None


def speed_steps(M, K, N, count=8):
    """Geometric ladder of P from 1 matrix unit's worth to the fastest the work allows."""
    c, W = _work(M, K, N)
    acc = 1 if c.k > 1 else 0
    pmax = W / math.ceil(W / ((1 + acc) * c.m * c.n))
    steps = sorted({max(1, round(pmax ** (i / (count - 1)))) for i in range(count)})
    out, seen = [], set()
    for p in steps:
        T = math.ceil(W / p)  # two speeds with the same time target are one step
        if p <= max(1, round(pmax)) and T not in seen:
            seen.add(T)
            out.append(p)
    return out


def grid(M: int, K: int, N: int, name: str = "", points: int = 90) -> dict:
    """Every (speed step, memory step) cell: hardware, time and the three curves."""
    c, W = _work(M, K, N)
    ts = tilings(M, K, N)
    speeds = speed_steps(M, K, N)
    targets = [2 * LD + math.ceil(W / p) * LM for p in speeds]
    smins = [s_min(M, K, N, T, ts) for T in targets]
    ssats = [s_sat(M, K, N, T, ts) for T in targets]
    top = max(s for s in ssats if s)
    mems = [2 ** e for e in range(2, 15) if 2 ** e < top * 2] or [4]
    if mems[-1] < top:
        mems.append(mems[-1] * 2)
    cells = []
    for i, T in enumerate(targets):
        row = []
        for S in mems:
            p = plan(M, K, N, T, S, ts)
            if p is None:
                row.append(None)
                continue
            prof = profile(M, K, N, p["Nd"], p["Nm"], S, points=points)
            idx = [round(j * (len(prof["t"]) - 1) / (points - 1)) for j in range(points)]
            row.append({
                "Nd": p["Nd"], "Nm": p["Nm"], "tiling": p["tiling"],
                "T": round(prof["T"], 1), "T_plan": round(p["T_plan"], 1), "limiter": prof["limiter"],
                "util": round(100 * prof["util"], 1), "dma_util": round(100 * min(1, prof["dma_util"]), 1),
                "mem_needed_pct": round(100 * p["mem_needed"] / S, 1),
                "mem_peak": round(prof["mem_peak_pct"], 1), "mem_mean": round(prof["mem_mean_pct"], 1),
                "t": [round(float(prof["t"][j]), 2) for j in idx],
                "mat": [round(float(prof["mat_pct"][j]), 1) for j in idx],
                "dma": [round(float(prof["dma_pct"][j]), 1) for j in idx],
                "mem": [round(float(prof["mem_pct"][j]), 1) for j in idx],
            })
        cells.append(row)
    return {
        "name": name or f"{M}x{K}x{N}", "M": M, "K": K, "N": N,
        "m": c.m, "k": c.k, "n": c.n, "W": W, "COMPUTE": c.COMPUTE, "ACC": c.ACC, "SAVE": c.SAVE,
        "LOAD_min": c.LOAD, "speeds": speeds, "targets": targets, "mems": mems,
        "s_min": [round(s, 1) if s else None for s in smins],
        "s_sat": [round(s, 1) if s else None for s in ssats],
        "cells": cells,
    }
