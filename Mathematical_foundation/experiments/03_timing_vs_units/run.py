#!/usr/bin/env python3
"""Experiment 03: predicted time over (DMA units, matrix units), one panel per size.

Each cell is the model's time T (foundation.predict_time) for Nd DMA units and
Nm matrix units at a fixed local memory S. Thin contours join equal times; thick
lines split the map by which term sets T (matrix, DMA, or dependency path). The
star is the cheapest setup (fewest units, Nd + Nm) within 5% of the best time
on the grid.

Usage:
    python Mathematical_foundation/experiments/03_timing_vs_units/run.py
    python .../run.py --sizes "64;512;4096" --mem 256 --max-units 32
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

from foundation import predict_time  # noqa: E402
from foundation.plotting import INK, INK2, SURFACE, save  # noqa: E402

# Sequential blue ramp (light = fast, dark = slow).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = mcolors.LinearSegmentedColormap.from_list("blue_seq", RAMP)
LIMIT_CODE = {"matrix": 0, "dma": 1, "path": 2}
LIMIT_TEXT = {"matrix": "matrix-bound", "dma": "DMA-bound", "path": "path-bound"}
NEAR_BEST = 1.05


def grid(N: int, S: int, units: list[int]):
    T = np.full((len(units), len(units)), np.nan)
    lim = np.full_like(T, np.nan)
    rows = []
    for i, nm in enumerate(units):
        for j, nd in enumerate(units):
            t = predict_time(N, N, N, nd, nm, S)
            if t is None:
                continue
            T[i, j], lim[i, j] = t.T, LIMIT_CODE[t.limiter]
            rows.append({"N": N, "S": S, "Nd": nd, "Nm": nm, "T_ticks": round(t.T, 2), "T_mat": t.t_mat,
                         "T_dma": round(t.t_dma, 2), "T_path": t.t_path, "limiter": t.limiter,
                         "loads": t.loads, "tiling": t.strategy})
    return T, lim, rows


def cheapest_near_best(T, units):
    tmin = np.nanmin(T)
    best = None
    for i, nm in enumerate(units):
        for j, nd in enumerate(units):
            if T[i, j] <= NEAR_BEST * tmin and (best is None or nd + nm < best[0] + best[1]):
                best = (nd, nm, T[i, j])
    return best


def fmt_ticks(t: float) -> str:
    return f"{t:,.0f}" if t >= 100 else f"{t:.0f}"


def panel(ax, fig, N, S, units):
    T, lim, rows = grid(N, S, units)
    x = np.arange(len(units))
    lo, hi = np.nanmin(T), np.nanmax(T)
    norm = mcolors.LogNorm(vmin=lo, vmax=hi) if hi > lo * 1.01 else mcolors.Normalize(vmin=lo * 0.9, vmax=lo * 1.1)
    im = ax.imshow(T, origin="lower", cmap=CMAP, norm=norm, aspect="equal", interpolation="nearest")
    # equal-time contours
    if hi > lo * 1.05:
        levels = np.unique(np.geomspace(lo * 1.1, hi * 0.95, 6).round(0))
        cs = ax.contour(x, x, T, levels=levels, colors="white", linewidths=0.8, alpha=0.9)
        ax.clabel(cs, fmt=fmt_ticks, fontsize=7, colors="white")
    # limiter boundaries: draw cell edges where the limiter changes
    for i in range(len(units)):
        for j in range(len(units)):
            if j + 1 < len(units) and lim[i, j] != lim[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=INK, lw=1.8)
            if i + 1 < len(units) and lim[i, j] != lim[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=INK, lw=1.8)
    # label each limiter region at its centroid
    for name, code in LIMIT_CODE.items():
        ii, jj = np.where(lim == code)
        if len(ii) >= 6 and np.ptp(jj) >= 4 and np.ptp(ii) >= 2:
            ci, cj = ii.mean(), jj.mean()
            dark = norm(np.nanmean(T[lim == code])) > 0.55
            ax.text(cj, ci, LIMIT_TEXT[name], ha="center", va="center", fontsize=9, weight="bold",
                    color="white" if dark else INK)
    nd, nm, tb = cheapest_near_best(T, units)
    ax.plot(units.index(nd), units.index(nm), marker="*", ms=15, color="#eb6834", mec="white", mew=1.2)
    ticks = [i for i, u in enumerate(units) if u == 1 or u % 4 == 0]
    ax.set_xticks(ticks, [str(units[i]) for i in ticks])
    ax.set_yticks(ticks, [str(units[i]) for i in ticks])
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("DMA units, Nd", color=INK2, fontsize=10)
    ax.set_ylabel("Matrix units, Nm", color=INK2, fontsize=10)
    ax.set_title(f"{N}³: {fmt_ticks(lo)} to {fmt_ticks(hi)} ticks", loc="left", color=INK, fontsize=11)
    ax.text(0, -0.17, f"★ Nd={nd}, Nm={nm}: {fmt_ticks(tb)} ticks, within 5% of best",
            transform=ax.transAxes, color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("time T (ticks)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8, which="both")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_ticks(v)))
    cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    cb.outline.set_visible(False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64;512;4096")
    ap.add_argument("--mem", type=int, default=256, help="local memory S, tiles")
    ap.add_argument("--max-units", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(";")]
    units = list(range(1, a.max_units + 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(sizes), figsize=(5.4 * len(sizes), 6.0), dpi=150, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, N in zip(axes, sizes):
        rows += panel(ax, fig, N, a.mem, units)
    fig.suptitle(f"Time vs number of units at S = {a.mem} tiles: small jobs are path-bound, large jobs need DMA and matrix units in balance",
                 x=0.01, ha="left", color=INK, fontsize=14)
    fig.text(0.01, 0.915, "Model time per matmul (log colour, light = fast). Thick lines split the map by "
             "which term sets T; white contours join equal times.", color=INK2, fontsize=10)
    stem = f"time_vs_units_S{a.mem}"
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(out / f"{stem}.png", facecolor=SURFACE)
    plt.close(fig)
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out / f"{stem}.png")


if __name__ == "__main__":
    main()
