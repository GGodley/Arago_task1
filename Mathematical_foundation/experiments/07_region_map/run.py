#!/usr/bin/env python3
"""Experiment 07: the optimizer's region map, speed against memory, for any workload.

Every edge comes from foundation/optimizer.py (the same code the dashboard ports):

    bottom      P = 1: one matrix unit's worth
    top         P_max = W / ceil(W / ((1 + acc) m n)): the work cap with every output open
    left edge   s_min(T*): least Little's-law memory over tilings with enough independent work
    right edge  s_sat(T*): memory of the tiling with the fewest DMA units (more memory saves none)
    midline     sqrt(s_min s_sat): splits Tiny (left) from Compact (right) -- a convention
    bands       Fast from 1/3, Extremely fast from 2/3 of the way up, on a log scale -- a convention

Usage:
    python Mathematical_foundation/experiments/07_region_map/run.py
    python .../run.py --dims "512,512,512;64,4096,4096"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from foundation import LD, LM, op_counts, tilings  # noqa: E402
from foundation.optimizer import s_min, s_sat  # noqa: E402
from foundation.plotting import AQUA, BLUE, GREYS, GRID, INK, INK2, SURFACE  # noqa: E402


def region(M, K, N, steps=60):
    c = op_counts(M, K, N)
    W = c.COMPUTE + c.ACC
    acc = 1 if c.k > 1 else 0
    pmax = W / math.ceil(W / ((1 + acc) * c.m * c.n))
    ts = tilings(M, K, N)
    rows = []
    for p in np.geomspace(1, pmax, steps):
        T = 2 * LD + math.ceil(W / p) * LM
        a, b = s_min(M, K, N, T, ts), s_sat(M, K, N, T, ts)
        if a and b:
            rows.append((W / (T - 2 * LD), a, max(a, b)))
    return pmax, np.array(rows)


def draw(ax, M, K, N):
    pmax, r = region(M, K, N)
    P, A, B = r[:, 0], r[:, 1], r[:, 2]
    G = np.sqrt(A * B)
    p13, p23 = pmax ** (1 / 3), pmax ** (2 / 3)
    hi = P >= p13
    ax.fill_betweenx(P, A, G, where=hi, color=BLUE, alpha=.22, lw=0, label="Tiny")
    ax.fill_betweenx(P, G, B, where=hi, color=AQUA, alpha=.22, lw=0, label="Compact")
    ax.fill_betweenx(P, A, B, where=P <= p13, color=GREYS[1], alpha=.3, lw=0, label="Relaxed (not offered)")
    ax.plot(A, P, color=INK, lw=1.8)
    ax.plot(B, P, color=INK, lw=1.2, ls="--")
    ax.plot(G, P, color=INK2, lw=1, ls=":")
    for y, lab in ((p13, "Fast"), (p23, "Extremely fast")):
        ax.axhline(y, color=INK2, lw=.8)
        ax.text(0.99, y * 1.12, lab, transform=ax.get_yaxis_transform(), ha="right", color=INK, fontsize=10)
    ax.axhline(pmax, color=INK2, lw=.8, ls="-.")
    ax.text(0.01, pmax * 1.12, "work cap: every output open", transform=ax.get_yaxis_transform(), color=INK2, fontsize=8.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlim(A.min() / 2, B.max() * 3)
    ax.set_ylim(0.8, pmax * 1.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=.6)
    ax.set_xlabel("local memory S (tiles)", color=INK2)
    ax.set_ylabel("speed P: matrix units' worth = W / (T − 8)", color=INK2)
    ax.set_title(f"{M}×{K}×{N}", loc="left", color=INK, fontsize=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="512,512,512;4096,4096,4096")
    a = ap.parse_args()
    dims = [tuple(int(x) for x in d.split(",")) for d in a.dims.split(";")]
    fig, axes = plt.subplots(1, len(dims), figsize=(6.5 * len(dims), 6.4), dpi=150, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dims):
        draw(ax, *d)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.86), frameon=False, fontsize=9)
    fig.suptitle("Where a workload can run: each edge computed from its own counts", x=.04, ha="left", color=INK, fontsize=14)
    fig.text(.04, .875, "Solid: least memory for each speed (smallest tiling with enough work, Little's law). Dashed: memory of the tiling\n"
             "with the fewest DMA units. Dotted: their geometric middle. Bands at 1/3 and 2/3 of the log speed range.",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .85))
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "region_map.png", facecolor=SURFACE)
    print("wrote", out / "region_map.png")


if __name__ == "__main__":
    main()
