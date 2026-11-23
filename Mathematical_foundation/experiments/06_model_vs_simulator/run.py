#!/usr/bin/env python3
"""Experiment 06: is the model right? Model outputs against the cycle-level simulator.

For each example the model (foundation.profile, built on predict_time) gives the
three outputs Task 1 asks for, and the simulator (matmul_sim, memory-aware blocked
scheduler) runs the same trace on the same hardware:

    total cycles               one number each
    matrix-unit usage          % of Nm busy, over time
    local memory used          % of S, over time (model: "needed" and "occupied")

Outputs:
    outputs/timeline_<name>.png   three stacked panels, simulator solid, model dashed
    outputs/summary.csv           the headline numbers side by side
    outputs/cycles_validity.png   model vs simulator cycles over the 1,787-run sweep
                                  (needs compare.csv from the sweep; skipped if missing)

Usage:
    python Mathematical_foundation/experiments/06_model_vs_simulator/run.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for matmul_sim

import matplotlib.pyplot as plt  # noqa: E402

from foundation.plotting import AQUA, BLUE, GREYS, GRID, INK, INK2, ORANGE, SURFACE, save  # noqa: E402
from foundation.profile import profile  # noqa: E402
from matmul_sim import HWConfig  # noqa: E402
from matmul_sim.schedulers import make_scheduler  # noqa: E402
from matmul_sim.simulator import simulate  # noqa: E402
from matmul_sim.translator import MatmulSpec, translate  # noqa: E402

OUT = HERE / "outputs"
COMPARE = Path("/mnt/project-files/timing_model/compare.csv")

# name, (M, K, N), (Nd, Nm, S), what it stands for
EXAMPLES = [
    ("small_128", (128, 128, 128), (2, 2, 16), "small: one 2x2 block of tiles"),
    ("medium_512", (512, 512, 512), (8, 4, 128), "medium: 8x8x8 tiles"),
    ("large_2048", (2048, 2048, 2048), (16, 32, 512), "large: 32x32x32 tiles"),
    ("decode_64x4096x4096", (64, 4096, 4096), (16, 8, 256),
     "LLM decode: 64 tokens through a 4096x4096 weight"),
]


def sim_curves(res, nd: int, nm: int, S: int):
    """Busy units per tick (matrix, DMA) and memory per tick from a simulator run."""
    T = res.total_ticks
    mat, dma = np.zeros(T + 1), np.zeros(T + 1)
    for c in res.schedule:
        arr = mat if c.kind in ("COMPUTE", "ACC") else dma
        arr[c.start] += 1
        arr[c.end] -= 1
    mat, dma = np.cumsum(mat)[:T], np.cumsum(dma)[:T]
    mem = np.zeros(T)
    ms = res.mem_series
    for i, (t, x) in enumerate(ms):
        e = ms[i + 1][0] if i + 1 < len(ms) else T
        mem[t:e] = x
    return np.arange(T) + 0.5, 100 * mat / nm, 100 * dma / nd, 100 * mem / S


def smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(np.pad(y, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def style(ax, ylabel):
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_ylim(0, 108)


def timeline(name, dims, hw, note, res, p):
    nd, nm, S = hw
    t, mat, dma, mem = sim_curves(res, nd, nm, S)
    w = max(1, len(t) // 60)  # smooth the simulator's tick-level jitter
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.6), dpi=150, sharex=True)
    M, K, N = dims
    fig.suptitle(f"{M}x{K}x{N} on Nd={nd}, Nm={nm}, S={S} tiles: model {p['T']:,.0f} ticks, "
                 f"simulator {res.total_ticks:,} ticks", x=0.06, ha="left", color=INK, fontsize=13)
    fig.text(0.06, 0.935, f"{note}. Simulator solid, model dashed. 1 tick = 128 cycles.",
             color=INK2, fontsize=10)

    ax = axes[0]
    ax.plot(t, smooth(mat, w), color=BLUE, lw=1.6, label=f"simulator, avg {100*res.matrix_util:.0f}%")
    ax.plot(p["t"], p["mat_pct"], color=INK, lw=1.4, ls="--", label=f"model, avg {100*p['util']:.0f}%")
    style(ax, "matrix units busy (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.99), frameon=False, fontsize=9, ncol=2)

    ax = axes[1]
    ax.plot(t, smooth(dma, w), color=ORANGE, lw=1.6, label=f"simulator, avg {100*res.dma_util:.0f}%")
    ax.plot(p["t"], p["dma_pct"], color=INK, lw=1.4, ls="--", label=f"model, avg {100*p['dma_util']:.0f}%")
    style(ax, "DMA units busy (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.99), frameon=False, fontsize=9, ncol=2)

    ax = axes[2]
    ax.plot(t, mem, color=AQUA, lw=1.6, label=f"simulator, avg {res.mem_mean_pct:.0f}%")
    ax.plot(p["t"], p["mem_pct"], color=INK, lw=1.4, ls="--",
            label=f"model needed, avg {p['mem_mean_pct']:.0f}%")
    ax.plot(p["t"], p["occ_pct"], color=GREYS[2], lw=1.4, ls=":",
            label=f"model with prefetch, avg {p['occ_mean_pct']:.0f}%")
    style(ax, "local memory used (%)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.99), frameon=False, fontsize=9, ncol=3)
    ax.set_xlabel("time (ticks)", color=INK2, fontsize=10)
    ax.set_xlim(0, max(res.total_ticks, p["T"]))

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"timeline_{name}.png", facecolor=SURFACE)
    plt.close(fig)


def validity():
    if not COMPARE.exists():
        print("compare.csv not found, skipping the validity scatter")
        return
    rows = [r for r in csv.DictReader(open(COMPARE)) if r["completed"] == "True" and r["T_model"]]
    sim = np.array([float(r["ticks"]) for r in rows])
    mod = np.array([float(r["T_model"]) for r in rows])
    S = np.array([int(r["mem"]) for r in rows])
    err = (mod - sim) / sim
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    groups = [(S <= 32, GREYS[0], "S = 4 to 32 tiles"), ((S > 32) & (S < 128), ORANGE, "S = 64"),
              (S >= 128, BLUE, "S = 128 to 512")]
    for mask, col, lab in groups:
        e = np.abs(err[mask])
        ax.scatter(sim[mask], mod[mask], s=9, color=col, alpha=0.55, lw=0,
                   label=f"{lab}: median error {100*np.median(e):.1f}%, {100*np.mean(e <= .25):.0f}% within 25%")
    lo, hi = 5, sim.max() * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1)
    ax.fill_between([lo, hi], [lo * .75, hi * .75], [lo * 1.25, hi * 1.25], color=GRID, alpha=.6, lw=0)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.set_xlabel("simulator (ticks)", color=INK2), ax.set_ylabel("model (ticks)", color=INK2)
    a = np.abs(err)
    ax.set_title(f"Model cycles track the simulator: median error {100*np.median(a):.1f}% over {len(rows):,} runs",
                 loc="left", color=INK, fontsize=12, pad=26)
    ax.text(0, 1.02, "Each dot is one workload and hardware setup. Grey band = within 25%. "
            "Points below the line: model faster.", transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, OUT / "cycles_validity.png")


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, dims, hw, note in EXAMPLES:
        M, K, N = dims
        nd, nm, S = hw
        res = simulate(translate(MatmulSpec(M, K, N)), HWConfig(nd, nm, S), make_scheduler("memaware", "blocked", 4))
        p = profile(M, K, N, nd, nm, S)
        timeline(name, dims, hw, note, res, p)
        row = dict(example=name, M=M, K=K, N=N, Nd=nd, Nm=nm, S=S, tiling=p["tiling"], limiter=p["limiter"],
                   cycles_model=int(p["cycles"]), cycles_sim=res.total_ticks * 128,
                   cycles_err_pct=round(100 * (p["T"] - res.total_ticks) / res.total_ticks, 1),
                   matrix_util_model=round(100 * p["util"], 1), matrix_util_sim=round(100 * res.matrix_util, 1),
                   dma_util_model=round(100 * p["dma_util"], 1), dma_util_sim=round(100 * res.dma_util, 1),
                   mem_needed_mean=round(p["mem_mean_pct"], 1), mem_prefetch_mean=round(p["occ_mean_pct"], 1),
                   mem_sim_mean=round(res.mem_mean_pct, 1), mem_sim_peak=round(100 * res.mem_peak / S, 1))
        rows.append(row)
        print(row)
    with open(OUT / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    validity()


if __name__ == "__main__":
    main()
