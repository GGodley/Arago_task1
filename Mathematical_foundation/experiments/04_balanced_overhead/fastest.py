#!/usr/bin/env python3
"""Experiment 04b: the fastest setup at each memory size, overhead included.

For each local memory S, search every (Nd, Nm) and keep the one with the lowest
time, where the overlap overhead O = Nd + 2 Nm - 1 is taken out of memory first:

    L      = loads_with_memory(S - O)
    T      = 2 Ld + max(ceil(W / Nm) Lm, (L + SAVE) Ld / Nd)
    ties   -> fewest units (Nd + Nm)

No cost is put on units, so the only thing that stops "more units" is memory:
every unit takes slots, and slots taken from reuse mean more loads.

Usage:
    python Mathematical_foundation/experiments/04_balanced_overhead/fastest.py
    python .../fastest.py --size 1024 --mem "16;32;...;2048"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from foundation import LD, LM, loads_with_memory, op_counts  # noqa: E402
from foundation.plotting import AQUA, BLUE, GREYS, INK, INK2, ORANGE, SURFACE  # noqa: E402
from run import k_label, style  # noqa: E402


def fastest(N: int, mems: list[int]) -> list[dict]:
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    Smax = max(mems)
    Lb = np.full(Smax + 1, np.inf)  # loads for every tiling budget B
    for B in range(3, Smax + 1):
        L = loads_with_memory(N, N, N, B)
        if L is not None:
            Lb[B] = L
    rows = []
    for S in mems:
        best = None
        for nd in range(1, S):
            nm = np.arange(1, (S - nd) // 2 + 2)
            B = S - (nd + 2 * nm - 1)
            ok = (B >= 3) & np.isfinite(Lb[np.clip(B, 0, Smax)])
            if not ok.any():
                break
            nm, B = nm[ok], B[ok]
            T = 2 * LD + np.maximum(np.ceil(W / nm) * LM, (Lb[B] + c.SAVE) * LD / nd)
            i = int(np.argmin(T + 1e-9 * nm))
            cand = (float(T[i]), nd + int(nm[i]), nd, int(nm[i]))
            if best is None or cand < best:
                best = cand
        T, _, nd, nm = best
        O = nd + 2 * nm - 1
        rows.append({"N": N, "S": S, "Nd": nd, "Nm": nm, "O": O, "O_pct_of_S": round(100 * O / S, 1),
                     "DMA_fastest": int(Lb[S - O] + c.SAVE), "DMA_ideal": loads_with_memory(N, N, N, S) + c.SAVE,
                     "T_ticks": int(T)})
    return rows


def fmt(v, _=None):
    return f"{v / 1000:,.0f}K" if v >= 10000 else f"{v:,.0f}"


def plot(rows, out: Path) -> None:
    N = rows[0]["N"]
    xs = [r["S"] for r in rows]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 12), dpi=150, sharex=True, facecolor=SURFACE)
    for ax in (a1, a2, a3):
        style(ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))

    a1.plot(xs, [r["T_ticks"] for r in rows], color=BLUE, lw=2.6, marker="o", ms=5)
    for r in rows:
        a1.annotate(fmt(r["T_ticks"]), (r["S"], r["T_ticks"]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
    a1.set_ylabel("fastest time T (ticks)", color=INK2, fontsize=11)
    a1.set_title(f"{N}³ at its fastest: time halves every time memory doubles, and ~90% of memory goes to overlap",
                 loc="left", color=INK, fontsize=12, pad=26)
    a1.text(0, 1.03, "Best (Nd, Nm) at each memory size, overlap overhead O = Nd + 2·Nm − 1 included, no cost on units",
            transform=a1.transAxes, color=INK2, fontsize=9.5)

    a2.plot(xs, [r["S"] for r in rows], color=GREYS[1], lw=1.2, ls=":", label="all of memory (O = S)")
    a2.plot(xs, [r["O"] for r in rows], color=BLUE, lw=2.6, marker="o", ms=5, label="overhead O (slots)")
    a2.plot(xs, [r["Nm"] for r in rows], color=AQUA, lw=2, marker="s", ms=4, label="matrix units Nm")
    a2.plot(xs, [r["Nd"] for r in rows], color=GREYS[2], lw=2, marker="^", ms=4, label="DMA units Nd")
    for r in rows:
        a2.annotate(f"{r['O_pct_of_S']:.0f}%", (r["S"], r["O"]), xytext=(-4, 7), textcoords="offset points",
                    ha="right", fontsize=8, color=INK2)
    a2.set_ylabel("units and slots", color=INK2, fontsize=11)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    a2.text(0.99, 0.04, "labels: overhead as % of memory", transform=a2.transAxes, ha="right", fontsize=8.5, color=INK2)

    a3.plot(xs, [r["DMA_ideal"] for r in rows], color=ORANGE, lw=2.4, marker="o", ms=4.5,
            label="no overhead (all memory used for reuse)")
    a3.plot(xs, [r["DMA_fastest"] for r in rows], color=BLUE, lw=2.4, marker="o", ms=5, mfc="white",
            label="fastest setup (memory left after overhead)")
    a3.set_ylabel("DMA instructions", color=INK2, fontsize=11)
    a3.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    a3.set_xlabel("Local memory S (tiles, log scale)", color=INK2, fontsize=11)
    a3.set_xticks(xs, [k_label(x) for x in xs])
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--mem", default="16;32;64;128;256;512;1024;2048;4096;8192")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    rows = fastest(a.size, [int(s) for s in a.mem.split(";")])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"fastest_{a.size}"
    plot(rows, out / f"{stem}.png")
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
