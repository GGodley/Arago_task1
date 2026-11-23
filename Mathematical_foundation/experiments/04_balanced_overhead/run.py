#!/usr/bin/env python3
"""Experiment 04: overlap overhead when the units are balanced, vs local memory.

Experiment 02 reserved a fixed overhead (Nd = Nm = 8, O = 23) at every memory
size. Here the matrix-unit count follows the balance found in experiment 03:
for a given Nd and S, Nm* is the fewest matrix units at which matrix time no
longer exceeds DMA time, i.e. adding more would not make the job faster.

The overhead feeds back on itself: Nm* sets O = Nd + 2 Nm* - 1, O shrinks the
tiling budget to S - O, that raises the loads L, which raises DMA time, which
lowers Nm*. We search Nm upward and stop at the first Nm where

    ceil(W / Nm) <= (L(S - O(Nm)) + SAVE) * Ld / Nd     (matrix no longer the bottleneck)

with Nm <= W (no point in more matrix units than matrix commands), and the DMA
slots capped at the job's transfer count (a 64³ job has 3 transfers, so at most
3 DMA units are ever busy, whatever Nd is).

Usage:
    python Mathematical_foundation/experiments/04_balanced_overhead/run.py
    python .../run.py --sizes "512;4096" --nd "4;16"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from foundation import LD, cdiv, loads_with_memory, op_counts, overlap_overhead  # noqa: E402
from foundation.plotting import AQUA, BLUE, GREYS, GRID, INK, INK2, ORANGE, SURFACE  # noqa: E402

FIXED = (8, 8)
MEMS = {64: "3;4;8;16;32;64;128;256;512", 128: "4;8;16;32;64;128;256;512", 256: "4;8;16;32;64;128;256;512",
        512: "4;8;16;32;64;128;256;512;1024", 1024: "4;8;16;32;64;128;256;512;1024;2048",
        2048: "4;8;16;32;64;128;256;512;1024;2048;4096;8192",
        4096: "4;8;16;32;64;128;256;512;1024;2048;4096;8192"}
ND_COLORS = [BLUE, AQUA]


def balanced(N: int, S: int, nd: int):
    """(Nm*, O, DMA instructions, T_dma, T_mat) or None if even Nm = 1 does not fit."""
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    best = None
    nd_busy = min(nd, c.dma_total)  # a 3-transfer job never has more than 3 DMA units busy
    for nm in range(1, W + 1):
        O = overlap_overhead(nd_busy, nm)
        L = loads_with_memory(N, N, N, S, O)
        if L is None:
            break
        t_dma, t_mat = (L + c.SAVE) * LD / nd, cdiv(W, nm)
        best = (nm, O, L + c.SAVE, t_dma, t_mat)
        if t_mat <= t_dma:
            break
    return best


def build_rows(N: int, mems: list[int], nds: list[int]) -> list[dict]:
    c = op_counts(N, N, N)
    rows = []
    for S in mems:
        L = loads_with_memory(N, N, N, S)
        Lf = loads_with_memory(N, N, N, S, overlap_overhead(*FIXED))
        row = {"N": N, "S": S, "DMA_ideal": L + c.SAVE, "DMA_fixed_8_8": None if Lf is None else Lf + c.SAVE}
        for nd in nds:
            b = balanced(N, S, nd)
            row[f"Nm_star_nd{nd}"] = b and b[0]
            row[f"O_nd{nd}"] = b and b[1]
            row[f"DMA_nd{nd}"] = b and b[2]
            row[f"T_nd{nd}"] = b and round(max(b[3], b[4]), 1)
        rows.append(row)
    return rows


def style(ax):
    ax.set_xscale("log", base=2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.minorticks_off()


def k_label(v: int) -> str:
    return f"{v // 1024}K" if v >= 1024 else str(v)


def plot(rows, nds, out: Path) -> None:
    N = rows[0]["N"]
    xs = [r["S"] for r in rows]
    fig, (top, bot) = plt.subplots(2, 1, figsize=(9, 9), dpi=150, sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for ax in (top, bot):
        style(ax)
    top.set_yscale("log")
    top.plot(xs, [r["DMA_ideal"] for r in rows], color=ORANGE, lw=2.6, marker="o", ms=4.5,
             label="no overhead (ideal tiling)")
    pf = [(r["S"], r["DMA_fixed_8_8"]) for r in rows if r["DMA_fixed_8_8"]]
    if pf:
        top.plot(*zip(*pf), color=GREYS[0], lw=1.8, ls="--", label="fixed overhead, Nd = 8, Nm = 8 (O = 23)")
    for nd, col in zip(nds, ND_COLORS):
        p = [(r["S"], r[f"DMA_nd{nd}"]) for r in rows if r[f"DMA_nd{nd}"]]
        if p:
            top.plot(*zip(*p), color=col, lw=2.2, marker="o", ms=5, mfc="white",
                     label=f"balanced: Nd = {nd}, Nm = Nm*(S)")
    top.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    top.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    top.set_ylabel("DMA instructions (log scale)", color=INK2, fontsize=11)
    top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    flat = rows[0]["DMA_ideal"] == rows[-1]["DMA_ideal"]
    title = (f"{N}³: one matrix unit is enough, so the overhead is just the DMA slots" if flat else
             f"{N}³: balanced overhead grows with memory, but its cost in DMA shrinks")
    top.set_title(title, loc="left", color=INK, fontsize=13, pad=26)
    top.text(0, 1.02, "DMA instructions vs local memory, with slots reserved for overlap: O = Nd + 2·Nm − 1",
             transform=top.transAxes, color=INK2, fontsize=10)

    bot.axhline(overlap_overhead(*FIXED), color=GREYS[0], lw=1.6, ls="--")
    for nd, col in zip(nds, ND_COLORS):
        p = [(r["S"], r[f"O_nd{nd}"], r[f"Nm_star_nd{nd}"]) for r in rows if r[f"O_nd{nd}"]]
        if not p:
            continue
        bot.plot([a for a, _, _ in p], [b for _, b, _ in p], color=col, lw=2.2, marker="o", ms=5, mfc="white")
        for i, (s, o, nm) in enumerate(p):
            if i % 2 == len(p) % 2 or i == len(p) - 1:
                bot.annotate(f"Nm={nm}\n{100 * o / s:.0f}% of S", (s, o), xytext=(0, 8), textcoords="offset points", ha="center",
                             fontsize=8, color=INK2)
    bot.set_yscale("log")
    bot.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    bot.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    bot.margins(y=0.25)
    bot.set_ylabel("overhead O (slots, log scale)", color=INK2, fontsize=11)
    bot.set_xlabel("Local memory S (tiles, log scale)", color=INK2, fontsize=11)
    bot.set_xticks(xs, [k_label(x) for x in xs])
    bot.text(0.01, 0.95, "Overhead reserved at each memory size (dashed: the fixed 23)", transform=bot.transAxes,
             color=INK2, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64;128;256;512;1024;2048;4096")
    ap.add_argument("--nd", default="4;16", help="DMA unit counts, one curve each (max 2)")
    ap.add_argument("--out", default=str(HERE / "outputs"))
    a = ap.parse_args()
    nds = [int(x) for x in a.nd.split(";")][:2]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for N in [int(s) for s in a.sizes.split(";")]:
        mems = [int(s) for s in MEMS.get(N, MEMS[4096]).split(";")]
        rows = build_rows(N, mems, nds)
        stem = f"balanced_overhead_{N}"
        plot(rows, nds, out / f"{stem}.png")
        with open(out / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", stem)


if __name__ == "__main__":
    main()
