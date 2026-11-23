#!/usr/bin/env python3
"""Experiment 02: instruction counts vs local memory size, for one matrix size.

MAC instructions and the two DMA bounds (no reuse, perfect reuse) do not depend
on memory, so they are flat lines. DMA with S tiles of memory (ideal tiling,
reloads included) falls from the no-reuse ceiling to the perfect-reuse floor,
at slope -1/2 on log-log axes (reuse ~ sqrt(S)), and flattens at S0 where a whole
panel fits.

Usage:
    python Mathematical_foundation/experiments/02_memory_vs_instructions/run.py
    python .../run.py --size 4096 --mem "4;8;16;...;8192"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

from foundation import loads_with_memory, op_counts  # noqa: E402
from foundation.plotting import BLUE, INK, INK2, ORANGE, frame, label_end, new_figure, save  # noqa: E402


def no_reload_memory(N: int) -> int:
    """S0: smallest memory with no reloads, min(m, n)*(k + acc) + 2."""
    c = op_counts(N, N, N)
    return min(c.m, c.n) * (c.k + (1 if c.k > 1 else 0)) + 2


def build_rows(N: int, mems: list[int]) -> list[dict]:
    c = op_counts(N, N, N)
    rows = []
    for S in mems:
        L = loads_with_memory(N, N, N, S)
        rows.append({"N": N, "S": S, "MAC_total": c.mac_total, "DMA_no_reuse": c.dma_no_reuse,
                     "DMA_perfect_reuse": c.dma_total, "LOAD_at_S": L, "DMA_at_S": L + c.SAVE,
                     "reuse_at_S": round(2 * c.m * c.n * c.k / L, 3)})
    return rows


def plot(rows, out: Path) -> None:
    N = rows[0]["N"]
    S0 = no_reload_memory(N)
    xs = [r["S"] for r in rows]
    fig, ax = new_figure()
    ax.fill_between(xs, [r["DMA_perfect_reuse"] for r in rows], [r["DMA_no_reuse"] for r in rows],
                    color=ORANGE, alpha=0.07, linewidth=0)
    ax.plot(xs, [r["MAC_total"] for r in rows], color=BLUE, linewidth=2.6, label="MAC instructions (COMPUTE + ACC)")
    ax.plot(xs, [r["DMA_no_reuse"] for r in rows], color=ORANGE, linewidth=1.8, linestyle="--",
            label="DMA, no reuse (ceiling)")
    ax.plot(xs, [r["DMA_perfect_reuse"] for r in rows], color=ORANGE, linewidth=1.8, linestyle=":",
            label="DMA, perfect reuse (floor)")
    ax.plot(xs, [r["DMA_at_S"] for r in rows], color=ORANGE, linewidth=2.8, marker="o", markersize=4.5,
            label="DMA with S tiles of local memory (ideal tiling)")
    ax.axvline(S0, color=INK2, linewidth=0.9, linestyle=(0, (4, 4)))
    ax.annotate(f"S₀ = {S0:,} tiles:\nno reloads\nbeyond here", (S0, (rows[0]["DMA_no_reuse"] * rows[0]["DMA_perfect_reuse"]) ** 0.5), xytext=(8, 0),
                textcoords="offset points", ha="left", color=INK2, fontsize=10)
    last = rows[-1]
    x_end = xs[-1]
    label_end(ax, x_end, last["DMA_no_reuse"], f"DMA no reuse {last['DMA_no_reuse']:,}", ORANGE, 9)
    label_end(ax, x_end, last["MAC_total"], f"MAC {last['MAC_total']:,}", BLUE, -9)
    label_end(ax, x_end, last["DMA_perfect_reuse"], f"DMA perfect reuse {last['DMA_perfect_reuse']:,}", ORANGE)
    # call out a few points on the curve
    for r in rows:
        if r["S"] in (16, 256):
            ax.annotate(f"S = {r['S']}: {r['DMA_at_S']:,}\n(reuse {r['reuse_at_S']:.0f}×)", (r["S"], r["DMA_at_S"]),
                        xytext=(10, 10), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [s for s in xs if s & (s - 1) == 0]  # powers of two only; S0 is labelled by its note
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{s:,}" if s < 1000 else f"{s // 1024}K" for s in ticks], fontsize=9)
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 9)
    first = rows[0]
    frame(ax, f"{N}³: memory takes DMA from {first['DMA_at_S'] / last['DMA_at_S']:.0f}× the floor down to the floor; MAC never moves",
          "Instructions per matmul vs local memory. DMA falls ~1/√S (reuse ≈ √S − 1) until a whole panel fits.",
          "Local memory S (tiles, log scale)", "Instruction count (log scale)")
    lo = min(r["DMA_perfect_reuse"] for r in rows)
    hi = max(r["DMA_no_reuse"] for r in rows)
    yt = [v * 10 ** e for e in range(0, 8) for v in (1, 2, 5) if lo / 1.5 <= v * 10 ** e <= hi * 1.5]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:,.0f}" for v in yt])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.8, 0.86), fontsize=9, labelcolor=INK)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=4096, help="square matrix size N (M = K = N)")
    ap.add_argument("--mem", default="4;8;16;32;64;128;256;512;1024;2048;4096;8192",
                    help="local memory sizes in tiles, ';'-separated")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    a = ap.parse_args()
    mems = sorted(int(s) for s in a.mem.replace(",", ";").split(";") if s.strip())
    S0 = no_reload_memory(a.size)
    if mems[0] < S0 < mems[-1] and S0 not in mems:
        mems = sorted(mems + [S0])  # include the knee itself
    rows = build_rows(a.size, mems)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"instructions_vs_memory_{a.size}"
    with open(a.out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(rows, a.out / f"{stem}.png")
    print(f"S0 = {no_reload_memory(a.size)} tiles")
    for r in rows:
        print(f"S={r['S']:>6}  DMA={r['DMA_at_S']:>9,}  reuse={r['reuse_at_S']:>7}  MAC={r['MAC_total']:,}")


if __name__ == "__main__":
    main()
