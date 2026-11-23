#!/usr/bin/env python3
"""Experiment 01: instruction counts vs matrix size, grouped by unit.

Produces four graphs and a CSV in ./outputs:
    mac_instructions.png   COMPUTE, ACC and their total (matrix units)
    dma_instructions.png   LOAD, SAVE and their total (DMA units), plus the
                           total with limited local memory (reloads included)
    mac_vs_dma_line.png    MAC total and DMA total on one axis, line chart
    mac_vs_dma_bar.png     the same as grouped bars
    op_counts.csv          every count, one row per size

Usage (from the repo root or anywhere):
    python Mathematical_foundation/experiments/01_instruction_counts/run.py
    python .../run.py --sizes "64;128;256;512;1024;2048;4096" --mem "64;256"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

from foundation import loads_with_memory, op_counts  # noqa: E402
from foundation.plotting import (AQUA, BLUE, GREYS, INK, INK2, ORANGE, frame,  # noqa: E402
                                 label_end, log2_size_axis, new_figure, save)

XLABEL = "Square matrix size N (M = K = N)"
YLABEL = "Instruction count (log scale)"


def build_rows(sizes: list[int], mems: list[int]) -> list[dict]:
    rows = []
    for s in sizes:
        row = op_counts(s, s, s).as_dict()
        for S in mems:
            row[f"DMA_total_S{S}"] = loads_with_memory(s, s, s, S) + row["SAVE"]
        rows.append(row)
    return rows


def plot_mac(rows, sizes, out: Path) -> None:
    fig, ax = new_figure()
    for key, name, color, lw in (("MAC_total", "Total (COMPUTE + ACC)", BLUE, 2.6),
                                 ("COMPUTE", "COMPUTE", ORANGE, 2.0),
                                 ("ACC", "ACC", AQUA, 2.0)):
        pts = [(r["N"], r[key]) for r in rows if r[key] > 0]  # ACC is 0 when k = 1; log axis can't show it
        ax.plot(*zip(*pts), color=color, linewidth=lw, marker="o", markersize=4, label=name)
    last = rows[-1]
    label_end(ax, last["N"], last["MAC_total"], f"Total {last['MAC_total']:,}", BLUE, 8)
    label_end(ax, last["N"], last["COMPUTE"], f"COMPUTE {last['COMPUTE']:,}", ORANGE, -4)
    label_end(ax, last["N"], last["ACC"], f"ACC {last['ACC']:,}", AQUA, -14)
    ax.set_yscale("log")
    log2_size_axis(ax, sizes)
    frame(ax, "MAC instructions grow as N³: 8× per doubling",
          "Matrix-unit commands per matmul. Independent of hardware: memory and unit counts don't change them.",
          XLABEL, YLABEL)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    save(fig, out)


def plot_dma(rows, sizes, mems, out: Path) -> None:
    fig, ax = new_figure()
    xs = [r["N"] for r in rows]
    ax.plot(xs, [r["DMA_total"] for r in rows], color=BLUE, linewidth=2.6, marker="o", markersize=4,
            label="Total, no reloads (LOAD + SAVE)")
    ax.plot(xs, [r["LOAD"] for r in rows], color=ORANGE, linewidth=2.0, marker="o", markersize=4,
            label="LOAD, compulsory")
    ax.plot(xs, [r["SAVE"] for r in rows], color=AQUA, linewidth=2.0, marker="o", markersize=4, label="SAVE")
    for i, S in enumerate(mems):
        ax.plot(xs, [r[f"DMA_total_S{S}"] for r in rows], color=GREYS[i % len(GREYS)], linewidth=1.8,
                linestyle="--", label=f"Total with {S} tiles of local memory (reloads included)")
    last = rows[-1]
    label_end(ax, last["N"], last["DMA_total"], f"Total {last['DMA_total']:,}", BLUE, 6)
    label_end(ax, last["N"], last["LOAD"], f"LOAD {last['LOAD']:,}", ORANGE, -2)
    label_end(ax, last["N"], last["SAVE"], f"SAVE {last['SAVE']:,}", AQUA, -12)
    for S in mems:
        v = last[f"DMA_total_S{S}"]
        ax.annotate(f"S = {S}: {v:,}", (last["N"], v), xytext=(8, 0), textcoords="offset points",
                    color=INK2, fontsize=10, va="center")
    ax.set_yscale("log")
    log2_size_axis(ax, sizes)
    frame(ax, "DMA instructions grow as N² when everything fits, faster when it doesn't",
          "LOAD + SAVE per matmul. Solid: each tile moved once. Dashed: ideal tiling with limited local memory.",
          XLABEL, YLABEL)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    save(fig, out)


def plot_combined_line(rows, sizes, out: Path) -> None:
    fig, ax = new_figure()
    xs = [r["N"] for r in rows]
    ax.plot(xs, [r["MAC_total"] for r in rows], color=BLUE, linewidth=2.6, marker="o", markersize=5,
            label="MAC instructions (COMPUTE + ACC)")
    ax.plot(xs, [r["DMA_total"] for r in rows], color=ORANGE, linewidth=2.6, marker="o", markersize=5,
            label="DMA instructions (LOAD + SAVE, no reloads)")
    last = rows[-1]
    label_end(ax, last["N"], last["MAC_total"], f"MAC {last['MAC_total']:,}", BLUE)
    label_end(ax, last["N"], last["DMA_total"], f"DMA {last['DMA_total']:,}", ORANGE)
    for r in rows:  # first size where MAC count reaches DMA count
        if r["MAC_total"] >= r["DMA_total"]:
            text = (f"Equal at {r['N']}³ ({r['MAC_total']:,} each)" if r["MAC_total"] == r["DMA_total"]
                    else f"MAC overtakes DMA at {r['N']}³")
            ax.annotate(text, (r["N"], r["MAC_total"]), xytext=(30, -45), textcoords="offset points",
                        color=INK2, fontsize=10, ha="left", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
            break
    ax.set_yscale("log")
    log2_size_axis(ax, sizes)
    frame(ax, f"MAC work outgrows DMA work: {last['MAC_total'] / last['DMA_total']:.0f}× more at {last['N']}³",
          "Instructions per matmul. MAC grows as N³, DMA as N², so the ratio doubles with every doubling of N.",
          XLABEL, YLABEL)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    save(fig, out)


def plot_combined_bar(rows, sizes, out: Path) -> None:
    fig, ax = new_figure()
    idx = list(range(len(rows)))
    w = 0.36
    mac = [r["MAC_total"] for r in rows]
    dma = [r["DMA_total"] for r in rows]
    b1 = ax.bar([i - w / 2 - 0.01 for i in idx], mac, width=w, color=BLUE, label="MAC instructions (COMPUTE + ACC)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in idx], dma, width=w, color=ORANGE,
                label="DMA instructions (LOAD + SAVE, no reloads)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(mac) * 6)
    for bars, vals in ((b1, mac), (b2, dma)):
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:,}", (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90 if v >= 10000 else 0)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{r['N']}³" for r in rows])
    last = rows[-1]
    tie = next((r["N"] for r in rows if r["MAC_total"] >= r["DMA_total"]), None)
    frame(ax, (f"MAC catches DMA at {tie}³ and leads beyond: " if tie else "")
          + f"{last['MAC_total'] / last['DMA_total']:.0f}× more at {last['N']}³",
          "Instructions per matmul, side by side for each size. Log scale: each gridline is 10×.",
          XLABEL, YLABEL)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="64;128;256;512;1024;2048;4096", help="square sizes, ';'-separated")
    ap.add_argument("--mem", default="64;256", help="local memory sizes in tiles for the reload curves")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.replace(",", ";").split(";") if s.strip()]
    mems = [int(s) for s in a.mem.replace(",", ";").split(";") if s.strip()]
    rows = build_rows(sizes, mems)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "op_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot_mac(rows, sizes, a.out / "mac_instructions.png")
    plot_dma(rows, sizes, mems, a.out / "dma_instructions.png")
    plot_combined_line(rows, sizes, a.out / "mac_vs_dma_line.png")
    plot_combined_bar(rows, sizes, a.out / "mac_vs_dma_bar.png")
    hdr = ["N", "COMPUTE", "ACC", "MAC_total", "LOAD", "SAVE", "DMA_total"] + [f"DMA_total_S{S}" for S in mems]
    print("  ".join(f"{h:>14}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>14,}" for h in hdr))
    print(f"\nwrote {len(list(a.out.iterdir()))} files to {a.out}")


if __name__ == "__main__":
    main()
