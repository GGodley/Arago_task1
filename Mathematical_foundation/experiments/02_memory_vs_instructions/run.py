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

from foundation import loads_with_memory, op_counts, overlap_overhead  # noqa: E402
from foundation.plotting import BLUE, INK, INK2, ORANGE, frame, label_end, new_figure, save  # noqa: E402


def no_reload_memory(N: int) -> int:
    """S0: smallest memory with no reloads, min(m, n)*(k + acc) + 2."""
    c = op_counts(N, N, N)
    return min(c.m, c.n) * (c.k + (1 if c.k > 1 else 0)) + 2


def build_rows(N: int, mems: list[int], overhead: int | None = None) -> list[dict]:
    c = op_counts(N, N, N)
    rows = []
    for S in mems:
        L = loads_with_memory(N, N, N, S)
        row = {"N": N, "S": S, "MAC_total": c.mac_total, "DMA_no_reuse": c.dma_no_reuse,
                     "DMA_perfect_reuse": c.dma_total, "LOAD_at_S": L, "DMA_at_S": L + c.SAVE,
                     "reuse_at_S": round(2 * c.m * c.n * c.k / L, 3)}
        if overhead is not None:
            Lo = loads_with_memory(N, N, N, S, overhead)
            row["overlap_overhead"] = overhead
            row["DMA_at_S_overlap"] = None if Lo is None else Lo + c.SAVE
        rows.append(row)
    return rows


def plot(rows, out: Path, nd: int | None = None, nm: int | None = None) -> None:
    N = rows[0]["N"]
    S0 = no_reload_memory(N)
    xs = [r["S"] for r in rows]
    fig, ax = new_figure()
    fig.set_size_inches(9, 6.2)
    ax.fill_between(xs, [r["DMA_perfect_reuse"] for r in rows], [r["DMA_no_reuse"] for r in rows],
                    color=ORANGE, alpha=0.07, linewidth=0)
    ax.plot(xs, [r["MAC_total"] for r in rows], color=BLUE, linewidth=2.6, label="MAC instructions (COMPUTE + ACC)")
    ax.plot(xs, [r["DMA_no_reuse"] for r in rows], color=ORANGE, linewidth=1.8, linestyle="--",
            label="DMA, no reuse (ceiling)")
    ax.plot(xs, [r["DMA_perfect_reuse"] for r in rows], color=ORANGE, linewidth=1.8, linestyle=":",
            label="DMA, perfect reuse (floor)")
    ax.plot(xs, [r["DMA_at_S"] for r in rows], color=ORANGE, linewidth=2.8, marker="o", markersize=4.5,
            label="DMA with S tiles of local memory (ideal tiling)")
    if "DMA_at_S_overlap" in rows[0]:
        O = rows[0]["overlap_overhead"]
        pts = [(r["S"], r["DMA_at_S_overlap"]) for r in rows if r["DMA_at_S_overlap"] is not None]
        if pts:
            ax.plot(*zip(*pts), color="#a8431a", linewidth=2.2, linestyle="-.", marker="o", markersize=4.5,
                    markerfacecolor="white", label=f"DMA with S tiles, after {O} slots of overlap overhead (Nd={nd}, Nm={nm})")
            S0o = S0 + O
            if xs[0] < S0o <= xs[-1] and S0o != S0:
                ax.axvline(S0o, color="#a8431a", linewidth=0.9, linestyle=(0, (2, 3)))
    if xs[0] < S0 <= xs[-1]:
        ax.axvline(S0, color=INK2, linewidth=0.9, linestyle=(0, (4, 4)))
        has_o = "DMA_at_S_overlap" in rows[0]
        note_x = S0 + rows[0]["overlap_overhead"] if has_o else S0
        note_y = rows[0]["DMA_no_reuse"] ** (0.3 if has_o else 0.5) * rows[0]["DMA_perfect_reuse"] ** (0.7 if has_o else 0.5)
        note = f"S\u2080 = {S0:,} tiles:\nno reloads\nbeyond here"
        if has_o:
            note += f"\n(S\u2080 + {rows[0]['overlap_overhead']} = {note_x:,}\nwith overlap)"
        ax.annotate(note, (note_x, note_y), xytext=(8, 0),
                    textcoords="offset points", ha="left", color=INK2, fontsize=10)
    last = rows[-1]
    x_end = xs[-1]
    label_end(ax, x_end, last["DMA_no_reuse"], f"DMA no reuse {last['DMA_no_reuse']:,}", ORANGE, 9)
    label_end(ax, x_end, last["MAC_total"], f"MAC {last['MAC_total']:,}", BLUE, -9)
    label_end(ax, x_end, last["DMA_perfect_reuse"], f"DMA perfect reuse {last['DMA_perfect_reuse']:,}", ORANGE)
    # call out a few points on the curve
    for r in rows:
        if r["S"] in (16, 256) and r["DMA_at_S"] > r["DMA_perfect_reuse"] and r["S"] < S0:
            ax.annotate(f"S = {r['S']}: {r['DMA_at_S']:,}\n(reuse {r['reuse_at_S']:.0f}×)", (r["S"], r["DMA_at_S"]),
                        xytext=(-8, -30), textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = [s for s in xs if s & (s - 1) == 0]  # powers of two only; S0 is labelled by its note
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{s:,}" if s < 1000 else f"{s // 1024}K" for s in ticks], fontsize=9)
    ax.set_xlim(xs[0] / 1.3, xs[-1] * 9)
    first = rows[0]
    swing = first["DMA_at_S"] / last["DMA_at_S"]
    if swing < 1.01:
        title = (f"{N}\u00b3: nothing to reuse, so memory changes nothing "
                 f"(DMA {last['DMA_at_S']:,}, MAC {last['MAC_total']:,} at every memory size)")
        sub = (f"Instructions per matmul vs local memory. The job needs {S0} slots at most; "
               "every tile is used once, so extra memory sits idle.")
    else:
        title = f"{N}\u00b3: memory takes DMA from {swing:.0f}\u00d7 the floor down to the floor; MAC never moves"
        sub = "Instructions per matmul vs local memory. DMA falls ~1/\u221aS (reuse \u2248 \u221aS \u2212 1) until a whole panel fits."
    frame(ax, title, sub, "Local memory S (tiles, log scale)", "Instruction count (log scale)")
    lo = min(min(r["DMA_perfect_reuse"], r["MAC_total"]) for r in rows)
    hi = max(max(r["DMA_no_reuse"], r["MAC_total"]) for r in rows)
    ax.set_ylim(lo / 1.6, hi * 1.6)
    yt = [v * 10 ** e for e in range(0, 8) for v in (1, 2, 5) if lo / 1.5 <= v * 10 ** e <= hi * 1.5]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:,.0f}" for v in yt])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, fontsize=9, labelcolor=INK)
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=4096, help="square matrix size N (M = K = N)")
    ap.add_argument("--mem", default="4;8;16;32;64;128;256;512;1024;2048;4096;8192",
                    help="local memory sizes in tiles, ';'-separated")
    ap.add_argument("--nd", type=int, help="DMA units; with --nm, adds the overlap-overhead curve")
    ap.add_argument("--nm", type=int, help="matrix units; with --nd, adds the overlap-overhead curve")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    a = ap.parse_args()
    overhead = overlap_overhead(a.nd, a.nm) if a.nd and a.nm else None
    mems = sorted(int(s) for s in a.mem.replace(",", ";").split(";") if s.strip())
    S0 = no_reload_memory(a.size)
    knees = [S0] + ([S0 + overhead, overhead + 3 + (1 if a.size > 64 else 0)] if overhead is not None else [])
    for knee in knees:
        if mems[0] < knee < mems[-1] and knee not in mems:
            mems = sorted(mems + [knee])  # include the knees themselves
    rows = build_rows(a.size, mems, overhead)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"instructions_vs_memory_{a.size}" + (f"_overlap_nd{a.nd}_nm{a.nm}" if overhead is not None else "")
    with open(a.out / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    plot(rows, a.out / f"{stem}.png", a.nd, a.nm)
    print(f"S0 = {no_reload_memory(a.size)} tiles")
    for r in rows:
        extra = ""
        if overhead is not None:
            v = r["DMA_at_S_overlap"]
            extra = f"  DMA(overlap)={'infeasible' if v is None else f'{v:,}':>10}"
        print(f"S={r['S']:>6}  DMA={r['DMA_at_S']:>9,}  reuse={r['reuse_at_S']:>7}  MAC={r['MAC_total']:,}{extra}")


if __name__ == "__main__":
    main()
