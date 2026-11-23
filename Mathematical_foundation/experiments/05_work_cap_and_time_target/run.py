#!/usr/bin/env python3
"""Experiment 05: the independent-work cap, and the cheapest setup for a time target.

Model (all in ticks, Ld = 4, Lm = 1), for a tiling t at memory S:

    O        = Nd + 2 Nm - 1                    overlap overhead (experiment 02)
    fits     : need_t + O <= S
    Nm_eff   = min(Nm, cap_t)                   independent-work cap (foundation/parallelism.py)
    T        = 2 Ld + max(ceil(W / Nm_eff) Lm, (L_t + SAVE) Ld / Nd)

Part A, fastest: at each S, the lowest T over every tiling and (Nd, Nm), units free.
Part B, time target: at each S, the fewest units (Nd + Nm) with T <= target.
    Nm = ceil(W / (target - 2 Ld)) is fixed by the target; each tiling then needs
    Nd = ceil((L_t + SAVE) Ld / (target - 2 Ld)), and is usable if cap_t >= Nm and
    it fits. More matrix units never lower Nd, they only take slots.

Usage:
    python Mathematical_foundation/experiments/05_work_cap_and_time_target/run.py
    python .../run.py --size 4096 --target 15000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # Mathematical_foundation/

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from foundation import LD, LM, in_flight_work, little_memory, op_counts, tilings  # noqa: E402
from foundation.plotting import AQUA, BLUE, GREYS, GRID, INK, INK2, ORANGE, SURFACE  # noqa: E402

MEMS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


OVERHEAD = "little"  # "little" (Little's law, default) or "fixed" (Nd + 2 Nm - 1); set by --overhead


def fastest_little(N, mems, capped=True):
    """Fastest T at each S with the Little's-law memory check.

    For a tiling, memory M = R + X / (T - 2 Ld) with X = in_flight_work, so
    M <= S gives T >= 2 Ld + X / (S - R). The work cap adds T >= 2 Ld + W / cap.
    Units are then the fewest that reach that T.
    """
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    rows = []
    for S in mems:
        best = None
        for t in tilings(N, N, N):
            if t.resident >= S:
                continue
            win = max(1.0, in_flight_work(t, c) / (S - t.resident))
            if capped:
                win = max(win, W / t.cap)
            nm = math.ceil(W / win - 1e-9)
            nd = math.ceil((t.loads + c.SAVE) * LD / win - 1e-9)
            T = 2 * LD + max(math.ceil(W / nm) * LM, (t.loads + c.SAVE) * LD / nd)
            key = (round(T, 6), nd + nm)
            if best is None or key < best[0]:
                best = (key, t, nd, nm, T)
        _, t, nd, nm, T = best
        O = little_memory(t, c, T - 2 * LD) - t.resident
        rows.append({"S": S, "T": round(T), "Nd": nd, "Nm": nm, "O": round(O, 1), "O_pct": round(100 * O / S, 1),
                     "tiling": t.name, "open_outputs": t.open, "cap": t.cap, "DMA": t.loads + c.SAVE})
    return rows


def cheapest_little(N, mems, target):
    """Fewest units with T <= target; a tiling fits when its Little's-law memory M <= S."""
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    budget = target - 2 * LD
    nm = math.ceil(W / budget / LM)
    ts = [t for t in tilings(N, N, N) if t.cap >= nm]
    rows = []
    for S in mems:
        best = None
        for t in ts:
            nd = math.ceil((t.loads + c.SAVE) * LD / budget)
            M = little_memory(t, c, budget)
            if M <= S and (best is None or (nd, M) < (best[0], best[2])):
                best = (nd, t, M)
        if best is None:
            rows.append({"S": S, "feasible": False})
            continue
        nd, t, M = best
        O = M - t.resident
        T = 2 * LD + max(math.ceil(W / nm) * LM, (t.loads + c.SAVE) * LD / nd)
        rows.append({"S": S, "feasible": True, "Nd": nd, "Nm": nm, "units": nd + nm, "O": round(O, 1),
                     "O_pct": round(100 * O / S, 1), "T": round(T), "tiling": t.name, "need": round(M, 1),
                     "resident": t.resident, "cap": t.cap, "DMA": t.loads + c.SAVE})
    return rows


def fastest(N, mems, capped=True):
    if OVERHEAD == "little":
        return fastest_little(N, mems, capped)
    return fastest_fixed(N, mems, capped)


def cheapest(N, mems, target):
    if OVERHEAD == "little":
        return cheapest_little(N, mems, target)
    return cheapest_fixed(N, mems, target)


def fastest_fixed(N, mems, capped=True):
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    ts = tilings(N, N, N)
    rows = []
    for S in mems:
        best = None
        for t in ts:
            omax = S - t.need
            top = omax // 2 if not capped else min(t.cap, omax // 2)
            if top < 1:
                continue
            nm = np.arange(1, top + 1)
            nd = omax - 2 * nm + 1
            ok = nd >= 1
            nm, nd = nm[ok], nd[ok]
            if not len(nm):
                continue
            eff = np.minimum(nm, t.cap) if capped else nm
            T = 2 * LD + np.maximum(np.ceil(W / eff) * LM, (t.loads + c.SAVE) * LD / nd)
            i = int(np.argmin(T))
            Ti = float(T[i])
            nd_min = math.ceil((t.loads + c.SAVE) * LD / (Ti - 2 * LD) - 1e-9)  # fewest DMA units for that time
            nm_min = min(int(nm[i]), math.ceil(W / (Ti - 2 * LD) - 1e-9))
            key = (round(Ti, 6), nd_min + nm_min)
            if best is None or key < best[0]:
                best = (key, t, nd_min, nm_min)
        (T, _), t, nd, nm = best
        O = nd + 2 * nm - 1
        rows.append({"S": S, "T": round(T), "Nd": nd, "Nm": nm, "O": O, "O_pct": round(100 * O / S, 1),
                     "tiling": t.name, "open_outputs": t.open, "cap": t.cap, "DMA": t.loads + c.SAVE})
    return rows


def cheapest_fixed(N, mems, target):
    c = op_counts(N, N, N)
    W = c.COMPUTE + c.ACC
    budget = target - 2 * LD
    nm = math.ceil(W / budget / LM)
    ts = [t for t in tilings(N, N, N) if t.cap >= nm]
    rows = []
    for S in mems:
        best = None
        for t in ts:
            nd = math.ceil((t.loads + c.SAVE) * LD / budget)
            if t.need + nd + 2 * nm - 1 <= S and (best is None or (nd, t.need) < (best[0], best[1].need)):
                best = (nd, t)
        if best is None:
            rows.append({"S": S, "feasible": False})
            continue
        nd, t = best
        O = nd + 2 * nm - 1
        T = 2 * LD + max(math.ceil(W / nm) * LM, (t.loads + c.SAVE) * LD / nd)
        rows.append({"S": S, "feasible": True, "Nd": nd, "Nm": nm, "units": nd + nm, "O": O,
                     "O_pct": round(100 * O / S, 1), "T": round(T), "tiling": t.name, "need": t.need,
                     "cap": t.cap, "DMA": t.loads + c.SAVE})
    return rows


def min_memory(N, target, smax):
    """Smallest S at which the target can be met at all (binary search; feasibility grows with S)."""
    lo, hi = 3, smax
    if not cheapest(N, [hi], target)[0]["feasible"]:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if cheapest(N, [mid], target)[0]["feasible"]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def style(ax, logy=True):
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}K" if v >= 10000 else f"{v:,.0f}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=10)


def k_label(v):
    return f"{v // 1024}K" if v >= 1024 else str(v)


def plot_fastest(N, cap, free, out):
    xs = [r["S"] for r in cap]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 9), dpi=150, sharex=True, facecolor=SURFACE)
    style(a1)
    style(a2, logy=False)
    a1.plot(xs, [r["T"] for r in free], color=GREYS[0], lw=1.8, ls="--", marker="o", ms=4,
            label="no work cap (experiment 04)")
    a1.plot(xs, [r["T"] for r in cap], color=BLUE, lw=2.6, marker="o", ms=5, label="with independent-work cap")
    for r in cap:
        a1.annotate(f"{r['T']:,}", (r["S"], r["T"]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
    a1.set_ylabel("fastest time T (ticks)", color=INK2, fontsize=11)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK)
    a1.set_title(f"{N}³ at its fastest: about 70-80% of memory holds tiles in flight",
                 loc="left", color=INK, fontsize=12, pad=26)
    a1.text(0, 1.03, "Matrix units capped at 2 × open output tiles; units free; in-flight tiles by Little's law (fixed Nd + 2Nm − 1 with --overhead fixed)",
            transform=a1.transAxes, color=INK2, fontsize=9.5)
    a2.plot(xs, [r["O_pct"] for r in free], color=GREYS[0], lw=1.8, ls="--", marker="o", ms=4, label="no work cap")
    a2.plot(xs, [r["O_pct"] for r in cap], color=BLUE, lw=2.6, marker="o", ms=5, label="with work cap")
    for r in cap:
        a2.annotate(f"Nm={r['Nm']:,}\nNd={r['Nd']:,}", (r["S"], r["O_pct"]), xytext=(0, -26),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    a2.set_ylim(0, 105)
    a2.set_ylabel("overhead, % of memory", color=INK2, fontsize=11)
    a2.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    a2.set_xticks(xs, [k_label(x) for x in xs])
    a2.minorticks_off()
    a2.set_xlabel("Local memory S (tiles, log scale)", color=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_target(N, rows, target, out):
    ok = [r for r in rows if r["feasible"]]
    xs = [r["S"] for r in ok]
    allx = [r["S"] for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 9), dpi=150, sharex=True, facecolor=SURFACE)
    style(a1, logy=False)
    style(a2, logy=False)
    a1.plot(xs, [r["units"] for r in ok], color=INK, lw=2.6, marker="o", ms=5, label="total units Nd + Nm")
    a1.plot(xs, [r["Nd"] for r in ok], color=BLUE, lw=2, marker="^", ms=5, label="DMA units Nd")
    a1.plot(xs, [r["Nm"] for r in ok], color=AQUA, lw=2, marker="s", ms=4, label="matrix units Nm (set by the target)")
    for r in ok:
        a1.annotate(f"{r['units']}", (r["S"], r["units"]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    bad = [r["S"] for r in rows if not r["feasible"]]
    if bad:
        a1.axvspan(min(allx) / 1.4, max(bad) * 1.4, color=ORANGE, alpha=0.08, lw=0)
        a1.text((min(allx) * max(bad)) ** 0.5, 0.5, "cannot meet\nthe target", transform=a1.get_xaxis_transform(),
                ha="center", va="center", fontsize=10, color=ORANGE)
    a1.set_ylim(0, max(r["units"] for r in ok) * 1.18)
    a1.set_ylabel("units needed", color=INK2, fontsize=11)
    a1.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    first = ok[0]
    a1.set_title(f"{N}³ under {target:,} ticks: needs {first['S']}+ tiles, and more memory buys fewer DMA units",
                 loc="left", color=INK, fontsize=12, pad=26)
    a1.text(0, 1.03, "Cheapest setup (fewest Nd + Nm) at each memory size, with overlap overhead and the work cap",
            transform=a1.transAxes, color=INK2, fontsize=9.5)
    a2.bar(xs, [r["O_pct"] for r in ok], width=[x * 0.16 for x in xs], color=BLUE, alpha=0.85)
    for r in ok:
        a2.annotate(f"{r['O_pct']:.0f}%\n({r['O']})", (r["S"], r["O_pct"]), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    a2.set_ylim(0, 115)
    a2.set_ylabel("overhead, % of memory (slots)", color=INK2, fontsize=11)
    ticks = [x for x in allx if x == first["S"] or abs(x / first["S"] - 1) > 0.3]
    a2.set_xticks(ticks, [k_label(x) for x in ticks], fontsize=9)
    a2.minorticks_off()
    a2.set_xlabel("Local memory S (tiles, log scale)", color=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def write(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--target", type=int, default=15000)
    ap.add_argument("--mem", default=";".join(map(str, MEMS)))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--overhead", choices=["little", "fixed"], default="little",
                    help="memory check: Little's law (default) or the fixed Nd + 2 Nm - 1")
    a = ap.parse_args()
    global OVERHEAD
    OVERHEAD = a.overhead
    sfx = "" if a.overhead == "little" else "_fixed"
    mems = [int(s) for s in a.mem.split(";")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, free = fastest(a.size, mems, True), fastest(a.size, mems, False)
    plot_fastest(a.size, cap, free, out / f"fastest_capped_{a.size}{sfx}.png")
    write([{"capped": True, **r} for r in cap] + [{"capped": False, **r} for r in free], out / f"fastest_capped_{a.size}{sfx}.csv")
    smin = min_memory(a.size, a.target, max(mems))
    tmems = sorted(set(mems) | ({smin} if smin else set()))
    tgt = cheapest(a.size, tmems, a.target)
    print("smallest memory that meets the target:", smin)
    plot_target(a.size, tgt, a.target, out / f"target_{a.target}_{a.size}{sfx}.png")
    write(tgt, out / f"target_{a.target}_{a.size}{sfx}.csv")
    print("fastest, capped vs free")
    for r1, r2 in zip(cap, free):
        print(r1, "| free:", r2["T"], r2["Nd"], r2["Nm"], r2["O_pct"])
    print("target", a.target)
    for r in tgt:
        print(r)


if __name__ == "__main__":
    main()
