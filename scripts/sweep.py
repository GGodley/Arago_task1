#!/usr/bin/env python3
"""Sweep the HW parameters for one or more workloads.

Examples:
    python scripts/sweep.py --preset small medium skewed                  # default grid, all schedulers
    python scripts/sweep.py --preset large_2048 --schedulers memaware greedy
    python scripts/sweep.py --mkn 1024 1024 1024 --nd 8 --nm 1 2 4 8 16 --mem 512
    python scripts/sweep.py --preset large_4096 --slices 8 8 512          # one variable at a time

Rows are appended to --out (default results/sweep.csv); points already there are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matmul_sim.sweep import grid, run_sweep  # noqa: E402
from matmul_sim.translator import MatmulSpec  # noqa: E402
from matmul_sim.workloads import get_preset  # noqa: E402

DEFAULT_ND = [1, 2, 4, 8, 16]
DEFAULT_NM = [1, 2, 4, 8, 16]
DEFAULT_MEM = [4, 8, 16, 32, 64, 128, 256, 512]
SCHEDULER_SETS = {
    "fifo": ("fifo", "row"),
    "greedy": ("greedy", "row"),
    "memaware": ("memaware", "row"),
    "memaware-snake": ("memaware", "snake"),
    "memaware-blocked": ("memaware", "blocked"),
    "greedy-blocked": ("greedy", "blocked"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--preset", nargs="+", help="preset classes or names")
    src.add_argument("--mkn", nargs=3, type=int, metavar=("M", "K", "N"))
    p.add_argument("--schedulers", nargs="+", choices=sorted(SCHEDULER_SETS),
                   default=["fifo", "greedy", "memaware", "memaware-blocked"])
    p.add_argument("--nd", nargs="+", type=int, default=DEFAULT_ND)
    p.add_argument("--nm", nargs="+", type=int, default=DEFAULT_NM)
    p.add_argument("--mem", nargs="+", type=int, default=DEFAULT_MEM)
    p.add_argument("--slices", nargs=3, type=int, metavar=("ND", "NM", "MEM"),
                   help="instead of the full grid, vary one variable at a time around this center")
    p.add_argument("--out", type=Path, default=Path("results/sweep.csv"))
    p.add_argument("--procs", type=int, default=None, help="worker processes (default: all cores)")
    args = p.parse_args(argv)

    workloads = [MatmulSpec(*args.mkn)] if args.mkn else [s for name in args.preset for s in get_preset(name)]
    scheds = [SCHEDULER_SETS[s] for s in args.schedulers]

    if args.slices:
        d0, m0, t0 = args.slices
        points = (grid(workloads, scheds, args.nd, [m0], [t0])
                  + grid(workloads, scheds, [d0], args.nm, [t0])
                  + grid(workloads, scheds, [d0], [m0], args.mem))
        points = list({pt.key(): pt for pt in points}.values())
    else:
        points = grid(workloads, scheds, args.nd, args.nm, args.mem)

    print(f"{len(points)} points: {len(workloads)} workloads x {len(scheds)} schedulers -> {args.out}")
    start = time.perf_counter()
    n = run_sweep(points, args.out, processes=args.procs)
    print(f"wrote {n} new rows in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
