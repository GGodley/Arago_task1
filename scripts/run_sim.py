#!/usr/bin/env python3
"""Run one simulation and print the results.

Examples:
    python scripts/run_sim.py --mkn 512 512 512 --nd 4 --nm 4 --mem 64
    python scripts/run_sim.py --trace results/traces/medium_512.json --hw examples/hw_default.json
    python scripts/run_sim.py --preset small_256 --nd 2 --nm 2 --mem 8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matmul_sim import HWConfig, load_trace  # noqa: E402
from matmul_sim.schedulers import SCHEDULERS, make_scheduler  # noqa: E402
from matmul_sim.simulator import simulate  # noqa: E402
from matmul_sim.translator import MatmulSpec, translate  # noqa: E402
from matmul_sim.workloads import all_presets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", type=Path, help="trace file (.json or .csv)")
    src.add_argument("--mkn", nargs=3, type=int, metavar=("M", "K", "N"))
    src.add_argument("--preset", help="a single preset name, e.g. medium_512")
    p.add_argument("--hw", type=Path, help="HW config JSON (overrides --nd/--nm/--mem)")
    p.add_argument("--nd", type=int, default=2, help="DMA units")
    p.add_argument("--nm", type=int, default=2, help="matrix units")
    p.add_argument("--mem", type=int, default=16, help="local memory in tiles")
    p.add_argument("--scheduler", choices=sorted(SCHEDULERS), default="fifo")
    p.add_argument("--json", type=Path, help="write the summary and memory series to this file")
    args = p.parse_args(argv)

    if args.trace:
        trace = load_trace(args.trace)
    elif args.mkn:
        trace = translate(MatmulSpec(*args.mkn))
    else:
        presets = all_presets()
        if args.preset not in presets:
            p.error(f"unknown preset {args.preset!r}; choose from {sorted(presets)}")
        trace = translate(presets[args.preset])

    hw = HWConfig.from_json(args.hw) if args.hw else HWConfig(args.nd, args.nm, args.mem)
    t0 = time.perf_counter()
    r = simulate(trace, hw, make_scheduler(args.scheduler))
    elapsed = time.perf_counter() - t0

    status = "completed" if r.completed else ("DEADLOCK" if r.deadlock else "STALLED")
    print(f"{r.workload} on {hw.label} with {r.scheduler}: {status} in {elapsed:.2f}s")
    print(f"  total          {r.total_ticks} ticks = {r.total_cycles} cycles")
    print(f"  lower bound    {r.bound.ticks} ticks ({r.bound.limiter}-bound), ratio {r.bound_ratio:.3f}")
    print(f"  matrix util    {r.matrix_util:.1%} (effective {r.effective_matrix_util:.1%})")
    print(f"  dma util       {r.dma_util:.1%}")
    print(f"  memory         peak {r.mem_peak}/{hw.local_mem_tiles} tiles, mean {r.mem_mean_pct:.1f}%")
    print(f"  commands       {r.commands}, reloads {r.reloads}, evictions {r.evictions}")
    if not r.completed:
        print(f"  ops done       {r.ops_done}/{r.ops_total}")
        for line in r.stuck:
            print(f"  stuck: {line}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({**r.summary(), "mem_series": r.mem_series}, indent=1) + "\n")
    return 0 if r.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
