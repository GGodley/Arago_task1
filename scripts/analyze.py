#!/usr/bin/env python3
"""Make the report figures and the recommendations table from sweep CSVs.

Examples:
    python scripts/analyze.py results/sweep.csv
    python scripts/analyze.py results/sweep.csv --workloads small_256 medium_512 --out results/figures
    python scripts/analyze.py results/sweep.csv --cost 1 0.25 0.01 --tolerance 0.1

Per workload: cycles-vs-memory, Nm x Nd heatmaps (memory-aware at two memory
sizes), Pareto plot, matrix utilization vs Nm. Plus a memory timeline and a
Gantt chart from fresh single runs, and recommendations.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matmul_sim import HWConfig  # noqa: E402
from matmul_sim import analyzer as az  # noqa: E402
from matmul_sim.optimizer import CostModel  # noqa: E402
from matmul_sim.schedulers import make_scheduler  # noqa: E402
from matmul_sim.simulator import simulate  # noqa: E402
from matmul_sim.sweep import load_rows  # noqa: E402
from matmul_sim.translator import MatmulSpec, translate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--workloads", nargs="+", help="limit to these workloads (default: all in the CSVs)")
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--cost", nargs=3, type=float, default=(1.0, 0.25, 0.01), metavar=("A_NM", "B_ND", "C_MEM"))
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--nd", type=int, default=4, help="Nd for cycles-vs-memory lines")
    p.add_argument("--nm", type=int, default=4, help="Nm for cycles-vs-memory lines")
    p.add_argument("--no-single-runs", action="store_true", help="skip the timeline and Gantt figures")
    args = p.parse_args(argv)

    rows = [r for path in args.csv for r in load_rows(path)]
    model = CostModel(*args.cost)
    workloads = args.workloads or sorted({r["workload"] for r in rows},
                                         key=lambda w: next(r["M"] * r["K"] * r["N"] for r in rows if r["workload"] == w))
    out = args.out
    made = []
    for wl in workloads:
        mems = sorted({r["local_mem_tiles"] for r in rows if r["workload"] == wl})
        if not mems:
            continue
        made.append(az.cycles_vs_memory(rows, wl, args.nd, args.nm, out / f"{wl}_cycles_vs_memory.png"))
        for mem in {mems[len(mems) // 3], mems[-1]}:
            made.append(az.heatmap_units(rows, wl, "memaware", mem, out / f"{wl}_heatmap_memaware_mem{mem}.png"))
        made.append(az.pareto_plot(rows, wl, out / f"{wl}_pareto.png", model))
        made.append(az.utilization_vs_units(rows, wl, "memaware", mems[-1], out / f"{wl}_matrix_util.png"))

    if not args.no_single_runs:
        tr = translate(MatmulSpec(512, 512, 512))
        hw = HWConfig(4, 4, 64)
        runs = [simulate(tr, hw, make_scheduler(n, o)) for n, o in (("memaware", "row"), ("memaware", "blocked"))]
        made.append(az.memory_timeline(runs, out / "memory_timeline_512_mem64.png", ["memaware", "memaware-blocked"]))
        small = simulate(translate(MatmulSpec(128, 128, 128)), HWConfig(2, 2, 6), make_scheduler("memaware"))
        made.append(az.gantt(small, out / "gantt_128_mem6.png"))

    table = az.recommendations_table(rows, model, args.tolerance)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recommendations.md").write_text(
        f"Cost = {model.matrix}·Nm + {model.dma}·Nd + {model.mem_tile}·mem; "
        f"recommended = cheapest within {args.tolerance:.0%} of the fastest.\n\n{table}\n")
    made = [m for m in made if m]
    print(f"wrote {len(made)} figures and recommendations.md to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
