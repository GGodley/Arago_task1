#!/usr/bin/env python3
"""Generate workload traces for matrix multiplications.

Examples:
    python scripts/gen_workload.py --mkn 512 512 512
    python scripts/gen_workload.py --mkn 100 200 300 --format csv --out my.csv
    python scripts/gen_workload.py --preset small --order snake
    python scripts/gen_workload.py --preset all --outdir results/traces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matmul_sim.translator import ACC_MODES, ORDERS, MatmulSpec, translate  # noqa: E402
from matmul_sim.workloads import PRESETS, get_preset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mkn", nargs=3, type=int, metavar=("M", "K", "N"), help="C(MxN) = A(MxK) x B(KxN)")
    src.add_argument("--preset", help=f"a class ({', '.join(PRESETS)}), 'all', or one preset name")
    p.add_argument("--order", choices=ORDERS, default="row", help="output tile visiting order (default: row)")
    p.add_argument("--block", nargs=2, type=int, default=(4, 4), metavar=("BI", "BJ"), help="group size for --order blocked")
    p.add_argument("--acc", choices=ACC_MODES, default="chain", help="ACC reduction shape (default: chain)")
    p.add_argument("--format", choices=("json", "csv"), default="json")
    p.add_argument("--out", type=Path, help="output file (single workload only)")
    p.add_argument("--outdir", type=Path, default=Path("results/traces"))
    args = p.parse_args(argv)

    if args.mkn:
        specs = [MatmulSpec(*args.mkn)]
    elif args.preset == "all":
        specs = [s for group in PRESETS.values() for s in group]
    else:
        specs = get_preset(args.preset)

    if args.out and len(specs) != 1:
        p.error("--out only works with a single workload; use --outdir")

    for spec in specs:
        trace = translate(spec, order=args.order, acc=args.acc, block=tuple(args.block))
        trace.validate()
        if args.out:
            path = args.out
        else:
            suffix = "" if (args.order, args.acc) == ("row", "chain") else f"_{args.order}_{args.acc}"
            path = args.outdir / f"{spec.label}{suffix}.{args.format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        getattr(trace, f"to_{args.format}")(path)
        c = trace.op_counts()
        print(f"{path}: {len(trace.ops)} ops (LOAD {c['LOAD']}, COMPUTE {c['COMPUTE']}, "
              f"ACC {c['ACC']}, SAVE {c['SAVE']}), footprint {spec.footprint_tiles} tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
