"""Parameter sweep: run the simulator over a grid of HW configs.

    rows = run_sweep(workloads, schedulers, nd=[...], nm=[...], mem=[...], out="results/sweep.csv")

Every (workload, scheduler, Nd, Nm, memory) point is one simulation. Runs go
to a process pool and each finished row is appended to the CSV straight away,
so a long sweep can be interrupted and resumed: points already in the file
are skipped.

Only summary metrics are kept (one CSV row per run, no command log). Memory
sizes below the workload's minimum are recorded as infeasible without
simulating, and a run that passes MAX_TICKS_FACTOR x its lower bound is stopped
and recorded as too_slow.

status column: completed | deadlock | infeasible | too_slow
"""

from __future__ import annotations

import csv
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .bounds import min_memory_tiles
from .hw_config import HWConfig
from .schedulers import make_scheduler
from .simulator import simulate
from .translator import MatmulSpec, translate
from .workloads import PRESETS

MAX_TICKS_FACTOR = 20

FIELDS = [
    "workload", "workload_class", "M", "K", "N", "scheduler", "order",
    "n_dma", "n_matrix", "local_mem_tiles", "status",
    "completed", "deadlock", "stalled", "infeasible_memory",
    "total_ticks", "total_cycles", "lower_bound_ticks", "bound_ratio", "bound_limiter",
    "matrix_util", "effective_matrix_util", "dma_util", "mem_peak", "mem_mean_pct",
    "reloads", "evictions", "elapsed_s",
]
KEY = ("workload", "scheduler", "order", "n_dma", "n_matrix", "local_mem_tiles")


@dataclass(frozen=True)
class Point:
    spec: MatmulSpec
    workload_class: str
    scheduler: str
    order: str
    n_dma: int
    n_matrix: int
    mem: int

    def key(self) -> tuple:
        return (self.spec.label, self.scheduler, self.order, str(self.n_dma), str(self.n_matrix), str(self.mem))


def class_of(spec: MatmulSpec) -> str:
    for cls, specs in PRESETS.items():
        if spec in specs:
            return cls
    return "custom"


def grid(workloads: Iterable[MatmulSpec], schedulers: Iterable[tuple[str, str]],
         nd: Iterable[int], nm: Iterable[int], mem: Iterable[int]) -> list[Point]:
    return [Point(w, class_of(w), s, o, d, m, t)
            for w, (s, o), d, m, t in itertools.product(list(workloads), list(schedulers), list(nd), list(nm), list(mem))]


# Each worker process keeps the traces it has built.
_TRACES: dict[MatmulSpec, object] = {}


def run_point(p: Point) -> dict:
    trace = _TRACES.get(p.spec)
    if trace is None:
        trace = _TRACES[p.spec] = translate(p.spec)
    base = {
        "workload": p.spec.label, "workload_class": p.workload_class,
        "M": p.spec.M, "K": p.spec.K, "N": p.spec.N,
        "scheduler": p.scheduler, "order": p.order,
        "n_dma": p.n_dma, "n_matrix": p.n_matrix, "local_mem_tiles": p.mem,
    }
    if p.mem < min_memory_tiles(trace):
        return {**base, "status": "infeasible", "completed": False, "deadlock": True, "stalled": False,
                "infeasible_memory": True, "elapsed_s": 0.0}
    t0 = time.perf_counter()
    r = simulate(trace, HWConfig(p.n_dma, p.n_matrix, p.mem), make_scheduler(p.scheduler, p.order),
                 record=False, max_ticks_factor=MAX_TICKS_FACTOR)
    s = r.summary()
    row = {**base, **{k: s[k] for k in FIELDS if k in s and k not in base}}
    row["status"] = "completed" if r.completed else ("deadlock" if r.deadlock else "too_slow")
    row["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return row


def _done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {tuple(row[k] for k in KEY) for row in csv.DictReader(f)}


def run_sweep(points: list[Point], out: str | Path, processes: int | None = None,
              progress: bool = True) -> int:
    """Run every point not already in `out`. Returns the number of new rows written."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    todo = [p for p in points if p.key() not in done]
    # Biggest workloads first, so the long runs overlap with everything else.
    todo.sort(key=lambda p: -(p.spec.M * p.spec.K * p.spec.N))
    new_file = not out.exists() or out.stat().st_size == 0
    processes = processes or os.cpu_count() or 1
    written = 0
    t0 = time.perf_counter()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        if processes == 1:
            results = (run_point(p) for p in todo)
            for row in results:
                w.writerow(row); f.flush(); written += 1
                if progress:
                    _report(written, len(todo), t0, row)
        else:
            with ProcessPoolExecutor(max_workers=processes) as pool:
                futures = [pool.submit(run_point, p) for p in todo]
                for fut in as_completed(futures):
                    row = fut.result()
                    w.writerow(row); f.flush(); written += 1
                    if progress:
                        _report(written, len(todo), t0, row)
    return written


def _report(n: int, total: int, t0: float, row: dict) -> None:
    if n == total or n % max(1, total // 20) == 0:
        print(f"  {n}/{total} done, {time.perf_counter() - t0:.0f}s elapsed "
              f"(last: {row['workload']} {row['scheduler']} Nd={row['n_dma']} Nm={row['n_matrix']} "
              f"mem={row['local_mem_tiles']})", flush=True)


def load_rows(path: str | Path) -> list[dict]:
    """Read a sweep CSV back with numbers and booleans converted."""
    ints = {"M", "K", "N", "n_dma", "n_matrix", "local_mem_tiles", "total_ticks", "total_cycles",
            "lower_bound_ticks", "mem_peak", "reloads", "evictions"}
    floats = {"bound_ratio", "matrix_util", "effective_matrix_util", "dma_util", "mem_mean_pct", "elapsed_s"}
    bools = {"completed", "deadlock", "stalled", "infeasible_memory"}
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            row: dict = {}
            for k, v in raw.items():
                if k in ints:
                    row[k] = int(v) if v not in ("", None) else None
                elif k in floats:
                    row[k] = float(v) if v not in ("", None) else None
                elif k in bools:
                    row[k] = v == "True"
                else:
                    row[k] = v
            rows.append(row)
    return rows
