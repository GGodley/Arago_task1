"""Pick HW configs from sweep results.

Cost model (relative area, weights are parameters):

    cost = a * Nm + b * Nd + c * memory_tiles

For one workload and scheduler, over the runs that completed:

    pareto_front(rows)         configs where no other config is both cheaper and faster
    recommend(rows, tol)       cheapest config within tol (default 10%) of the fastest
    fastest_within(rows, B)    fastest config costing at most B
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    matrix: float = 1.0     # a: per matrix unit
    dma: float = 0.25       # b: per DMA unit
    mem_tile: float = 0.01  # c: per tile of local memory

    def cost(self, row: dict) -> float:
        return self.matrix * row["n_matrix"] + self.dma * row["n_dma"] + self.mem_tile * row["local_mem_tiles"]


def completed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("completed")]


def pareto_front(rows: list[dict], model: CostModel = CostModel()) -> list[dict]:
    """Non-dominated (cost, cycles) points, sorted by cost. Ties keep the cheaper/faster one."""
    pts = sorted(completed(rows), key=lambda r: (model.cost(r), r["total_cycles"]))
    front, best = [], float("inf")
    for r in pts:
        if r["total_cycles"] < best:
            front.append(r)
            best = r["total_cycles"]
    return front


def recommend(rows: list[dict], model: CostModel = CostModel(), tolerance: float = 0.10) -> dict | None:
    """Cheapest completed config whose cycles are within `tolerance` of the fastest."""
    ok = completed(rows)
    if not ok:
        return None
    fastest = min(r["total_cycles"] for r in ok)
    good = [r for r in ok if r["total_cycles"] <= fastest * (1 + tolerance)]
    return min(good, key=lambda r: (model.cost(r), r["total_cycles"]))


def fastest_within(rows: list[dict], budget: float, model: CostModel = CostModel()) -> dict | None:
    ok = [r for r in completed(rows) if model.cost(r) <= budget]
    return min(ok, key=lambda r: (r["total_cycles"], model.cost(r))) if ok else None


def group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return dict(out)


def scheduler_label(row: dict) -> str:
    return row["scheduler"] if row.get("order", "row") == "row" else f"{row['scheduler']}-{row['order']}"
