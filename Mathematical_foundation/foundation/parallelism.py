"""Step 2c: the independent-work cap on matrix units.

A matrix unit can only run a product whose A and B tiles are both in memory,
and every product of output tile (i, j) is added into that tile's single
running sum. A running sum is a serial chain: it takes one ACC per tick (Lm),
and each ACC consumes one product. So one open output tile keeps at most

    1 + acc  matrix units busy      (one COMPUTE feeding one ACC per tick)

and a tiling with `open` output tiles in memory at once caps useful matrix
units at

    Nm_eff <= (1 + acc) * open

    block bm x bn:  open = bm bn      (all its running sums are resident)
    A or B panel p: open = p          (one column / row of p sums at a time)

Units above the cap sit idle, but still hold their slots in the overlap
overhead, so they are pure cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .counts import cdiv, tiles


@dataclass(frozen=True)
class Tiling:
    name: str
    need: int   # memory slots the tiling needs (resident tiles + one streaming step)
    loads: int  # LOAD instructions
    open: int   # output tiles open (running sums) at once
    cap: int    # max useful matrix units, (1 + acc) * open


def tilings(M: int, K: int, N: int) -> list[Tiling]:
    """Every block and panel tiling with the memory it needs, its loads and its cap.

    Matches loads_with_memory: block needs (bm + 1)(bn + 1) slots, a panel of p
    needs p (k + acc) + 2.
    """
    m, k, n = tiles(M, K, N)
    acc = 1 if k > 1 else 0
    floor = k * (m + n)
    out = []
    for bm in range(1, m + 1):
        for bn in range(1, n + 1):
            L = max(floor, k * (m * cdiv(n, bn) + n * cdiv(m, bm)))
            out.append(Tiling(f"block{bm}x{bn}", (bm + 1) * (bn + 1), L, bm * bn, (1 + acc) * bm * bn))
    for p in range(1, m + 1):
        L = max(floor, m * k + n * k * cdiv(m, p))
        out.append(Tiling(f"Apanel{p}", p * (k + acc) + 2, L, p, (1 + acc) * p))
    for p in range(1, n + 1):
        L = max(floor, n * k + m * k * cdiv(n, p))
        out.append(Tiling(f"Bpanel{p}", p * (k + acc) + 2, L, p, (1 + acc) * p))
    return out
