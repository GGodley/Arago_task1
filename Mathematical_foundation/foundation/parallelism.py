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

Memory the tiling uses (Little's law, quasi-optimal scheduling): slots in use =
resident tiles + arrival rate x time each in-flight tile holds its slot, over a
compute window of T - 2 Ld ticks:

    M = R + (L tau + COMPUTE 2 Lm + SAVE Ld) / (T - 2 Ld),   tau = Ld + Lm (1 + acc)

    R, block bm x bn:  acc bm bn      (the running sums; streamed A and B tiles are
                                       counted in the load term)
    R, panel p:        p k + acc p    (the whole panel stays, plus its sums)

This replaces the fixed overhead Nd + 2 Nm - 1, which assumed every unit busy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .counts import OpCounts, cdiv, tiles


@dataclass(frozen=True)
class Tiling:
    name: str
    need: int   # memory slots the tiling needs (resident tiles + one streaming step)
    loads: int  # LOAD instructions
    open: int   # output tiles open (running sums) at once
    cap: int    # max useful matrix units, (1 + acc) * open
    resident: int = 0  # tiles held for the whole tiling (Little's-law R)


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
            out.append(Tiling(f"block{bm}x{bn}", (bm + 1) * (bn + 1), L, bm * bn, (1 + acc) * bm * bn,
                              acc * bm * bn))
    for p in range(1, m + 1):
        L = max(floor, m * k + n * k * cdiv(m, p))
        out.append(Tiling(f"Apanel{p}", p * (k + acc) + 2, L, p, (1 + acc) * p, p * (k + acc)))
    for p in range(1, n + 1):
        L = max(floor, n * k + m * k * cdiv(n, p))
        out.append(Tiling(f"Bpanel{p}", p * (k + acc) + 2, L, p, (1 + acc) * p, p * (k + acc)))
    return out


def in_flight_work(t: Tiling, c: OpCounts) -> float:
    """Slot-ticks of in-flight tiles over the whole job; divide by the window for slots."""
    from .timing import LD, LM
    acc = 1 if c.ACC else 0
    return t.loads * (LD + LM * (1 + acc)) + c.COMPUTE * 2 * LM + c.SAVE * LD


def little_memory(t: Tiling, c: OpCounts, window: float) -> float:
    """Memory the tiling uses at steady state over a compute window of `window` ticks."""
    return t.resident + in_flight_work(t, c) / window
