"""Instruction counts for C = A @ B, grouped by the unit that executes them.

With m, k, n = ceil(M/64), ceil(K/64), ceil(N/64) tiles:

    matrix units (MAC)                 DMA units
      COMPUTE = m*n*k                    LOAD (compulsory) = k*(m + n)
      ACC     = m*n*(k - 1)              SAVE              = m*n
                                         LOAD (no reuse)   = 2*m*n*k   (each COMPUTE loads its own A, B)

With only S tiles of local memory some input tiles must be loaded more than once.
loads_with_memory() returns the fewest LOADs any block or panel tiling needs
(derivation doc, Step 2):

    block bm x bn   k*(m*ceil(n/bn) + n*ceil(m/bm)),   needs (bm+1)(bn+1) <= S
    A panel of p    m*k + n*k*ceil(m/p),               p = (S-2) // (k + acc)
    (B panel symmetric)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

TILE = 64  # matrix units and DMA engines work on 64x64 tiles


def cdiv(a: int, b: int) -> int:
    """Ceiling division for positive integers."""
    return -(-a // b)


def tiles(M: int, K: int, N: int) -> tuple[int, int, int]:
    """Matrix dimensions in tiles, padding partial tiles up."""
    return cdiv(M, TILE), cdiv(K, TILE), cdiv(N, TILE)


@dataclass(frozen=True)
class OpCounts:
    M: int
    K: int
    N: int
    m: int
    k: int
    n: int
    COMPUTE: int
    ACC: int
    LOAD: int  # compulsory: every input tile exactly once
    SAVE: int

    @property
    def mac_total(self) -> int:
        return self.COMPUTE + self.ACC

    @property
    def dma_total(self) -> int:
        """Perfect reuse: every input tile loaded exactly once."""
        return self.LOAD + self.SAVE

    @property
    def load_no_reuse(self) -> int:
        """No reuse: every COMPUTE loads its own A and B tile, 2*m*n*k LOADs."""
        return 2 * self.m * self.n * self.k

    @property
    def dma_no_reuse(self) -> int:
        return self.load_no_reuse + self.SAVE

    def as_dict(self) -> dict:
        return {**asdict(self), "MAC_total": self.mac_total, "DMA_total": self.dma_total,
                "LOAD_no_reuse": self.load_no_reuse, "DMA_no_reuse": self.dma_no_reuse}


def op_counts(M: int, K: int, N: int) -> OpCounts:
    """Exact command counts; independent of the hardware configuration."""
    m, k, n = tiles(M, K, N)
    return OpCounts(M, K, N, m, k, n,
                    COMPUTE=m * n * k,
                    ACC=m * n * (k - 1),
                    LOAD=k * (m + n),
                    SAVE=m * n)


def overlap_overhead(nd: int, nm: int) -> int:
    """Extra slots needed so transfers and matrix ops can overlap.

    Rule 3 (a tile holds its slot from issue until its last reader or SAVE finishes) means:
      - every busy DMA unit holds one slot beyond the static working set: a LOAD's
        destination tile while it is in flight, or, for a SAVE, the finished sum it is
        writing out while the next block has already opened its own sum  ->  Nd
      - every busy matrix unit may hold a product waiting for its ACC, and an ACC
        allocates its result before freeing its two inputs  ->  2 per unit, minus
        the 1 product slot the sequential budget already includes   ->  2*Nm - 1
    """
    return nd + 2 * nm - 1


def loads_with_memory(M: int, K: int, N: int, S: int, overhead: int = 0) -> int | None:
    """Fewest LOADs (compulsory + reloads) achievable with S tiles of local memory.

    Ideal tiling only: this is what a quasi-optimal schedule needs, not what any
    particular scheduler achieves. `overhead` slots are reserved first (see
    overlap_overhead); returns None if what is left cannot hold even a 1x1 block.
    """
    S = S - overhead
    if S < 3 + (1 if cdiv(K, TILE) > 1 else 0):
        return None
    m, k, n = tiles(M, K, N)
    acc = 1 if k > 1 else 0
    best = 2 * m * n * k  # no reuse at all
    for bm in range(1, m + 1):
        for bn in range(1, n + 1):
            if (bm + 1) * (bn + 1) > S:
                break
            best = min(best, k * (m * cdiv(n, bn) + n * cdiv(m, bm)))
    p = (S - 2) // (k + acc)
    if p >= 1:
        best = min(best,
                   m * k + n * k * cdiv(m, min(m, p)),
                   n * k + m * k * cdiv(n, min(n, p)))
    return max(best, k * (m + n))
