"""Translator: slice C = A x B into 64x64 tile ops with dependencies.

Given M, K, N, the translator pads each dimension up to a multiple of 64 and
emits a trace of LOAD / COMPUTE / ACC / SAVE ops (see trace.py):

    for each output tile C[i,j]            (visited in the chosen order)
        for k in 0..Kt-1
            LOAD A_i_k, LOAD B_k_j         (only on first use: one LOAD per tile)
            COMPUTE P_i_j_k = A_i_k x B_k_j
        reduce P_i_j_0 .. P_i_j_{Kt-1}     (chain or tree of ACCs)
        SAVE the final tile

The order of ops in the trace is a *suggested program order*. Schedulers may
use it as a tie-breaker. It does not constrain execution: only deps do.

The trace does not depend on the HW config. Reuse and reloads are decided at
run time by the scheduler and simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

from .hw_config import TILE_SIZE
from .trace import Op, OpType, Trace

ORDERS = ("row", "snake", "blocked")
ACC_MODES = ("chain", "tree")


@dataclass(frozen=True)
class MatmulSpec:
    """C (M x N) = A (M x K) x B (K x N)."""

    M: int
    K: int
    N: int
    name: str | None = None

    def __post_init__(self) -> None:
        for d in ("M", "K", "N"):
            v = getattr(self, d)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{d} must be a positive integer, got {v!r}")

    @property
    def tiles(self) -> tuple[int, int, int]:
        """(Mt, Kt, Nt): dimensions in 64x64 tiles, rounded up."""
        return tuple(math.ceil(d / TILE_SIZE) for d in (self.M, self.K, self.N))  # type: ignore[return-value]

    @property
    def label(self) -> str:
        return self.name or f"mm_{self.M}x{self.K}x{self.N}"

    @property
    def useful_fraction(self) -> float:
        """Fraction of tile work that is real (not padding): MKN / (Mp Kp Np)."""
        Mt, Kt, Nt = self.tiles
        return (self.M * self.K * self.N) / (Mt * Kt * Nt * TILE_SIZE**3)

    @property
    def footprint_tiles(self) -> int:
        """Distinct tiles of A, B and C: what local memory needs to hold everything at once."""
        Mt, Kt, Nt = self.tiles
        return Mt * Kt + Kt * Nt + Mt * Nt


def expected_op_counts(Mt: int, Kt: int, Nt: int) -> dict[str, int]:
    """Op counts for a matmul of Mt x Kt x Nt tiles (same for chain and tree ACC)."""
    return {
        "LOAD": Mt * Kt + Kt * Nt,
        "COMPUTE": Mt * Nt * Kt,
        "ACC": Mt * Nt * (Kt - 1),
        "SAVE": Mt * Nt,
    }


def output_order(Mt: int, Nt: int, order: str = "row", block: tuple[int, int] = (4, 4)) -> Iterator[tuple[int, int]]:
    """Order in which output tiles C[i,j] are visited.

    row      i outer, j inner.
    snake    like row, but j reverses direction on every other row, so the
             last B column of one row is reused by the first tile of the next.
    blocked  visit block[0] x block[1] groups of output tiles one group at a
             time (row order inside and across groups). A group shares
             block[0] rows of A and block[1] columns of B, which bounds the
             working set for a given local memory size.
    """
    if order == "row":
        for i in range(Mt):
            for j in range(Nt):
                yield i, j
    elif order == "snake":
        for i in range(Mt):
            cols = range(Nt) if i % 2 == 0 else range(Nt - 1, -1, -1)
            for j in cols:
                yield i, j
    elif order == "blocked":
        bi, bj = block
        if bi < 1 or bj < 1:
            raise ValueError(f"block sizes must be >= 1, got {block}")
        for i0 in range(0, Mt, bi):
            for j0 in range(0, Nt, bj):
                for i in range(i0, min(i0 + bi, Mt)):
                    for j in range(j0, min(j0 + bj, Nt)):
                        yield i, j
    else:
        raise ValueError(f"unknown order {order!r}, expected one of {ORDERS}")


def translate(
    spec: MatmulSpec,
    order: str = "row",
    acc: str = "chain",
    block: tuple[int, int] = (4, 4),
) -> Trace:
    """Build the trace for one matmul."""
    if acc not in ACC_MODES:
        raise ValueError(f"unknown acc mode {acc!r}, expected one of {ACC_MODES}")
    Mt, Kt, Nt = spec.tiles

    ops: list[Op] = []
    producer: dict[str, int] = {}  # tile name -> id of the op that writes it

    def emit(op_type: OpType, srcs: tuple[str, ...], dst: str | None) -> str | None:
        op_id = len(ops)
        deps = tuple(sorted({producer[s] for s in srcs}))
        ops.append(Op(op_id, op_type, srcs, dst, deps))
        if dst is not None:
            producer[dst] = op_id
        return dst

    def load(tile: str) -> str:
        if tile not in producer:
            emit(OpType.LOAD, (), tile)
        return tile

    for i, j in output_order(Mt, Nt, order, block):
        partials: list[str] = []
        running: str | None = None
        for k in range(Kt):
            a = load(f"A_{i}_{k}")
            b = load(f"B_{k}_{j}")
            p = emit(OpType.COMPUTE, (a, b), f"P_{i}_{j}_{k}")
            if acc == "chain":
                # S_i_j_k = S_i_j_{k-1} + P_i_j_k; the first partial seeds the sum.
                running = p if running is None else emit(OpType.ACC, (running, p), f"S_{i}_{j}_{k}")
            else:
                partials.append(p)

        if acc == "tree":
            level = 0
            while len(partials) > 1:
                nxt = []
                for n in range(0, len(partials) - 1, 2):
                    nxt.append(emit(OpType.ACC, (partials[n], partials[n + 1]), f"T_{i}_{j}_{level}_{n // 2}"))
                if len(partials) % 2:
                    nxt.append(partials[-1])  # odd one out moves up a level unchanged
                partials = nxt
                level += 1
            running = partials[0]

        emit(OpType.SAVE, (running,), None)

    workload = {
        "name": spec.label,
        "M": spec.M, "K": spec.K, "N": spec.N,
        "tile": TILE_SIZE,
        "Mt": Mt, "Kt": Kt, "Nt": Nt,
        "useful_fraction": round(spec.useful_fraction, 6),
        "footprint_tiles": spec.footprint_tiles,
        "order": order,
        "acc": acc,
    }
    if order == "blocked":
        workload["block"] = list(block)
    return Trace(ops=ops, workload=workload)


def acc_depth(Kt: int, acc: str = "chain") -> int:
    """ACC ops on the critical path of one output tile."""
    if Kt <= 1:
        return 0
    return Kt - 1 if acc == "chain" else math.ceil(math.log2(Kt))
