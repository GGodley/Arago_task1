"""Translator: slice C = A x B into 64x64 tile ops and assign dependencies.

The translator only slices and expands. It knows nothing about the hardware
(unit counts, memory size), so it makes no ordering or pairing decisions:

    for each output tile C[i,j]
        for k in 0..Kt-1
            LOAD A_i_k, LOAD B_k_j          (only on first use: one LOAD per tile)
            COMPUTE P_i_j_k = A_i_k x B_k_j
        ACC C_i_j = sum of P_i_j_0 .. P_i_j_{Kt-1}   (one reduction group; only if Kt > 1)
        SAVE C_i_j

The ops are written in this fixed nested-loop order (i, then j, then k). The
order carries no meaning: execution is constrained only by deps, and the
scheduler decides what runs when. How the partials of an ACC group are paired
is decided at run time too (see trace.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .hw_config import TILE_SIZE
from .trace import Op, OpType, Trace


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
    """Hardware command counts for a matmul of Mt x Kt x Nt tiles (compare with Trace.hw_op_counts)."""
    return {
        "LOAD": Mt * Kt + Kt * Nt,
        "COMPUTE": Mt * Nt * Kt,
        "ACC": Mt * Nt * (Kt - 1),
        "SAVE": Mt * Nt,
    }


def translate(spec: MatmulSpec) -> Trace:
    """Build the trace for one matmul."""
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

    for i in range(Mt):
        for j in range(Nt):
            partials = []
            for k in range(Kt):
                a = load(f"A_{i}_{k}")
                b = load(f"B_{k}_{j}")
                partials.append(emit(OpType.COMPUTE, (a, b), f"P_{i}_{j}_{k}"))
            result = emit(OpType.ACC, tuple(partials), f"C_{i}_{j}") if Kt > 1 else partials[0]
            emit(OpType.SAVE, (result,), None)

    workload = {
        "name": spec.label,
        "M": spec.M, "K": spec.K, "N": spec.N,
        "tile": TILE_SIZE,
        "Mt": Mt, "Kt": Kt, "Nt": Nt,
        "useful_fraction": round(spec.useful_fraction, 6),
        "footprint_tiles": spec.footprint_tiles,
    }
    return Trace(ops=ops, workload=workload)
