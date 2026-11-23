"""Hardware configuration: the parameters we sweep plus the fixed values from the brief.

Time model: 1 tick = 128 cycles (one matrix-unit operation). All latencies are
whole numbers of ticks, so an event-driven simulator that jumps between tick
boundaries is exact. Results are reported in cycles = ticks * tick_cycles.

Memory model: local memory is counted in tiles. One tile = one 64x64 block,
the base unit of the whole system.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Fixed by the assignment brief.
TILE_SIZE = 64          # matrix units and DMA engines handle 64x64 blocks only
LM_CYCLES = 128         # any matrix-unit operation
LD_CYCLES = 512         # one DMA transfer, global <-> local

# Smallest local memory that can finish any matmul (see README, "Minimum memory").
MIN_MEM_TILES_K1 = 3    # Kt == 1: A, B, P
MIN_MEM_TILES = 4       # Kt > 1: accumulator, A, B, new partial


@dataclass(frozen=True)
class HWConfig:
    n_dma: int                    # Nd: DMA units = global memory R/W ports
    n_matrix: int                 # Nm: matrix units
    local_mem_tiles: int          # local memory size, in 64x64 tiles
    tick_cycles: int = LM_CYCLES  # cycles per tick
    matrix_ticks: int = 1         # COMPUTE / ACC latency
    dma_ticks: int = LD_CYCLES // LM_CYCLES  # LOAD / SAVE latency (4)

    def __post_init__(self) -> None:
        for name in ("n_dma", "n_matrix", "local_mem_tiles", "tick_cycles", "matrix_ticks", "dma_ticks"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"HWConfig.{name} must be a positive integer, got {value!r}")

    @property
    def label(self) -> str:
        return f"Nd={self.n_dma} Nm={self.n_matrix} mem={self.local_mem_tiles}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "HWConfig":
        known = set(HWConfig.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown HWConfig keys: {sorted(unknown)}")
        return HWConfig(**d)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def from_json(path: str | Path) -> "HWConfig":
        return HWConfig.from_dict(json.loads(Path(path).read_text()))
