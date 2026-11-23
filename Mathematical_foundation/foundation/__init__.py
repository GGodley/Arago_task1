"""Closed-form building blocks for the blocked-matmul performance model.

Everything here is derived from first principles (see ../docs/equations.md);
the event-driven simulator in matmul_sim/ is used only to validate it.
"""

from .counts import TILE, OpCounts, cdiv, loads_with_memory, op_counts, overlap_overhead, tiles
from .parallelism import Tiling, tilings
from .timing import LD, LM, Timing, predict_time

__all__ = ["TILE", "OpCounts", "cdiv", "loads_with_memory", "op_counts", "overlap_overhead", "tiles", "LD", "LM", "Timing", "predict_time", "Tiling", "tilings"]
