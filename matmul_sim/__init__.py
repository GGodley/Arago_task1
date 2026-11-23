"""Performance model for blocked matrix multiplication on a DMA + matrix-unit accelerator."""

from .hw_config import HWConfig
from .trace import Op, OpType, Trace, TraceError, load_trace

__all__ = ["HWConfig", "Op", "OpType", "Trace", "TraceError", "load_trace"]
