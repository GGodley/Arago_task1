"""Performance model for blocked matrix multiplication on a DMA + matrix-unit accelerator."""

from .hw_config import HWConfig
from .hw_model import HWError, HWModel
from .trace import Op, OpType, Trace, TraceError, load_trace

__all__ = ["HWConfig", "HWError", "HWModel", "Op", "OpType", "Trace", "TraceError", "load_trace"]
