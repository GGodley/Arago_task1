from .base import Command, Scheduler
from .fifo import FIFOScheduler
from .reuse import ORDERS, GreedyReuseScheduler, MemoryAwareScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    FIFOScheduler.name: FIFOScheduler,
    GreedyReuseScheduler.name: GreedyReuseScheduler,
    MemoryAwareScheduler.name: MemoryAwareScheduler,
}


def make_scheduler(name: str, order: str = "row", block: int = 4) -> Scheduler:
    """Build a scheduler by name. order/block apply to the reuse schedulers."""
    try:
        cls = SCHEDULERS[name]
    except KeyError:
        raise ValueError(f"unknown scheduler {name!r}; available: {sorted(SCHEDULERS)}") from None
    if issubclass(cls, GreedyReuseScheduler):
        return cls(order=order, block=block)
    return cls()


__all__ = ["Command", "Scheduler", "FIFOScheduler", "GreedyReuseScheduler", "MemoryAwareScheduler",
           "ORDERS", "SCHEDULERS", "make_scheduler"]
