from .base import Command, Scheduler
from .fifo import FIFOScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    FIFOScheduler.name: FIFOScheduler,
}


def make_scheduler(name: str) -> Scheduler:
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise ValueError(f"unknown scheduler {name!r}; available: {sorted(SCHEDULERS)}") from None


__all__ = ["Command", "Scheduler", "FIFOScheduler", "SCHEDULERS", "make_scheduler"]
