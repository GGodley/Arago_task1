"""Scheduler interface.

The simulator owns the clock and the HW model. The scheduler owns *ordering*:
it holds the commands that are ready to run and, at every decision point,
offers them to the simulator in priority order. It also decides which two
tiles an ACC adds and which tile to evict when memory is full.

Protocol (all calls come from the simulator):

    reset(sim)              once, before the run
    on_ready(cmd)           cmd can be considered from now on
    candidates(sim)         generator: yield ready commands in priority order.
                            After each yield the simulator sets cmd.issued.
                            Commands not issued must stay pending. The
                            simulator closes the generator when it is done
                            (all units busy, or nothing left to try).
    pair(sim, group, avail) choose two of the ready tiles of an ACC group
    victim(sim, candidates) choose a tile to evict, or None to not evict

Inside candidates(), the scheduler should skip commands that cannot run:
those whose unit pool is full (sim.hw.pool(unit).free_count == 0) and, once
sim.mem_blocked is set, those that need a new memory slot (cmd.needs_memory).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from ..trace import OpType

if TYPE_CHECKING:  # pragma: no cover
    from ..simulator import Simulator


@dataclass(eq=False)
class Command:
    """One hardware command, as offered to the scheduler.

    kind      LOAD | COMPUTE | ACC | SAVE
    priority  default ordering key: the trace op id (for ACC, the group's op id;
              for a reload, the id of the tile's original LOAD)
    op_id     trace op this command belongs to (ACC: the group op id; None for reloads)
    srcs/dst  tiles read and written. For ACC they are filled in at issue time,
              once the pair is chosen.
    """

    kind: OpType
    priority: int
    op_id: int | None
    srcs: tuple[str, ...] = ()
    dst: str | None = None
    is_reload: bool = False
    issued: bool = field(default=False, compare=False)

    @property
    def unit(self) -> str:
        return self.kind.unit

    @property
    def needs_memory(self) -> bool:
        return self.kind is not OpType.SAVE

    def __repr__(self) -> str:  # compact, for logs and test failures
        tag = "RELOAD" if self.is_reload else self.kind.value
        return f"<{tag} op={self.op_id} {','.join(self.srcs)}->{self.dst}>"


class Scheduler:
    """Base class. Subclasses must implement on_ready and candidates."""

    name = "base"
    #: set True if victim() may return a tile; lets the simulator skip work otherwise
    evicts = False

    def reset(self, sim: "Simulator") -> None:
        pass

    def on_ready(self, cmd: Command) -> None:
        raise NotImplementedError

    def candidates(self, sim: "Simulator") -> Iterator[Command]:
        raise NotImplementedError

    def pair(self, sim: "Simulator", group: int, avail: list[str]) -> tuple[str, str]:
        """Default: the two tiles that became ready first."""
        return avail[0], avail[1]

    def victim(self, sim: "Simulator", candidates: list[str]) -> str | None:
        """Default: never evict (the run deadlocks if memory fills with inputs)."""
        return None
