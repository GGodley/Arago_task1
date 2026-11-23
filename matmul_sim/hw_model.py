"""HW model: a passive scoreboard of the hardware.

It records what is in use and what is free, and nothing else. It has no
clock and no policy: the simulator tells it what starts and finishes and
when, and the scheduler reads it to decide what to issue. Every state change
is logged with its tick, which gives the occupancy time series and the unit
timelines for the analyzer.

Two parts:

UnitPool   n identical units (the DMA pool or the matrix pool). Each slot is
           free or holds one command until a known end tick.

LocalMemory  capacity in 64x64 tiles. Each resident tile has:
    state           PENDING (slot allocated, being written) or READY
    remaining_uses  how many issued-or-future commands still need to read it;
                    the simulator sets it from the dependency graph. When it
                    reaches 0 and the tile is not a final output awaiting
                    SAVE, the simulator frees it.
    pins            how many *running* commands are reading it right now.
                    A pinned tile can't be evicted or freed.

Invariants enforced here (violations raise HWError, which means a simulator
or scheduler bug, not a modelled event):
    - allocated tiles never exceed capacity
    - a tile is allocated at most once at a time
    - only READY, unpinned tiles can be evicted or freed
    - a unit slot runs one command at a time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .hw_config import HWConfig


class HWError(RuntimeError):
    """An illegal operation on the scoreboard: a bug in the caller."""


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BusyInterval:
    """One command occupying one unit slot, for utilization and Gantt charts."""

    unit: str
    slot: int
    cmd: int
    start: int
    end: int


@dataclass
class _Slot:
    cmd: int
    start: int
    end: int


class UnitPool:
    """n identical units of one kind ('dma' or 'matrix')."""

    def __init__(self, kind: str, n: int) -> None:
        if n < 1:
            raise ValueError(f"{kind} pool needs at least one unit")
        self.kind = kind
        self.n = n
        self._slots: list[_Slot | None] = [None] * n
        self.history: list[BusyInterval] = []
        self.busy_ticks = 0

    # -- queries --
    @property
    def free_count(self) -> int:
        return sum(s is None for s in self._slots)

    @property
    def busy_count(self) -> int:
        return self.n - self.free_count

    def running(self) -> list[tuple[int, int, int]]:
        """(slot, cmd, end tick) for every busy slot."""
        return [(i, s.cmd, s.end) for i, s in enumerate(self._slots) if s is not None]

    def next_end(self) -> int | None:
        """Earliest end tick among running commands, or None if idle."""
        ends = [s.end for s in self._slots if s is not None]
        return min(ends) if ends else None

    # -- updates --
    def start(self, cmd: int, now: int, duration: int) -> int:
        """Occupy a free slot with cmd from now to now + duration. Returns the slot index."""
        if duration < 1:
            raise HWError(f"{self.kind}: duration must be >= 1, got {duration}")
        for i, s in enumerate(self._slots):
            if s is None:
                self._slots[i] = _Slot(cmd, now, now + duration)
                return i
        raise HWError(f"{self.kind}: no free unit for command {cmd}")

    def finish(self, slot: int, now: int) -> int:
        """Free a slot whose command ends at now. Returns the command id."""
        s = self._slots[slot]
        if s is None:
            raise HWError(f"{self.kind}: slot {slot} is already free")
        if s.end != now:
            raise HWError(f"{self.kind}: command {s.cmd} ends at {s.end}, not {now}")
        self._slots[slot] = None
        self.history.append(BusyInterval(self.kind, slot, s.cmd, s.start, s.end))
        self.busy_ticks += s.end - s.start
        return s.cmd


# --------------------------------------------------------------------------
# Local memory
# --------------------------------------------------------------------------

class TileState(str, Enum):
    PENDING = "PENDING"  # slot allocated, the writing command is still running
    READY = "READY"      # contents valid, can be read


@dataclass
class TileEntry:
    state: TileState
    remaining_uses: int
    pins: int = 0


@dataclass(frozen=True)
class MemEvent:
    tick: int
    kind: str      # alloc | ready | free | evict
    tile: str
    used: int      # tiles allocated after this event


class LocalMemory:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("local memory needs at least one tile")
        self.capacity = capacity
        self.tiles: dict[str, TileEntry] = {}
        self.events: list[MemEvent] = []
        self.peak = 0

    # -- queries --
    @property
    def used(self) -> int:
        return len(self.tiles)

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def can_allocate(self, n: int = 1) -> bool:
        return self.free >= n

    def is_resident(self, tile: str) -> bool:
        return tile in self.tiles

    def is_ready(self, tile: str) -> bool:
        e = self.tiles.get(tile)
        return e is not None and e.state is TileState.READY

    def entry(self, tile: str) -> TileEntry:
        try:
            return self.tiles[tile]
        except KeyError:
            raise HWError(f"tile {tile!r} is not resident") from None

    def evictable(self) -> list[str]:
        """READY tiles no running command is reading."""
        return [t for t, e in self.tiles.items() if e.state is TileState.READY and e.pins == 0]

    # -- updates --
    def allocate(self, tile: str, remaining_uses: int, now: int) -> None:
        """Reserve a slot for a tile about to be written (LOAD, COMPUTE or ACC issued)."""
        if tile in self.tiles:
            raise HWError(f"tile {tile!r} is already resident")
        if self.free < 1:
            raise HWError(f"local memory full ({self.capacity} tiles), cannot allocate {tile!r}")
        if remaining_uses < 0:
            raise HWError(f"tile {tile!r}: remaining_uses must be >= 0")
        self.tiles[tile] = TileEntry(TileState.PENDING, remaining_uses)
        self.peak = max(self.peak, self.used)
        self._log(now, "alloc", tile)

    def mark_ready(self, tile: str, now: int) -> None:
        """The command writing tile has finished."""
        e = self.entry(tile)
        if e.state is TileState.READY:
            raise HWError(f"tile {tile!r} is already READY")
        e.state = TileState.READY
        self._log(now, "ready", tile)

    def pin(self, tile: str) -> None:
        """A command that reads tile has started."""
        e = self.entry(tile)
        if e.state is not TileState.READY:
            raise HWError(f"cannot read tile {tile!r}: not READY")
        e.pins += 1

    def unpin(self, tile: str) -> int:
        """A command that read tile has finished: one fewer pin and one fewer use.

        Returns the remaining uses. The caller decides whether to free the tile
        (a final output must stay until its SAVE completes).
        """
        e = self.entry(tile)
        if e.pins < 1:
            raise HWError(f"tile {tile!r} is not pinned")
        if e.remaining_uses < 1:
            raise HWError(f"tile {tile!r} has no remaining uses to consume")
        e.pins -= 1
        e.remaining_uses -= 1
        return e.remaining_uses

    def add_uses(self, tile: str, n: int) -> None:
        """Adjust a tile's remaining uses (e.g. after a reload re-creates it)."""
        e = self.entry(tile)
        if e.remaining_uses + n < 0:
            raise HWError(f"tile {tile!r}: remaining uses would go negative")
        e.remaining_uses += n

    def free_tile(self, tile: str, now: int) -> None:
        """Release a tile that nobody needs any more."""
        self._remove(tile, now, "free")

    def evict(self, tile: str, now: int) -> int:
        """Drop a tile that is still needed. Returns its remaining uses (for the reload)."""
        e = self.entry(tile)
        uses = e.remaining_uses
        self._remove(tile, now, "evict")
        return uses

    # -- internals --
    def _remove(self, tile: str, now: int, kind: str) -> None:
        e = self.entry(tile)
        if e.state is not TileState.READY:
            raise HWError(f"cannot {kind} tile {tile!r}: still being written")
        if e.pins:
            raise HWError(f"cannot {kind} tile {tile!r}: read by {e.pins} running command(s)")
        del self.tiles[tile]
        self._log(now, kind, tile)

    def _log(self, now: int, kind: str, tile: str) -> None:
        self.events.append(MemEvent(now, kind, tile, self.used))

    def occupancy_series(self) -> list[tuple[int, int]]:
        """(tick, tiles allocated) step series: the value holds from that tick until the next entry.

        Several events on one tick collapse to the value after the last one.
        Starts at (0, 0).
        """
        series: list[tuple[int, int]] = [(0, 0)]
        for ev in self.events:
            if series[-1][0] == ev.tick:
                series[-1] = (ev.tick, ev.used)
            else:
                series.append((ev.tick, ev.used))
        # drop consecutive duplicates
        out = [series[0]]
        for point in series[1:]:
            if point[1] != out[-1][1]:
                out.append(point)
        return out


# --------------------------------------------------------------------------
# Scoreboard
# --------------------------------------------------------------------------

@dataclass
class HWModel:
    """The whole scoreboard for one HW config."""

    config: HWConfig
    dma: UnitPool = field(init=False)
    matrix: UnitPool = field(init=False)
    memory: LocalMemory = field(init=False)

    def __post_init__(self) -> None:
        self.dma = UnitPool("dma", self.config.n_dma)
        self.matrix = UnitPool("matrix", self.config.n_matrix)
        self.memory = LocalMemory(self.config.local_mem_tiles)

    def pool(self, unit: str) -> UnitPool:
        if unit == "dma":
            return self.dma
        if unit == "matrix":
            return self.matrix
        raise ValueError(f"unknown unit {unit!r}")

    def duration(self, unit: str) -> int:
        """Latency in ticks of a command on this unit."""
        return self.config.dma_ticks if unit == "dma" else self.config.matrix_ticks

    def next_event(self) -> int | None:
        """Earliest tick at which some running command finishes."""
        ends = [t for t in (self.dma.next_end(), self.matrix.next_end()) if t is not None]
        return min(ends) if ends else None

    def idle(self) -> bool:
        return self.dma.busy_count == 0 and self.matrix.busy_count == 0

    def utilization(self, total_ticks: int) -> dict[str, float]:
        """Busy fraction per pool over [0, total_ticks)."""
        if total_ticks <= 0:
            return {"dma": 0.0, "matrix": 0.0}
        return {
            "dma": self.dma.busy_ticks / (self.dma.n * total_ticks),
            "matrix": self.matrix.busy_ticks / (self.matrix.n * total_ticks),
        }
