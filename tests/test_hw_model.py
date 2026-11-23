import pytest

from matmul_sim import HWConfig
from matmul_sim.hw_model import HWError, HWModel, LocalMemory, TileState, UnitPool


# ---- UnitPool -------------------------------------------------------------

def test_pool_start_finish_and_history():
    p = UnitPool("dma", 2)
    assert p.free_count == 2
    s0 = p.start(cmd=10, now=0, duration=4)
    s1 = p.start(cmd=11, now=1, duration=4)
    assert (s0, s1) == (0, 1) and p.free_count == 0
    assert p.next_end() == 4
    with pytest.raises(HWError, match="no free unit"):
        p.start(cmd=12, now=1, duration=4)
    assert p.finish(s0, now=4) == 10
    assert p.free_count == 1 and p.busy_ticks == 4
    assert p.start(cmd=12, now=4, duration=4) == 0  # freed slot is reused
    assert p.history[0].cmd == 10 and (p.history[0].start, p.history[0].end) == (0, 4)


def test_pool_finish_errors():
    p = UnitPool("matrix", 1)
    with pytest.raises(HWError, match="already free"):
        p.finish(0, now=0)
    p.start(cmd=1, now=0, duration=1)
    with pytest.raises(HWError, match="ends at 1"):
        p.finish(0, now=5)
    with pytest.raises(HWError, match="duration"):
        UnitPool("matrix", 1).start(cmd=1, now=0, duration=0)


def test_pool_needs_units():
    with pytest.raises(ValueError):
        UnitPool("dma", 0)


# ---- LocalMemory ----------------------------------------------------------

def test_memory_lifecycle():
    m = LocalMemory(2)
    m.allocate("A", remaining_uses=1, now=0)
    assert m.used == 1 and m.entry("A").state is TileState.PENDING
    assert not m.is_ready("A") and m.is_resident("A")
    m.mark_ready("A", now=4)
    assert m.is_ready("A")
    m.pin("A")
    assert m.evictable() == []
    assert m.unpin("A") == 0
    m.free_tile("A", now=5)
    assert m.used == 0 and m.peak == 1


def test_memory_capacity_enforced():
    m = LocalMemory(1)
    m.allocate("A", 1, now=0)
    assert not m.can_allocate()
    with pytest.raises(HWError, match="full"):
        m.allocate("B", 1, now=0)


def test_memory_double_allocate():
    m = LocalMemory(4)
    m.allocate("A", 1, now=0)
    with pytest.raises(HWError, match="already resident"):
        m.allocate("A", 1, now=0)


def test_cannot_read_pending_or_drop_pending_or_pinned():
    m = LocalMemory(4)
    m.allocate("A", 2, now=0)
    with pytest.raises(HWError, match="not READY"):
        m.pin("A")
    with pytest.raises(HWError, match="still being written"):
        m.evict("A", now=0)
    m.mark_ready("A", now=4)
    m.pin("A")
    with pytest.raises(HWError, match="read by 1"):
        m.free_tile("A", now=4)


def test_evict_returns_remaining_uses():
    m = LocalMemory(4)
    m.allocate("B", 3, now=0)
    m.mark_ready("B", now=4)
    m.pin("B")
    m.unpin("B")
    assert m.evictable() == ["B"]
    assert m.evict("B", now=5) == 2
    assert not m.is_resident("B")
    assert [e.kind for e in m.events] == ["alloc", "ready", "evict"]


def test_unpin_errors_and_add_uses():
    m = LocalMemory(4)
    m.allocate("A", 0, now=0)
    m.mark_ready("A", now=1)
    with pytest.raises(HWError, match="not pinned"):
        m.unpin("A")
    m.pin("A")
    with pytest.raises(HWError, match="no remaining uses"):
        m.unpin("A")
    m.add_uses("A", 2)
    assert m.unpin("A") == 1
    with pytest.raises(HWError, match="negative"):
        m.add_uses("A", -5)
    with pytest.raises(HWError, match="not resident"):
        m.entry("Z")


def test_occupancy_series_collapses_same_tick():
    m = LocalMemory(4)
    m.allocate("A", 1, now=0)
    m.allocate("B", 1, now=0)
    m.mark_ready("A", now=4)   # no change in count
    m.mark_ready("B", now=4)
    m.allocate("P", 1, now=4)
    for t in ("A", "B"):
        m.pin(t)
    for t in ("A", "B"):
        m.unpin(t)
    m.mark_ready("P", now=5)
    m.free_tile("A", now=5)
    m.free_tile("B", now=5)
    assert m.occupancy_series() == [(0, 2), (4, 3), (5, 1)]


# ---- HWModel: hand-driven single-tile matmul ------------------------------

def test_single_tile_matmul_by_hand():
    """LOAD A, LOAD B (ticks 0-4), COMPUTE P (4-5), SAVE P (5-9): 9 ticks = 1152 cycles."""
    hw = HWModel(HWConfig(n_dma=2, n_matrix=1, local_mem_tiles=3))
    mem = hw.memory
    assert hw.duration("dma") == 4 and hw.duration("matrix") == 1

    # tick 0: both LOADs
    slots = {}
    for cmd, tile in ((0, "A"), (1, "B")):
        mem.allocate(tile, remaining_uses=1, now=0)
        slots[cmd] = hw.dma.start(cmd, now=0, duration=hw.duration("dma"))
    assert hw.next_event() == 4

    # tick 4: LOADs finish, COMPUTE starts
    for cmd, tile in ((0, "A"), (1, "B")):
        hw.dma.finish(slots[cmd], now=4)
        mem.mark_ready(tile, now=4)
    mem.allocate("P", remaining_uses=1, now=4)
    mem.pin("A"); mem.pin("B")
    s = hw.matrix.start(2, now=4, duration=1)
    assert mem.used == 3 and not mem.can_allocate()

    # tick 5: COMPUTE finishes; inputs freed; SAVE starts
    hw.matrix.finish(s, now=5)
    mem.mark_ready("P", now=5)
    for t in ("A", "B"):
        assert mem.unpin(t) == 0
        mem.free_tile(t, now=5)
    mem.pin("P")
    s = hw.dma.start(3, now=5, duration=4)

    # tick 9: SAVE finishes; output freed
    hw.dma.finish(s, now=9)
    mem.unpin("P")
    mem.free_tile("P", now=9)

    assert hw.idle() and hw.next_event() is None
    total = 9
    assert total * hw.config.tick_cycles == 1152
    assert hw.utilization(total) == {"dma": pytest.approx(12 / 18), "matrix": pytest.approx(1 / 9)}
    assert mem.occupancy_series() == [(0, 2), (4, 3), (5, 1), (9, 0)]
    assert mem.peak == 3


def test_pool_lookup():
    hw = HWModel(HWConfig(n_dma=1, n_matrix=2, local_mem_tiles=4))
    assert hw.pool("dma").n == 1 and hw.pool("matrix").n == 2
    with pytest.raises(ValueError):
        hw.pool("gpu")
    assert hw.utilization(0) == {"dma": 0.0, "matrix": 0.0}
