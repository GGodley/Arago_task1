from collections import defaultdict

import pytest

from matmul_sim import HWConfig
from matmul_sim.bounds import lower_bound
from matmul_sim.schedulers import FIFOScheduler, make_scheduler
from matmul_sim.simulator import simulate
from matmul_sim.translator import MatmulSpec, translate


class EvictFirst(FIFOScheduler):
    """Test-only: evicts the first candidate. Naive on purpose (can thrash)."""

    name = "evict_first"
    evicts = True

    def victim(self, sim, candidates, cmd):
        return candidates[0]


def run(mkn, nd, nm, mem, sched=None):
    trace = translate(MatmulSpec(*mkn))
    return trace, simulate(trace, HWConfig(n_dma=nd, n_matrix=nm, local_mem_tiles=mem), sched or FIFOScheduler())


def check_invariants(trace, r):
    """Properties every finished run must satisfy."""
    cap = r.config.local_mem_tiles
    assert all(used <= cap for _, used in r.mem_series)
    assert r.mem_peak <= cap
    # One command per unit slot at a time.
    by_slot = defaultdict(list)
    for c in r.schedule:
        by_slot[(c.unit, c.slot)].append((c.start, c.end))
    for intervals in by_slot.values():
        intervals.sort()
        for (s0, e0), (s1, _) in zip(intervals, intervals[1:]):
            assert e0 <= s1
    # Every input was written (by a finished command) before the command reading it started.
    written_at = defaultdict(list)
    for c in r.schedule:
        if c.dst:
            written_at[c.dst].append(c.end)
    for c in r.schedule:
        for s in c.srcs:
            assert any(t <= c.start for t in written_at[s]), (c, s)
    # Latencies.
    for c in r.schedule:
        assert c.end - c.start == (r.config.dma_ticks if c.unit == "dma" else r.config.matrix_ticks)
    # Work done matches the trace (plus reloads).
    expected = trace.hw_op_counts()
    expected["LOAD"] += r.reloads
    assert r.commands == expected
    assert sum(1 for c in r.schedule if c.reload) == r.reloads
    # Never faster than the analytical bound.
    assert r.total_ticks >= r.bound.ticks
    assert r.total_cycles == r.total_ticks * 128


# ---- hand-checked timings ---------------------------------------------------

def test_single_tile_is_9_ticks():
    trace, r = run((64, 64, 64), nd=2, nm=1, mem=3)
    assert r.completed and r.total_ticks == 9 and r.total_cycles == 1152
    assert r.bound.ticks == 9 and r.bound_ratio == 1.0
    assert r.matrix_util == pytest.approx(1 / 9)
    assert r.dma_util == pytest.approx(12 / 18)
    assert r.mem_series == [(0, 2), (4, 3), (5, 1), (9, 0)]
    check_invariants(trace, r)


def test_single_tile_one_dma_serializes_loads():
    # LOAD A 0-4, LOAD B 4-8, COMPUTE 8-9, SAVE 9-13
    trace, r = run((64, 64, 64), nd=1, nm=1, mem=3)
    assert r.total_ticks == 13
    check_invariants(trace, r)


def test_one_acc_by_hand():
    # Nd=2: LOAD A00,B00 0-4 | LOAD A01,B10 4-8 ; P0 4-5 ; P1 8-9 ; ACC 9-10 ; SAVE 10-14
    trace, r = run((64, 128, 64), nd=2, nm=1, mem=8)
    assert r.total_ticks == 14
    kinds = [(c.kind, c.start) for c in sorted(r.schedule, key=lambda c: (c.start, c.kind))]
    assert ("ACC", 9) in kinds and ("SAVE", 10) in kinds
    check_invariants(trace, r)


def test_acc_pairs_like_a_tree_with_enough_units():
    # Kt=8, 17 transfers on 17 DMA units: all loads 0-4, 8 COMPUTEs 4-5,
    # ACC levels 5-6, 6-7, 7-8, SAVE 8-12 = 12 ticks = the path bound.
    trace, r = run((64, 512, 64), nd=17, nm=8, mem=64)
    assert r.total_ticks == 12 == r.bound.path
    acc_starts = sorted(c.start for c in r.schedule if c.kind == "ACC")
    assert acc_starts == [5, 5, 5, 5, 6, 6, 7]
    check_invariants(trace, r)


def test_acc_pairs_like_a_chain_with_one_unit():
    trace, r = run((64, 512, 64), nd=17, nm=1, mem=64)
    # 15 matrix commands back to back after the loads, then the SAVE.
    assert r.total_ticks == 4 + 15 + 4
    check_invariants(trace, r)


# ---- deadlock and thrashing --------------------------------------------------

def test_fifo_deadlocks_when_memory_fills_with_inputs():
    trace, r = run((128, 128, 128), nd=1, nm=1, mem=4)
    assert r.deadlock and not r.completed and not r.stalled
    assert r.ops_done < r.ops_total
    assert any("free memory slot" in s for s in r.stuck)
    assert r.stuck[-1] == "memory 4/4 tiles in use"


def test_same_workload_completes_with_enough_memory():
    trace, r = run((128, 128, 128), nd=1, nm=1, mem=12)
    assert r.completed and not r.deadlock
    check_invariants(trace, r)


def test_eviction_and_reload_complete_the_run():
    trace, r = run((128, 128, 128), nd=2, nm=2, mem=8, sched=EvictFirst())
    assert r.completed and r.reloads > 0 and r.evictions >= r.reloads
    check_invariants(trace, r)


def test_naive_eviction_can_thrash_and_is_reported_as_stalled():
    trace = translate(MatmulSpec(128, 128, 128))
    r = simulate(trace, HWConfig(1, 1, 4), EvictFirst(), max_ticks=500)
    assert r.stalled and not r.completed and not r.deadlock
    assert r.reloads > 50


# ---- sweeps: invariants and the bound everywhere ------------------------------

@pytest.mark.parametrize("mkn", [(64, 64, 64), (128, 128, 128), (256, 256, 256), (100, 200, 300), (64, 640, 64), (256, 64, 256)])
@pytest.mark.parametrize("nd, nm", [(1, 1), (2, 4), (8, 2), (16, 16)])
def test_invariants_with_ample_memory(mkn, nd, nm):
    spec = MatmulSpec(*mkn)
    trace, r = run(mkn, nd, nm, mem=spec.footprint_tiles + spec.tiles[0] * spec.tiles[2] * spec.tiles[1])
    assert r.completed
    check_invariants(trace, r)


@pytest.mark.parametrize("mem", [3, 4, 6, 8, 12, 16, 24])
def test_small_memory_either_completes_correctly_or_deadlocks(mem):
    trace, r = run((256, 256, 256), nd=2, nm=2, mem=mem)
    if r.completed:
        check_invariants(trace, r)
    else:
        assert r.deadlock


def test_medium_run_close_to_bound():
    trace, r = run((512, 512, 512), nd=8, nm=4, mem=512)
    assert r.completed
    check_invariants(trace, r)
    assert r.bound_ratio < 1.25


# ---- bound and misc ---------------------------------------------------------

def test_lower_bound_terms():
    b = lower_bound(translate(MatmulSpec(128, 128, 128)), HWConfig(n_dma=2, n_matrix=2, local_mem_tiles=16))
    # 8 COMPUTE + 4 ACC = 12 matrix cmds -> 4 + 6 + 4 ; 8 LOAD + 4 SAVE = 12 transfers -> 6*4 ; path 4+1+1+4
    assert (b.matrix, b.dma, b.path) == (14, 24, 10)
    assert b.ticks == 24 and b.limiter == "dma"


def test_mem_mean_pct():
    _, r = run((64, 64, 64), nd=2, nm=1, mem=3)
    # (2*4 + 3*1 + 1*4) / 9 ticks / 3 tiles
    assert r.mem_mean_pct == pytest.approx((8 + 3 + 4) / 9 / 3 * 100)


def test_summary_and_registry():
    _, r = run((128, 128, 128), nd=2, nm=2, mem=16)
    s = r.summary()
    assert s["completed"] and s["cmd_compute"] == 8 and s["scheduler"] == "fifo"
    assert make_scheduler("fifo").name == "fifo"
    with pytest.raises(ValueError):
        make_scheduler("magic")
