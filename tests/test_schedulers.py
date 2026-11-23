import pytest

from matmul_sim import HWConfig
from matmul_sim.bounds import min_memory_tiles
from matmul_sim.schedulers import (
    GreedyReuseScheduler, MemoryAwareScheduler, make_scheduler,
)
from matmul_sim.schedulers.reuse import output_rank
from matmul_sim.simulator import simulate
from matmul_sim.translator import MatmulSpec, translate
from tests.test_simulator import check_invariants

SHAPES = [(128, 128, 128), (256, 256, 256), (64, 512, 64), (256, 64, 256), (100, 200, 300), (192, 128, 320)]


def sim(mkn, nd, nm, mem, name="memaware", order="row"):
    trace = translate(MatmulSpec(*mkn))
    return trace, simulate(trace, HWConfig(nd, nm, mem), make_scheduler(name, order))


# ---- tile orders ------------------------------------------------------------

@pytest.mark.parametrize("order", ["row", "snake", "blocked"])
@pytest.mark.parametrize("Mt, Nt", [(1, 1), (3, 5), (4, 4), (5, 3), (7, 9)])
def test_output_rank_is_a_permutation(order, Mt, Nt):
    ranks = [output_rank(i, j, Mt, Nt, order, 2) for i in range(Mt) for j in range(Nt)]
    assert sorted(ranks) != [] and len(set(ranks)) == Mt * Nt


def test_output_rank_orders():
    order = lambda o: [ij for _, ij in sorted((output_rank(i, j, 2, 3, o, 2), (i, j)) for i in range(2) for j in range(3))]
    assert order("row") == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert order("snake") == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    assert order("blocked") == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)]


def test_registry_and_names():
    assert make_scheduler("greedy").name == "greedy"
    assert make_scheduler("memaware", "snake").name == "memaware-snake"
    assert isinstance(make_scheduler("memaware", "blocked", 2), MemoryAwareScheduler)
    with pytest.raises(ValueError):
        GreedyReuseScheduler(order="spiral")


# ---- memory-aware: never deadlocks at or above the minimum memory -------------

@pytest.mark.parametrize("mkn", SHAPES)
@pytest.mark.parametrize("order", ["row", "snake", "blocked"])
def test_memaware_completes_from_minimum_memory(mkn, order):
    trace = translate(MatmulSpec(*mkn))
    lo = min_memory_tiles(trace)
    for mem in (lo, lo + 1, lo + 2, 8, 12, 16, 32):
        for nd, nm in ((1, 1), (2, 2), (4, 8)):
            r = simulate(trace, HWConfig(nd, nm, mem), make_scheduler("memaware", order))
            assert r.completed, (mkn, order, mem, nd, nm, r.stuck)
            check_invariants(trace, r)


@pytest.mark.parametrize("name", ["fifo", "greedy", "memaware"])
def test_below_minimum_memory_is_infeasible(name):
    trace, r = sim((128, 128, 128), 2, 2, 3, name)
    assert min_memory_tiles(trace) == 4
    assert r.deadlock and r.infeasible_memory


def test_kt1_minimum_is_three_tiles():
    trace, r = sim((256, 64, 256), 2, 2, 3)
    assert min_memory_tiles(trace) == 3 and r.completed and not r.infeasible_memory


# ---- greedy vs memory-aware ---------------------------------------------------

@pytest.mark.parametrize("mem", [4, 6, 8, 16])
def test_greedy_deadlocks_where_memaware_completes(mem):
    _, g = sim((256, 256, 256), 2, 2, mem, "greedy")
    trace, m = sim((256, 256, 256), 2, 2, mem, "memaware")
    assert g.deadlock and not g.infeasible_memory
    assert m.completed
    check_invariants(trace, m)


def test_greedy_reuses_better_than_fifo_with_ample_memory():
    _, f = sim((512, 512, 512), 4, 8, 256, "fifo")
    trace, g = sim((512, 512, 512), 4, 8, 256, "greedy")
    assert g.completed and f.completed
    assert g.total_ticks <= f.total_ticks
    check_invariants(trace, g)


@pytest.mark.parametrize("name", ["greedy", "memaware"])
@pytest.mark.parametrize("order", ["row", "snake", "blocked"])
def test_close_to_bound_with_ample_memory(name, order):
    trace, r = sim((512, 512, 512), 8, 4, 512, name, order)
    assert r.completed and r.reloads == 0
    assert r.bound_ratio < 1.05
    check_invariants(trace, r)


def test_more_memory_means_fewer_reloads():
    trace = translate(MatmulSpec(512, 512, 512))
    reloads = [simulate(trace, HWConfig(2, 2, m), make_scheduler("memaware")).reloads for m in (4, 16, 64, 256)]
    assert reloads[0] > reloads[1] > reloads[2] >= reloads[3] == 0


def test_blocked_order_cuts_reloads_when_memory_is_tight():
    trace = translate(MatmulSpec(1024, 1024, 1024))
    row = simulate(trace, HWConfig(16, 16, 128), make_scheduler("memaware", "row"))
    blk = simulate(trace, HWConfig(16, 16, 128), make_scheduler("memaware", "blocked"))
    assert row.completed and blk.completed
    assert blk.reloads < row.reloads and blk.total_ticks < row.total_ticks


@pytest.mark.parametrize("order, nd, nm, mem", [("row", 1, 2, 32), ("blocked", 1, 1, 16), ("blocked", 1, 1, 32)])
def test_memaware_single_dma_tight_memory(order, nd, nm, mem):
    # Regression: an opportunistic COMPUTE that may not evict used to mark memory
    # blocked and hold back the urgent reloads, deadlocking with one DMA unit.
    trace, r = sim((512, 512, 512), nd, nm, mem, order=order)
    assert r.completed, r.stuck
    check_invariants(trace, r)
