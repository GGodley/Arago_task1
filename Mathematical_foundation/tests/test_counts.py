"""The closed-form counts must agree with the trace the simulator actually runs."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Mathematical_foundation"))

from foundation import loads_with_memory, op_counts  # noqa: E402
from matmul_sim.translator import MatmulSpec, translate  # noqa: E402


@pytest.mark.parametrize("mkn", [(64, 64, 64), (128, 128, 128), (256, 256, 256), (100, 200, 300),
                                 (64, 4096, 64), (512, 64, 256)])
def test_counts_match_translator(mkn):
    c = op_counts(*mkn)
    t = translate(MatmulSpec(*mkn)).hw_op_counts()
    assert (c.COMPUTE, c.ACC, c.LOAD, c.SAVE) == (t["COMPUTE"], t["ACC"], t["LOAD"], t["SAVE"])


def test_known_values():
    c = op_counts(4096, 4096, 4096)
    assert c.mac_total == 520_192 and c.dma_total == 12_288


def test_loads_with_memory_bounds():
    # enough memory: compulsory only; less memory: never fewer loads
    assert loads_with_memory(512, 512, 512, 512) == op_counts(512, 512, 512).LOAD
    assert loads_with_memory(512, 512, 512, 64) == 192          # 6-row panel
    assert loads_with_memory(512, 512, 512, 4) == 2 * 8 * 8 * 8  # no reuse at S = 4
    prev = None
    for S in (4, 8, 16, 32, 64, 128, 256):
        L = loads_with_memory(1024, 1024, 1024, S)
        assert prev is None or L <= prev
        prev = L


def test_timing_limits():
    from foundation import predict_time

    t = predict_time(64, 64, 64, 2, 1, 16)  # one tile: path-bound, 2 Ld + Lm
    assert t.T == 9 and t.limiter == "path"
    big = predict_time(4096, 4096, 4096, 32, 1, 1024)  # one matrix unit: T = 2 Ld + W
    assert big.limiter == "matrix" and big.T == 8 + 64 * 64 * 127
    slow = predict_time(4096, 4096, 4096, 1, 32, 1024)  # one DMA unit: DMA-bound
    assert slow.limiter == "dma"


def test_tilings_match_loads_and_cap():
    from foundation import loads_with_memory, tilings

    ts = tilings(4096, 4096, 4096)
    for S in (8, 64, 543, 4162):
        assert min(t.loads for t in ts if t.need <= S) == loads_with_memory(4096, 4096, 4096, S)
    b = next(t for t in ts if t.name == "block4x5")
    assert b.open == 20 and b.cap == 40


def test_profile_areas_match_counted_work():
    """Area under the usage curves equals the counted work (Step 1)."""
    import numpy as np

    from foundation.profile import profile

    for M, nd, nm, S in [(512, 8, 4, 128), (1024, 16, 16, 256)]:
        p = profile(M, M, M, nd, nm, S, points=20000)
        m = M // 64
        W = m * m * (2 * m - 1)
        busy = np.trapezoid(p["mat_pct"] / 100 * nm, p["t"])
        assert busy == pytest.approx(W, rel=0.02)
        assert p["mem_tiles"].max() <= S


def test_little_memory():
    """Little's-law memory: resident sums plus rate x hold time for in-flight tiles."""
    from foundation import little_memory, tilings

    c = op_counts(4096, 4096, 4096)
    t = next(x for x in tilings(4096, 4096, 4096) if x.name == "block13x13")
    assert t.resident == 169
    expect = 169 + (t.loads * 6 + c.COMPUTE * 2 + c.SAVE * 4) / 14992
    assert little_memory(t, c, 14992) == pytest.approx(expect)


@pytest.mark.parametrize("dims", [(64, 64, 64), (1, 1, 1), (100, 200, 300), (64, 8192, 64),
                                  (4096, 64, 4096), (64, 4096, 4096), (512, 512, 512)])
def test_optimizer_meets_every_target(dims):
    """Every optimizer cell finishes within its speed target and fits its memory."""
    from foundation.optimizer import grid

    g = grid(*dims, points=5)
    for i, target in enumerate(g["targets"]):
        for j, _S in enumerate(g["mems"]):
            cell = g["cells"][i][j]
            if cell:
                assert cell["T"] <= target + 1e-6
                assert cell["mem_needed_pct"] <= 100 + 1e-6
        assert any(g["cells"][i]), "every speed step has some memory that reaches it"


def test_single_tile_edge_case():
    """64^3 is one tile: path-bound at 9 ticks, 2 DMA units, 3 tiles of memory."""
    from foundation.optimizer import plan, s_min

    p = plan(64, 64, 64, 9, 4)
    assert (p["Nd"], p["Nm"]) == (2, 1)
    assert s_min(64, 64, 64, 9) == 3
