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
