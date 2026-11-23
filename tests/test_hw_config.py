from pathlib import Path

import pytest

from matmul_sim import HWConfig
from matmul_sim.hw_config import LD_CYCLES, LM_CYCLES

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_defaults_match_brief():
    hw = HWConfig(n_dma=1, n_matrix=1, local_mem_tiles=4)
    assert hw.tick_cycles == LM_CYCLES == 128
    assert hw.matrix_ticks * hw.tick_cycles == LM_CYCLES
    assert hw.dma_ticks * hw.tick_cycles == LD_CYCLES == 512


@pytest.mark.parametrize("field", ["n_dma", "n_matrix", "local_mem_tiles"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rejects_non_positive_int(field, value):
    kwargs = dict(n_dma=1, n_matrix=1, local_mem_tiles=4)
    kwargs[field] = value
    with pytest.raises(ValueError):
        HWConfig(**kwargs)


def test_json_round_trip(tmp_path):
    hw = HWConfig(n_dma=3, n_matrix=5, local_mem_tiles=32)
    hw.to_json(tmp_path / "hw.json")
    assert HWConfig.from_json(tmp_path / "hw.json") == hw


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown HWConfig keys"):
        HWConfig.from_dict({"n_dma": 1, "n_matrix": 1, "local_mem_tiles": 4, "Nd": 2})


def test_example_loads():
    assert HWConfig.from_json(EXAMPLES / "hw_default.json").label == "Nd=2 Nm=2 mem=16"
