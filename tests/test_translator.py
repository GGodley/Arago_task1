import subprocess
import sys
from pathlib import Path

import pytest

from matmul_sim import OpType, load_trace
from matmul_sim.translator import MatmulSpec, expected_op_counts, translate
from matmul_sim.workloads import PRESETS, all_presets, get_preset

ROOT = Path(__file__).resolve().parent.parent
SHAPES = [(64, 64, 64), (64, 128, 64), (128, 128, 128), (192, 64, 320), (100, 200, 300), (64, 640, 64)]


@pytest.mark.parametrize("mkn", SHAPES)
def test_valid_and_counts_match_formula(mkn):
    spec = MatmulSpec(*mkn)
    t = translate(spec)
    t.validate()
    Mt, Kt, Nt = spec.tiles
    assert t.hw_op_counts() == expected_op_counts(Mt, Kt, Nt)
    # One ACC group per output tile when Kt > 1, none otherwise.
    assert t.op_counts()["ACC"] == (Mt * Nt if Kt > 1 else 0)


def test_matches_hand_written_example():
    hand = load_trace(ROOT / "examples" / "mm_64x128x64.json")
    gen = translate(MatmulSpec(64, 128, 64))
    # Same ops; program order may differ (the translator loads lazily).
    assert {(o.type, o.srcs, o.dst) for o in gen.ops} == {(o.type, o.srcs, o.dst) for o in hand.ops}


def test_single_tile():
    t = translate(MatmulSpec(64, 64, 64))
    assert [op.type for op in t.ops] == [OpType.LOAD, OpType.LOAD, OpType.COMPUTE, OpType.SAVE]
    assert t.ops[-1].srcs == ("P_0_0_0",)


def test_kt1_has_no_acc():
    assert translate(MatmulSpec(256, 64, 256)).op_counts()["ACC"] == 0


def test_acc_group_lists_all_partials_without_pairing():
    t = translate(MatmulSpec(64, 5 * 64, 64))  # Kt = 5
    accs = [op for op in t.ops if op.type is OpType.ACC]
    assert len(accs) == 1
    acc = accs[0]
    assert acc.srcs == tuple(f"P_0_0_{k}" for k in range(5))
    assert acc.dst == "C_0_0"
    computes = [op.id for op in t.ops if op.type is OpType.COMPUTE]
    assert acc.deps == tuple(computes)
    assert t.ops[-1].srcs == ("C_0_0",)
    assert t.hw_op_counts()["ACC"] == 4


def test_padding():
    spec = MatmulSpec(100, 200, 300)
    assert spec.tiles == (2, 4, 5)
    assert spec.useful_fraction == pytest.approx(100 * 200 * 300 / (128 * 256 * 320))
    assert MatmulSpec(128, 128, 128).useful_fraction == 1.0


def test_each_input_loaded_once_and_producers_first():
    t = translate(MatmulSpec(256, 256, 256))
    loads = [op.dst for op in t.ops if op.type is OpType.LOAD]
    assert len(loads) == len(set(loads)) == 2 * 4 * 4
    for op in t.ops:
        assert all(d < op.id for d in op.deps)


def test_metadata():
    w = translate(MatmulSpec(100, 200, 300)).workload
    assert (w["Mt"], w["Kt"], w["Nt"]) == (2, 4, 5)
    assert w["footprint_tiles"] == 2 * 4 + 4 * 5 + 2 * 5
    assert "order" not in w and "acc" not in w


@pytest.mark.parametrize("mkn", [(0, 64, 64), (64, -1, 64), (64, 64, 1.5)])
def test_bad_spec(mkn):
    with pytest.raises(ValueError):
        MatmulSpec(*mkn)


def test_presets_unique_and_resolvable():
    assert len(all_presets()) == sum(len(v) for v in PRESETS.values())
    assert get_preset("medium_512")[0].M == 512
    assert len(get_preset("small")) == 3
    with pytest.raises(KeyError):
        get_preset("huge")


def test_cli_writes_valid_traces(tmp_path):
    for fmt in ("json", "csv"):
        out = tmp_path / f"t.{fmt}"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_workload.py"),
                        "--mkn", "100", "200", "300", "--format", fmt, "--out", str(out)],
                       check=True, capture_output=True)
        assert load_trace(out).hw_op_counts() == expected_op_counts(2, 4, 5)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_workload.py"),
                    "--preset", "small", "--outdir", str(tmp_path / "d")],
                   check=True, capture_output=True)
    assert len(list((tmp_path / "d").glob("*.json"))) == 3
