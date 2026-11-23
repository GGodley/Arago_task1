import subprocess
import sys
from pathlib import Path

import pytest

from matmul_sim import OpType, load_trace
from matmul_sim.translator import (
    MatmulSpec, acc_depth, expected_op_counts, output_order, translate,
)
from matmul_sim.workloads import PRESETS, all_presets, get_preset

ROOT = Path(__file__).resolve().parent.parent
SHAPES = [(64, 64, 64), (64, 128, 64), (128, 128, 128), (192, 64, 320), (100, 200, 300), (64, 640, 64)]


@pytest.mark.parametrize("mkn", SHAPES)
@pytest.mark.parametrize("acc", ["chain", "tree"])
@pytest.mark.parametrize("order", ["row", "snake", "blocked"])
def test_valid_and_counts_match_formula(mkn, acc, order):
    spec = MatmulSpec(*mkn)
    t = translate(spec, order=order, acc=acc, block=(2, 2))
    t.validate()
    assert t.op_counts() == expected_op_counts(*spec.tiles)


def test_matches_hand_written_example():
    # Same ops, possibly in a different program order (the translator loads lazily).
    hand = load_trace(ROOT / "examples" / "mm_64x128x64.json")
    gen = translate(MatmulSpec(64, 128, 64))
    assert {(o.type, o.srcs, o.dst) for o in gen.ops} == {(o.type, o.srcs, o.dst) for o in hand.ops}


def test_single_tile():
    t = translate(MatmulSpec(64, 64, 64))
    assert [op.type for op in t.ops] == [OpType.LOAD, OpType.LOAD, OpType.COMPUTE, OpType.SAVE]
    assert t.ops[-1].srcs == ("P_0_0_0",)


def test_padding():
    spec = MatmulSpec(100, 200, 300)
    assert spec.tiles == (2, 4, 5)
    assert spec.useful_fraction == pytest.approx(100 * 200 * 300 / (128 * 256 * 320))
    assert MatmulSpec(128, 128, 128).useful_fraction == 1.0


def test_kt1_has_no_acc():
    t = translate(MatmulSpec(256, 64, 256))
    assert t.op_counts()["ACC"] == 0


def test_chain_acc_is_sequential():
    t = translate(MatmulSpec(64, 256, 64), acc="chain")
    accs = [op for op in t.ops if op.type is OpType.ACC]
    assert [op.dst for op in accs] == ["S_0_0_1", "S_0_0_2", "S_0_0_3"]
    assert accs[1].srcs[0] == "S_0_0_1" and accs[2].srcs[0] == "S_0_0_2"


def test_tree_acc_shape():
    t = translate(MatmulSpec(64, 5 * 64, 64), acc="tree")  # Kt = 5
    accs = [op for op in t.ops if op.type is OpType.ACC]
    assert [op.dst for op in accs] == ["T_0_0_0_0", "T_0_0_0_1", "T_0_0_1_0", "T_0_0_2_0"]
    assert accs[-1].srcs == ("T_0_0_1_0", "P_0_0_4")
    assert t.ops[-1].srcs == ("T_0_0_2_0",)


@pytest.mark.parametrize("kt, chain, tree", [(1, 0, 0), (2, 1, 1), (5, 4, 3), (8, 7, 3), (64, 63, 6)])
def test_acc_depth(kt, chain, tree):
    assert acc_depth(kt, "chain") == chain
    assert acc_depth(kt, "tree") == tree


def test_each_input_loaded_once_and_before_use():
    t = translate(MatmulSpec(256, 256, 256), order="snake")
    loads = [op.dst for op in t.ops if op.type is OpType.LOAD]
    assert len(loads) == len(set(loads))
    for op in t.ops:  # program order: producers come before consumers
        assert all(d < op.id for d in op.deps)


@pytest.mark.parametrize("order, expected", [
    ("row", [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]),
    ("snake", [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]),
])
def test_output_orders(order, expected):
    assert list(output_order(2, 3, order)) == expected


def test_blocked_order():
    got = list(output_order(3, 3, "blocked", (2, 2)))
    assert got == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert sorted(got) == sorted(output_order(3, 3, "row"))


def test_metadata():
    w = translate(MatmulSpec(100, 200, 300), order="blocked", block=(2, 3)).workload
    assert (w["Mt"], w["Kt"], w["Nt"]) == (2, 4, 5)
    assert w["footprint_tiles"] == 2 * 4 + 4 * 5 + 2 * 5
    assert w["block"] == [2, 3]


@pytest.mark.parametrize("bad", [dict(order="zigzag"), dict(acc="fused"), dict(order="blocked", block=(0, 2))])
def test_bad_options(bad):
    with pytest.raises(ValueError):
        translate(MatmulSpec(128, 128, 128), **bad)


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
        assert load_trace(out).op_counts() == expected_op_counts(2, 4, 5)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_workload.py"),
                        "--preset", "small", "--order", "snake", "--outdir", str(tmp_path / "d")],
                       check=True, capture_output=True)
    assert len(list((tmp_path / "d").glob("*.json"))) == 3
