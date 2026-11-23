from pathlib import Path

import pytest

from matmul_sim import Op, OpType, Trace, TraceError, load_trace

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def small_trace() -> Trace:
    return Trace([
        Op(0, OpType.LOAD, (), "A_0_0"),
        Op(1, OpType.LOAD, (), "B_0_0"),
        Op(2, OpType.COMPUTE, ("A_0_0", "B_0_0"), "P_0_0_0", (0, 1)),
        Op(3, OpType.SAVE, ("P_0_0_0",), None, (2,)),
    ], {"name": "mm_64x64x64"})


def replace_op(trace: Trace, new: Op) -> Trace:
    return Trace([new if op.id == new.id else op for op in trace.ops], trace.workload)


def test_valid_trace_passes():
    small_trace().validate()


def test_op_counts():
    assert small_trace().op_counts() == {"LOAD": 2, "COMPUTE": 1, "ACC": 0, "SAVE": 1}


def test_acc_group_counts():
    t = Trace([
        Op(0, OpType.LOAD, (), "A"), Op(1, OpType.LOAD, (), "B"),
        Op(2, OpType.COMPUTE, ("A", "B"), "P0", (0, 1)),
        Op(3, OpType.COMPUTE, ("A", "B"), "P1", (0, 1)),
        Op(4, OpType.COMPUTE, ("A", "B"), "P2", (0, 1)),
        Op(5, OpType.ACC, ("P0", "P1", "P2"), "C", (2, 3, 4)),
        Op(6, OpType.SAVE, ("C",), None, (5,)),
    ])
    t.validate()
    assert t.op_counts()["ACC"] == 1
    assert t.hw_op_counts() == {"LOAD": 2, "COMPUTE": 3, "ACC": 2, "SAVE": 1}


def test_units():
    assert OpType.LOAD.unit == OpType.SAVE.unit == "dma"
    assert OpType.COMPUTE.unit == OpType.ACC.unit == "matrix"


@pytest.mark.parametrize("ext", ["json", "csv"])
def test_round_trip(tmp_path, ext):
    t = small_trace()
    path = tmp_path / f"t.{ext}"
    getattr(t, f"to_{ext}")(path)
    loaded = load_trace(path)
    assert loaded.ops == t.ops
    assert loaded.workload == t.workload


@pytest.mark.parametrize("name", ["mm_64x128x64.json", "mm_64x128x64.csv"])
def test_examples_load(name):
    t = load_trace(EXAMPLES / name)
    assert t.op_counts() == {"LOAD": 4, "COMPUTE": 2, "ACC": 1, "SAVE": 1}


def test_json_and_csv_examples_match():
    assert load_trace(EXAMPLES / "mm_64x128x64.json").ops == load_trace(EXAMPLES / "mm_64x128x64.csv").ops


@pytest.mark.parametrize("bad, message", [
    (Op(2, OpType.COMPUTE, ("A_0_0",), "P_0_0_0", (0,)), "expected 2 srcs"),
    (Op(2, OpType.ACC, ("A_0_0",), "P_0_0_0", (0,)), "expected at least 2 srcs"),
    (Op(2, OpType.ACC, ("A_0_0", "A_0_0"), "P_0_0_0", (0,)), "repeated src"),
    (Op(2, OpType.COMPUTE, ("A_0_0", "B_0_0"), None, (0, 1)), "missing dst"),
    (Op(3, OpType.SAVE, ("P_0_0_0",), "X", (2,)), "must not have a dst"),
    (Op(2, OpType.COMPUTE, ("A_0_0", "B_0_0"), "P_0_0_0", (0,)), "not in deps"),
    (Op(2, OpType.COMPUTE, ("A_0_0", "Z"), "P_0_0_0", (0, 1)), "never produced"),
    (Op(2, OpType.COMPUTE, ("A_0_0", "B_0_0"), "P_0_0_0", (0, 1, 9)), "does not exist"),
    (Op(1, OpType.LOAD, (), "A_0_0"), "written by both"),
])
def test_invalid_ops_rejected(bad, message):
    with pytest.raises(TraceError, match=message):
        replace_op(small_trace(), bad).validate()


def test_unsaved_result_rejected():
    t = Trace(small_trace().ops[:3])
    with pytest.raises(TraceError, match="never used or saved"):
        t.validate()


def test_unused_load_rejected():
    t = Trace(small_trace().ops + [Op(9, OpType.LOAD, (), "A_9_9")])
    with pytest.raises(TraceError, match="never used"):
        t.validate()


def test_duplicate_id_rejected():
    t = Trace(small_trace().ops + [Op(0, OpType.LOAD, (), "A_9_9")])
    with pytest.raises(TraceError, match="duplicate op id"):
        t.validate()


def test_cycle_detected():
    # LOAD 0 artificially waits on SAVE 3, which closes a loop through the data deps.
    t = replace_op(small_trace(), Op(0, OpType.LOAD, (), "A_0_0", (3,)))
    with pytest.raises(TraceError, match="dependency cycle"):
        t.validate()


def test_unknown_type_rejected(tmp_path):
    p = tmp_path / "t.json"
    p.write_text('{"format_version": 1, "ops": [{"id": 0, "type": "MAC"}]}')
    with pytest.raises(TraceError, match="unknown type"):
        load_trace(p)


def test_unknown_extension_rejected(tmp_path):
    with pytest.raises(TraceError, match="unsupported trace extension"):
        load_trace(tmp_path / "t.txt")
