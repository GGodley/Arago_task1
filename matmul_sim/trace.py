"""Workload trace format: data model, JSON/CSV I/O and validation.

A trace is the contract between the translator (workload generator) and the
simulator. It describes *what* has to happen, not *when*: the scheduler decides
the order at run time, and the simulator may insert reload LOADs when the
scheduler evicts a tile. A trace is therefore independent of the HW config.

Every op has the same shape:

    id    unique integer
    type  LOAD | COMPUTE | ACC | SAVE
    srcs  local tiles the op reads
    dst   local tile the op writes (None for SAVE, which writes to global memory)
    deps  ids of ops that must finish before this op may start

Arity per type:

    LOAD     srcs = []              dst = tile  (global -> local, DMA)
    COMPUTE  srcs = [A, B]          dst = P     (P = A x B, matrix unit)
    ACC      srcs = [P0, ..., Pn-1] dst = C     (C = P0 + ... + Pn-1, n >= 2)
    SAVE     srcs = [C]             dst = None  (local -> global, DMA)

ACC is a *reduction group*, not a single hardware command. It lists every
partial that sums into one output tile and leaves the pairing open, because
the translator does not know how many matrix units there are. At run time
the group expands into n-1 binary ACC commands (1 tick each on a matrix
unit). Each command adds any two tiles of the group that are ready (partials
or earlier sums), writes a new tile and frees its two inputs. The last one
writes dst. So an ACC group's deps are satisfied incrementally: work can start
as soon as two of its srcs exist, not when all of them do.

Tiles are named by strings and each tile is written by exactly one op
(single assignment). Every src must be produced by an op listed in deps.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1


class OpType(str, Enum):
    LOAD = "LOAD"
    COMPUTE = "COMPUTE"
    ACC = "ACC"
    SAVE = "SAVE"

    @property
    def unit(self) -> str:
        """Which HW unit executes this op: 'dma' or 'matrix'."""
        return "dma" if self in (OpType.LOAD, OpType.SAVE) else "matrix"


# (min srcs, max srcs or None for unbounded, whether dst is required)
_ARITY = {
    OpType.LOAD: (0, 0, True),
    OpType.COMPUTE: (2, 2, True),
    OpType.ACC: (2, None, True),
    OpType.SAVE: (1, 1, False),
}


@dataclass(frozen=True)
class Op:
    id: int
    type: OpType
    srcs: tuple[str, ...] = ()
    dst: str | None = None
    deps: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "srcs": list(self.srcs),
            "dst": self.dst,
            "deps": list(self.deps),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Op":
        try:
            op_type = OpType(d["type"])
        except ValueError as e:
            raise TraceError(f"op {d.get('id')}: unknown type {d.get('type')!r}") from e
        return Op(
            id=int(d["id"]),
            type=op_type,
            srcs=tuple(d.get("srcs") or ()),
            dst=d.get("dst"),
            deps=tuple(int(x) for x in (d.get("deps") or ())),
        )


@dataclass
class Trace:
    ops: list[Op]
    # Free-form description of where the trace came from, e.g.
    # {"name": "mm_128x128x128", "M": 128, "K": 128, "N": 128, "tile": 64}.
    workload: dict[str, Any] = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    # ---- queries -------------------------------------------------------
    def op_counts(self) -> dict[str, int]:
        """Number of trace records per type (an ACC group counts once)."""
        counts = {t.value: 0 for t in OpType}
        for op in self.ops:
            counts[op.type.value] += 1
        return counts

    def hw_op_counts(self) -> dict[str, int]:
        """Number of hardware commands per type: an ACC group of n srcs is n-1 ACCs."""
        counts = {t.value: 0 for t in OpType}
        for op in self.ops:
            counts[op.type.value] += len(op.srcs) - 1 if op.type is OpType.ACC else 1
        return counts

    def by_id(self) -> dict[int, Op]:
        return {op.id: op for op in self.ops}

    # ---- validation ----------------------------------------------------
    def validate(self) -> None:
        """Raise TraceError if the trace is malformed. See module docstring."""
        errors: list[str] = []
        ids: dict[int, Op] = {}
        producer: dict[str, int] = {}

        for op in self.ops:
            if op.id in ids:
                errors.append(f"duplicate op id {op.id}")
            ids[op.id] = op

            lo, hi, needs_dst = _ARITY[op.type]
            n = len(op.srcs)
            if n < lo or (hi is not None and n > hi):
                want = f"{lo}" if lo == hi else f"at least {lo}"
                errors.append(f"op {op.id} ({op.type.value}): expected {want} srcs, got {n}")
            if len(set(op.srcs)) != n:
                errors.append(f"op {op.id} ({op.type.value}): repeated src")
            if needs_dst and not op.dst:
                errors.append(f"op {op.id} ({op.type.value}): missing dst")
            if not needs_dst and op.dst:
                errors.append(f"op {op.id} ({op.type.value}): must not have a dst")
            if op.dst:
                if op.dst in producer:
                    errors.append(f"tile {op.dst!r} written by both op {producer[op.dst]} and op {op.id}")
                producer[op.dst] = op.id

        consumed: set[str] = set()
        saved: set[str] = set()
        for op in self.ops:
            for d in op.deps:
                if d not in ids:
                    errors.append(f"op {op.id}: dep {d} does not exist")
                elif d == op.id:
                    errors.append(f"op {op.id}: depends on itself")
            for s in op.srcs:
                consumed.add(s)
                if op.type is OpType.SAVE:
                    saved.add(s)
                p = producer.get(s)
                if p is None:
                    errors.append(f"op {op.id}: src tile {s!r} is never produced")
                elif p not in op.deps:
                    errors.append(f"op {op.id}: src tile {s!r} is produced by op {p}, which is not in deps")

        # Results of compute must end up somewhere: consumed later or saved (A7).
        for op in self.ops:
            if op.type in (OpType.COMPUTE, OpType.ACC) and op.dst and op.dst not in consumed:
                errors.append(f"op {op.id}: result {op.dst!r} is never used or saved")
            if op.type is OpType.LOAD and op.dst and op.dst not in consumed:
                errors.append(f"op {op.id}: loaded tile {op.dst!r} is never used")

        if not errors:
            cycle = _find_cycle(self.ops)
            if cycle:
                errors.append("dependency cycle: " + " -> ".join(map(str, cycle)))

        if errors:
            raise TraceError("invalid trace:\n  " + "\n  ".join(errors))

    # ---- I/O -----------------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        data = {
            "format_version": self.format_version,
            "workload": self.workload,
            "ops": [op.to_dict() for op in self.ops],
        }
        Path(path).write_text(json.dumps(data, indent=1) + "\n")

    def to_csv(self, path: str | Path) -> None:
        """CSV: one op per row; list fields are ';'-separated.

        Workload metadata goes in leading '#' comment lines as key=value.
        """
        with open(path, "w", newline="") as f:
            f.write(f"# format_version={self.format_version}\n")
            for k, v in self.workload.items():
                f.write(f"# {k}={json.dumps(v)}\n")
            w = csv.writer(f)
            w.writerow(["id", "type", "srcs", "dst", "deps"])
            for op in self.ops:
                w.writerow([
                    op.id,
                    op.type.value,
                    ";".join(op.srcs),
                    op.dst or "",
                    ";".join(map(str, op.deps)),
                ])


class TraceError(ValueError):
    pass


def load_trace(path: str | Path, validate: bool = True) -> Trace:
    """Load a trace from .json or .csv, chosen by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        trace = _load_json(path)
    elif suffix == ".csv":
        trace = _load_csv(path)
    else:
        raise TraceError(f"unsupported trace extension {suffix!r} (use .json or .csv)")
    if trace.format_version != FORMAT_VERSION:
        raise TraceError(f"unsupported format_version {trace.format_version} (expected {FORMAT_VERSION})")
    if validate:
        trace.validate()
    return trace


def _load_json(path: Path) -> Trace:
    data = json.loads(path.read_text())
    return Trace(
        ops=[Op.from_dict(d) for d in data["ops"]],
        workload=data.get("workload", {}),
        format_version=int(data.get("format_version", FORMAT_VERSION)),
    )


def _load_csv(path: Path) -> Trace:
    workload: dict[str, Any] = {}
    version = FORMAT_VERSION
    body: list[str] = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition("=")
                if key == "format_version":
                    version = int(value)
                elif key:
                    workload[key] = json.loads(value)
            else:
                body.append(line)

    def split(s: str) -> list[str]:
        return [x for x in s.split(";") if x]

    ops = []
    for row in csv.DictReader(body):
        ops.append(Op.from_dict({
            "id": row["id"],
            "type": row["type"],
            "srcs": split(row["srcs"]),
            "dst": row["dst"] or None,
            "deps": split(row["deps"]),
        }))
    return Trace(ops=ops, workload=workload, format_version=version)


def _find_cycle(ops: list[Op]) -> list[int] | None:
    """Return one dependency cycle as a list of op ids, or None (iterative DFS)."""
    deps = {op.id: op.deps for op in ops}
    WHITE, GREY, BLACK = 0, 1, 2
    color = dict.fromkeys(deps, WHITE)
    for root in deps:
        if color[root] != WHITE:
            continue
        stack: list[tuple[int, int]] = [(root, 0)]
        path: list[int] = []
        while stack:
            node, i = stack.pop()
            if i == 0:
                color[node] = GREY
                path.append(node)
            children = deps[node]
            if i < len(children):
                stack.append((node, i + 1))
                child = children[i]
                if color[child] == GREY:
                    return path[path.index(child):] + [child]
                if color[child] == WHITE:
                    stack.append((child, 0))
            else:
                color[node] = BLACK
                path.pop()
    return None
