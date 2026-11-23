# Arago Task 1: Blocked matmul performance model

A performance model for large matrix multiplications on an accelerator made of
global memory, `Nd` DMA units, a small local memory and `Nm` matrix units. It
takes a workload (a trace of operations) and a hardware configuration, and
reports matrix-unit utilization, local memory occupancy over time and total
cycles. A sweep script then searches the hardware parameters for each workload.

Status: **Phase 1 (translator)**. The trace and HW config formats and the
workload generator are in place. The simulator itself comes in later phases
(see [Build plan](#build-plan)).

## Quick start

```bash
pip install -e ".[test]"   # or just: pip install numpy matplotlib pytest
python -m pytest
```

## Generating workloads

```bash
python scripts/gen_workload.py --mkn 512 512 512                 # one matmul -> results/traces/
python scripts/gen_workload.py --mkn 100 200 300 --format csv --out my.csv
python scripts/gen_workload.py --preset small                    # a workload class
python scripts/gen_workload.py --preset all                      # every preset
```

The translator only **slices and expands**. It knows nothing about the
hardware, so it makes no ordering or pairing decisions: ops are written in a
fixed i, j, k loop order that carries no meaning, and each output tile's ACC
is a group whose pairing is decided at run time (see [ACC groups](#acc-groups)).
All ordering is the scheduler's job.

Dimensions that are not multiples of 64 are padded up. The trace metadata
records `useful_fraction` (real work / padded work) and `footprint_tiles`
(distinct A, B and C tiles).

Hardware command counts for a matmul of `Mt x Kt x Nt` tiles
(`Trace.hw_op_counts()`):

| Op | Count |
|---|---|
| LOAD | Mt·Kt + Kt·Nt |
| COMPUTE | Mt·Nt·Kt |
| ACC | Mt·Nt·(Kt-1) |
| SAVE | Mt·Nt |

## Workload classes

Classes are defined against the hardware through two ratios: output tiles
`Mt·Nt` vs `Nm` (parallelism) and footprint vs local memory (reuse).

| Class | Presets | Footprint (tiles) | Character |
|---|---|---|---|
| small | 64³, 128³, 256³ | 3 to 48 | Fits in memory; latency-bound; few parallel ops |
| medium | 512³, 1024³ | 192 to 768 | Around the swept memory sizes; the regime flips |
| large | 2048³, 4096³ | 3,072 to 12,288 | Far beyond memory; bandwidth- and reuse-bound |
| skewed | 64x4096x4096, 4096x64x4096, 64x8192x64, 100x200x300 | 38 to 4,224 | Decode-like, Kt = 1 (no ACC), long ACC chain, padding |

The largest preset (4096³) is 532k ops and generates in a few seconds.

## Architecture

```
  matmul dims (M,K,N)
        |
  [1 Translator] --> trace file (JSON or CSV): ops + dependencies
        |
  [4 Simulator] -- event loop, 1 tick = 128 cycles ----+
        |   asks: what can I issue now?                |
  [2 Scheduler] (swappable policy)                     |
        |   reads/updates                              |
  [3 HW model] scoreboard: units, memory, residency    |
        |                                              |
   results (metrics + event log) ----------------------+
        |
  [Sweep / Optimizer] grid over (Nd, Nm, memory), cost weights as parameters
        |
  [5 Analyzer] plots + tables for the report
```

| Block | Role |
|---|---|
| Translator | Slices `C = A x B` into 64x64 tile ops and assigns dependencies. No HW knowledge, no ordering. Writes the trace. |
| Scheduler | Picks which ready ops to issue each tick, which two ready tiles each ACC adds, and which tile to evict when memory is full. Owns all ordering strategies (FIFO, greedy reuse-first, memory-aware, row/snake/blocked tile orders, ...). Swappable. |
| HW model | Passive scoreboard: DMA and matrix unit slots, local memory slots, tile residency. |
| Simulator | Event-driven clock. Retires ops, frees memory, inserts reload LOADs after evictions, asks the scheduler for work, detects deadlock. |
| Analyzer | Plots and tables from simulator results. |

## Time and hardware model

Fixed by the brief:

| Quantity | Value |
|---|---|
| Tile size (matrix units and DMA) | 64 x 64 |
| Matrix-unit op latency `Lm` | 128 cycles |
| DMA transfer latency `Ld` | 512 cycles |
| Local memory latency | 0 |
| Global memory size | infinite |

Variable: `Nd` (DMA units), `Nm` (matrix units), local memory size in tiles.

- **Tick.** 1 tick = 128 cycles. COMPUTE and ACC take 1 tick, LOAD and SAVE take
  4 ticks. All latencies are whole ticks, so jumping between events is exact.
  Results are reported in cycles.
- **Matrix unit.** Performs COMPUTE or ACC, one at a time. The result is written
  to local memory at the end of the tick, and the next command can start on the
  next tick.
- **DMA unit.** Performs a read (LOAD) or a write (SAVE), one at a time. A
  transfer must finish before that unit starts another. All `Nd` units form one
  shared pool.
- **Local memory.** Counted in tiles. One 64x64 tile is the base unit of the
  whole system. A slot is allocated when the op writing it is issued, and freed
  when its last consumer finishes, or when its SAVE completes for final results.

### Command set

| Op | Unit | Ticks | Reads | Writes |
|---|---|---|---|---|
| `LOAD X` | DMA | 4 | global tile X | new local tile X |
| `COMPUTE P = A x B` | matrix | 1 | local A, B | new local tile P |
| `ACC R = X + Y` | matrix | 1 | any two ready tiles X, Y of the same output tile's group | new local tile R (X, Y freed on completion) |
| `SAVE R` | DMA | 4 | local R | global memory (local slot freed when done) |

Each output tile `C[i,j]` takes `Kt` COMPUTEs, `Kt-1` ACCs and one SAVE, where
`Kt = ceil(K/64)`. Fused multiply-accumulate is not modelled. It is discussed in
the report as a possible HW improvement.

### Minimum memory

Processing one output tile strictly in order needs 3 tiles when `Kt = 1` (A, B,
P) and 4 when `Kt > 1` (running sum, A, B, new partial). Below that a workload
is infeasible. The memory sweep starts there.

## Trace format

A trace says **what** must happen, not **when**. It lists each tile LOAD once
and does not depend on the HW config. When memory is too small, the scheduler
evicts tiles at run time, and the simulator inserts the reload LOADs and counts
them.

Every op has the same five fields:

| Field | Meaning |
|---|---|
| `id` | unique integer |
| `type` | `LOAD`, `COMPUTE`, `ACC` or `SAVE` |
| `srcs` | local tiles read (LOAD: none; COMPUTE: 2; ACC: 2 or more; SAVE: 1) |
| `dst` | local tile written (required except for SAVE, which writes to global memory) |
| `deps` | ids of ops that must finish first; must include the producer of every src |

Rules checked by `Trace.validate()`:

- ids are unique and every dep exists; no dependency cycles;
- arity per type as above, with no repeated src;
- each tile is written by exactly one op;
- every src is produced by some op, and that op is in `deps`;
- every loaded tile is used, and every COMPUTE/ACC result is used or SAVEd.

Tile naming used by the translator (any unique string is valid):

| Name | Meaning |
|---|---|
| `A_i_k`, `B_k_j` | input tiles |
| `P_i_j_k` | partial product `A_i_k x B_k_j` |
| `C_i_j` | final sum for output tile `C[i,j]` (dst of its ACC group; when `Kt = 1` the single partial is saved directly) |

### ACC groups

An ACC record is a **reduction group**: it lists every partial that sums into
one output tile, `C_i_j = P_i_j_0 + ... + P_i_j_{Kt-1}`, and leaves the
pairing open. The translator can't choose a pairing because it doesn't know
how many matrix units there are.

Rules for the scheduler and simulator (Phases 3 and 4):

1. A group of n srcs expands at run time into **n-1 binary ACC commands**,
   1 tick each on a matrix unit.
2. A command may add **any two ready tiles of the same group**: two partials,
   a partial and an earlier sum, or two earlier sums.
3. Each command writes a new local tile (allocated at issue) and frees its two
   inputs when it completes. The last command writes the group's `dst`.
4. A group's deps are satisfied **incrementally**: the first command can start
   as soon as any two srcs exist, not when all of them do.
5. With one free matrix unit the group behaves like a chain (Kt-1 ticks on the
   critical path). With many it can behave like a tree (ceil(log2 Kt) ticks).
   Which shape emerges is a result of the scheduler and HW config, not a
   translator option.

### JSON

```json
{
 "format_version": 1,
 "workload": {"name": "mm_64x128x64", "M": 64, "K": 128, "N": 64, "tile": 64},
 "ops": [
  {"id": 0, "type": "LOAD", "srcs": [], "dst": "A_0_0", "deps": []},
  {"id": 4, "type": "COMPUTE", "srcs": ["A_0_0", "B_0_0"], "dst": "P_0_0_0", "deps": [0, 1]},
  {"id": 6, "type": "ACC", "srcs": ["P_0_0_0", "P_0_0_1"], "dst": "C_0_0", "deps": [4, 5]},
  {"id": 7, "type": "SAVE", "srcs": ["C_0_0"], "dst": null, "deps": [6]}
 ]
}
```

### CSV

One op per row. List fields are `;`-separated. Workload metadata goes in
leading `# key=value` lines (values JSON-encoded).

```
# format_version=1
# name="mm_64x128x64"
id,type,srcs,dst,deps
0,LOAD,,A_0_0,
4,COMPUTE,A_0_0;B_0_0,P_0_0_0,0;1
6,ACC,P_0_0_0;P_0_0_1,C_0_0,4;5
7,SAVE,C_0_0,,6
```

Full examples: [`examples/mm_64x128x64.json`](examples/mm_64x128x64.json) and
[`examples/mm_64x128x64.csv`](examples/mm_64x128x64.csv).

## HW config format

```json
{
  "n_dma": 2,
  "n_matrix": 2,
  "local_mem_tiles": 16,
  "tick_cycles": 128,
  "matrix_ticks": 1,
  "dma_ticks": 4
}
```

`n_dma`, `n_matrix` and `local_mem_tiles` are required. The rest default to the
values from the brief. Unknown keys are rejected. Example:
[`examples/hw_default.json`](examples/hw_default.json).

## Assumptions

| # | Assumption |
|---|---|
| A1 | Local memory is measured in 64x64 tile slots. |
| A2 | Output slots are allocated at issue. Inputs and partials are freed after their last consumer. Final results are freed once SAVE reaches global memory. |
| A3 | One shared pool of `Nd` DMA units. Each does one read or write at a time. |
| A4 | ACC is a separate op. Fused MAC is only discussed as an improvement. |
| A5 | Each ACC command takes 2 same-size tiles and writes a new result tile. Which two is decided at run time within the output tile's group. |
| A6 | The trace has one LOAD per distinct tile. The scheduler decides keep or evict, and the simulator inserts reloads. |
| A7 | Every output tile must be SAVEd before the run ends. |
| A8 | Sweep: `Nm` 1..16, `Nd` 1..16, memory from 3 or 4 tiles up to 512. |
| A9 | Workloads: small 64-256, medium 512-1024, large 2048-4096 per side, plus skewed shapes. |
| A10 | Cost weights for the optimizer are parameters. |
| A11 | Local memory bandwidth is unconstrained (zero latency, one port per matrix unit), as in the brief. |

## Build plan

| Phase | Work | Status |
|---|---|---|
| 0 | Trace and HW config formats, validation, tests | done |
| 1 | Translator: tiling, padding, dependencies, ACC groups, trace writers | **this PR** |
| 2 | HW model: unit and memory slots, residency, event log | |
| 3 | Simulator + FIFO scheduler, deadlock detection, metrics, lower bound | |
| 4 | Greedy and memory-aware schedulers, eviction and reload | |
| 5 | Analyzer plots | |
| 6 | Sweep and optimizer | |
| 7 | Experiments, report, packaging | |

## Layout

```
matmul_sim/
  trace.py       # trace data model, JSON/CSV I/O, validation
  hw_config.py   # HW parameters and fixed constants
  translator.py  # matmul -> trace (tiling, padding, deps, ACC groups)
  workloads.py   # named presets: small, medium, large, skewed
scripts/
  gen_workload.py
examples/        # hand-written example trace and HW config
tests/
```
