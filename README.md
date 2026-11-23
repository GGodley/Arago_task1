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
python scripts/gen_workload.py --preset small --order snake      # a workload class
python scripts/gen_workload.py --preset all                      # every preset
```

Options:

| Option | Values | Meaning |
|---|---|---|
| `--order` | `row` (default), `snake`, `blocked` | Order in which output tiles are visited. Sets the suggested program order only; execution is constrained by deps alone. |
| `--block BI BJ` | default `4 4` | Group size for `blocked`: groups of BI x BJ output tiles share BI rows of A and BJ columns of B. |
| `--acc` | `chain` (default), `tree` | Shape of the ACC reduction per output tile. `chain` puts Kt-1 ACCs on the critical path, `tree` puts ceil(log2 Kt). Same op count. |

Dimensions that are not multiples of 64 are padded up. The trace metadata
records `useful_fraction` (real work / padded work) and `footprint_tiles`
(distinct A, B and C tiles).

Op counts for a matmul of `Mt x Kt x Nt` tiles:

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
| Translator | Slices `C = A x B` into 64x64 tile ops and assigns dependencies. Writes the trace. |
| Scheduler | Picks which ready ops to issue each tick. Swappable (FIFO, greedy reuse-first, memory-aware, ...). Decides which tile to evict when memory is full. |
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
| `ACC R = S + P` | matrix | 1 | local S, P | new local tile R (S, P freed on completion) |
| `SAVE R` | DMA | 4 | local R | global memory (local slot freed when done) |

Each output tile `C[i,j]` takes `Kt` COMPUTEs, `Kt-1` ACCs and one SAVE, where
`Kt = ceil(K/64)`. Fused multiply-accumulate is not modelled. It is discussed in
the report as a possible HW improvement.

### Minimum memory

Processing one output tile strictly in order needs 3 tiles when `Kt = 1` (A, B,
P) and 4 when `Kt > 1` (accumulator, A, B, new partial). Below that a workload
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
| `srcs` | local tiles read (LOAD: none; COMPUTE/ACC: 2; SAVE: 1) |
| `dst` | local tile written (required except for SAVE, which writes to global memory) |
| `deps` | ids of ops that must finish first; must include the producer of every src |

Rules checked by `Trace.validate()`:

- ids are unique and every dep exists; no dependency cycles;
- arity per type as above;
- each tile is written by exactly one op;
- every src is produced by some op, and that op is in `deps`;
- every loaded tile is used, and every COMPUTE/ACC result is used or SAVEd.

Tile naming used by the translator (any unique string is valid):

| Name | Meaning |
|---|---|
| `A_i_k`, `B_k_j` | input tiles |
| `P_i_j_k` | partial product `A_i_k x B_k_j` |
| `S_i_j_k` | running sum for `C[i,j]` after step k (`S_i_j_1 = P_i_j_0 + P_i_j_1`) |

### JSON

```json
{
 "format_version": 1,
 "workload": {"name": "mm_64x128x64", "M": 64, "K": 128, "N": 64, "tile": 64},
 "ops": [
  {"id": 0, "type": "LOAD", "srcs": [], "dst": "A_0_0", "deps": []},
  {"id": 4, "type": "COMPUTE", "srcs": ["A_0_0", "B_0_0"], "dst": "P_0_0_0", "deps": [0, 1]},
  {"id": 6, "type": "ACC", "srcs": ["P_0_0_0", "P_0_0_1"], "dst": "S_0_0_1", "deps": [4, 5]},
  {"id": 7, "type": "SAVE", "srcs": ["S_0_0_1"], "dst": null, "deps": [6]}
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
6,ACC,P_0_0_0;P_0_0_1,S_0_0_1,4;5
7,SAVE,S_0_0_1,,6
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
| A5 | ACC takes 2 same-size tiles and writes a new result tile. |
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
| 1 | Translator: tiling, padding, DAG, program order, trace writers | **this PR** |
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
  translator.py  # matmul -> trace (tiling, padding, deps, orders, ACC shape)
  workloads.py   # named presets: small, medium, large, skewed
scripts/
  gen_workload.py
examples/        # hand-written example trace and HW config
tests/
```
