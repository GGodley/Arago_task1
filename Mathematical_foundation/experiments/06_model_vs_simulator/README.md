# 06 Model vs simulator

Checks the model's three Task 1 outputs (total cycles, matrix-unit usage over time,
local memory used over time) against the cycle-level simulator on the same trace and
hardware. The simulator uses the memory-aware blocked scheduler.

```
python Mathematical_foundation/experiments/06_model_vs_simulator/run.py
```

## Examples

| example | hardware (Nd, Nm, S) | cycles model / sim | matrix usage model / sim | DMA usage model / sim | memory mean: needed / with prefetch / sim |
|---|---|---|---|---|---|
| 128³ (small) | 2, 2, 16 | 3,072 / 3,328 (−8%) | 25% / 23% | 100% / 92% | 41% / 44% / 34% |
| 512³ (medium) | 8, 4, 128 | 31,744 / 32,000 (−1%) | 97% / 96% | 39% / 38% | 55% / 82% / 66% |
| 2048³ (large) | 16, 32, 512 | 259,072 / 285,824 (−9%) | 100% / 90% | 63% / 94% | 78% / 99% / 93% |
| 64×4096×4096 (LLM decode) | 16, 8, 256 | 135,168 / 136,576 (−1%) | 96% / 95% | 100% / 99% | 37% / 37% / 42% |

`cycles_validity.png` repeats the cycles check over the 1,787 runs of the earlier sweep
(`/mnt/project-files/timing_model/compare.csv`): median error 3.1%; 0.4% with S of
128 to 512 tiles, 12.5% with S of 4 to 32 tiles.

## What it shows

- **Cycles** match within 10% on all four examples. The model is a quasi-optimal
  schedule, so it is a little faster than the simulator; the gap grows when memory is
  tiny, where the simulator's scheduler gets about half the reuse the model assumes.
- **Matrix usage** matches in level and shape: a ramp after the first load (Ld), a flat
  steady state, and a drop at the end.
- **DMA usage** matches when the job is DMA-bound or the scheduler reuses as well as
  the model's tiling. At 2048³ the simulator's scheduler uses smaller blocks than the
  model's 11×32, reloads more, and its DMA runs at 94% against the model's 63%. That is
  the scheduler falling short of quasi-optimal, not the equations.
- **Memory** depends on scheduling policy, so the model gives two curves. *Needed* is
  the least the schedule needs (resident sums plus Little's-law in-flight tiles); the
  optimizer sizes S against it. *With prefetch* adds the loads a scheduler runs ahead
  with when the DMA has spare time. The simulator falls between the two in each example.

## Files

- `outputs/timeline_<example>.png`: three stacked panels, simulator solid, model dashed
- `outputs/summary.csv`: the table above
- `outputs/cycles_validity.png`: model vs simulator cycles over the sweep
