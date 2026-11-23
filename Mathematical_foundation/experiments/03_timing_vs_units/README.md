# 03 Timing vs units

Predicted time T over a grid of DMA units (Nd) and matrix units (Nm), one panel
per matrix size, at a fixed local memory S. Uses `foundation.predict_time`, the
closed-form model validated against the simulator (median error 2.5%).

**How to read it.** Light cells are fast. White contours join setups with equal
time. Thick black lines split the map by which term sets T:

- **matrix-bound**: adding DMA units does nothing; contours run horizontally.
- **DMA-bound**: adding matrix units does nothing; contours run vertically.
- **path-bound**: the dependency chain (load, compute, sum tree, save) sets T; no unit count helps.

The boundary between the matrix and DMA regions is the balance law Nd/Nm ≈ 4/b
(b = tile block side that fits in S). The star is the cheapest setup (smallest
Nd + Nm) within 5% of the fastest time on the grid.

## Hardware values

| Parameter | Values | Why |
|---|---|---|
| Nd, Nm | 1 to 32 each | covers our simulator sweep (1 to 16) and goes far enough to see the regions flatten |
| S | 64, 256, 1,024 tiles | tight, middle, and roomy; 4096³ only stops reloading at 4,162 |
| Sizes | 64³, 512³, 4096³ | small, the crossover, large |

## Outputs

| File | What |
|---|---|
| `outputs/time_vs_units_S{64,256,1024}.png` | three heatmaps, one per size, at that memory |
| `outputs/time_vs_units_S{...}.csv` | every cell: T and its three terms, limiter, loads, tiling |

## Findings

| | star at S = 64 | S = 256 | S = 1,024 | Nd = Nm = 8 |
|---|---|---|---|---|
| 512³ | Nd 30, Nm 29: 43 ticks | Nd 20, Nm 31: 39 | Nd 20, Nm 31: 39 | 128 ticks |
| 4096³ | Nd 25, Nm 29: 17,946 | Nd 10, Nm 31: 16,789 | Nd 6, Nm 31: 16,789 | 65,032 ticks |

- **64³ is path-bound everywhere** (9 ticks) except Nd = 1. One DMA unit more
  than one, one matrix unit, and the job is done; extra hardware is wasted.
- **Large jobs: memory buys DMA units.** For 4096³ the same speed needs 25 DMA
  units at S = 64 but only 6 at S = 1,024, because more memory means more reuse
  and fewer loads. The balance line gets steeper as S grows.
- **Matrix units set the floor for large jobs.** Once DMA keeps up, T ≈ W / Nm.

Not yet included: the 2·Nm matrix-side overlap slots (experiment 02). The DMA
side is already dynamic here: F = S − R − 1 free slots cap useful DMA parallelism
at F·Ld/τ.

Regenerate:

```
python run.py --mem 64
python run.py --mem 256
python run.py --mem 1024
python run.py --sizes "128;1024;2048" --mem 256 --max-units 16
```
