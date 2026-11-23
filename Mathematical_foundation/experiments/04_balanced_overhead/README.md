# 04 Balanced overhead

Experiment 02 reserved a fixed overlap overhead (Nd = Nm = 8, O = 23 slots) at
every memory size. Experiment 03 showed that the useful number of matrix units
depends on Nd and on memory. This experiment feeds that balance back into the
memory chart: at each S, for a chosen Nd, the matrix units are set to Nm*, the
fewest at which matrix time no longer exceeds DMA time.

```
O(Nm)  = Nd_busy + 2 Nm - 1            Nd_busy = min(Nd, transfers in the job)
L      = loads_with_memory(S - O)      tiling budget after the overhead
T_dma  = (L + SAVE) Ld / Nd
Nm*    = first Nm with ceil(W / Nm) <= T_dma,   Nm <= W
```

The loop matters because the overhead feeds back on itself: more matrix units
take more slots, which leaves less memory for reuse, which means more loads and
a longer DMA time, so fewer matrix units are useful.

## Outputs

| File | What |
|---|---|
| `outputs/balanced_overhead_{64,...,4096}.png` | top: DMA instructions vs S (ideal, fixed 8/8, balanced at Nd = 4 and 16); bottom: overhead O vs S, labelled with Nm* and O as a share of S |
| `outputs/balanced_overhead_{...}.csv` | every point: Nm*, O, DMA instructions, and time max(T_mat, T_dma) per Nd |

## Findings

| 4096³ | S = 32 | S = 256 | S = 1,024 | S = 8,192 |
|---|---|---|---|---|
| Nd = 4: Nm*, O | 4, 11 (34% of S) | 12, 27 (11%) | 22, 47 (5%) | 43, 89 (1%) |
| Nd = 16: Nm*, O | 6, 27 (84%) | 40, 95 (37%) | 85, 185 (18%) | 170, 355 (4%) |
| DMA vs ideal, Nd = 16 | 4.3× | 1.3× | 1.0× | 1.0× |

- **The overhead grows with memory, but its share of memory shrinks.** More
  memory means more reuse, so DMA can feed more matrix units, and each one
  takes 2 slots. Nm* grows about like √S, so O/S falls about like 1/√S.
- **The cost is all at small memory.** With Nd = 16 and S = 32, the overhead
  takes 84% of memory, reuse collapses and DMA goes back to the no-reuse
  ceiling. From about S = 1,024 up the overhead costs nothing measurable.
- **The fixed 8/8 was too much for small memory and too little for large.**
  With Nd = 4 the balanced overhead is below 23 up to S = 128 and above it
  after.
- **Small jobs:** 64³ needs 1 matrix unit and at most 3 DMA units, so its
  overhead is 4 slots whatever the hardware. 512³ stops growing at S = 128
  (Nm* = 5 with Nd = 4, 20 with Nd = 16) because it has no more reuse to gain.

Time is not the full model here (no dependency path or Little's-law cap); use
experiment 03 for time.

Regenerate:

```
python run.py                                  # all sizes, Nd = 4 and 16
python run.py --sizes "4096" --nd "8;32"
```
