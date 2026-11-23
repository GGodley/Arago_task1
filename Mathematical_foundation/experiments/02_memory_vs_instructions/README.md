# 02: Instruction counts vs local memory size

**Question.** For a fixed matmul, how do MAC and DMA instruction counts change
as local memory grows?

**Method.** Closed-form counts from `foundation/counts.py` for one square size
(default 4096³), with memory S from 4 to 8,192 tiles plus the no-reload point
S₀. DMA at S uses the ideal tiling, best of block and panel, with reloads
included. Run `python run.py` (options: `--size 4096`, `--mem "4;8;...;8192"`).

| Output | Shows |
|---|---|
| `outputs/instructions_vs_memory_4096.png` | MAC, DMA no-reuse ceiling, DMA perfect-reuse floor, DMA at each S |
| `outputs/instructions_vs_memory_4096.csv` | every point, including reuse factor 2mnk/L |
| `outputs/instructions_vs_memory_{64,256,512,1024,2048}.png` (+ CSV) | the same for the other sizes |
| `outputs/instructions_vs_memory_4096_overlap_nd16_nm16.png`, `_512_overlap_nd8_nm8.png` | adds the curve after reserving overlap overhead |

**Findings (4096³).**

- MAC instructions stay at 520,192 regardless of memory.
- DMA starts at the no-reuse ceiling (528,384 at S = 4, reuse 1×) and falls
  roughly as 1/√S: 184,320 at S = 16 (reuse 3×), 40,960 at S = 256 (reuse 14×).
- It reaches the perfect-reuse floor of 12,288 only at S₀ = 4,162 tiles, when a
  whole 64-row panel of A fits. That is a 43× range controlled by memory alone.

**Findings (64³ and 256³).**

- 64³ is a single tile: MAC 1, DMA 3 (A, B, C) at every memory size. There is
  nothing to reuse, so memory beyond the 3-slot minimum sits idle.
- 256³ drops from 144 DMA (no reuse) to its floor of 48 by S₀ = 22 tiles, a 3×
  swing. Small jobs saturate their reuse with very little memory.

Regenerate with `--size 64 --mem "3;4;8;16;32;64;128;256;512"` and
`--size 256 --mem "4;8;16;32;64;128;256;512"`.

## Overlap overhead (`--nd`, `--nm`)

The base curve assumes one thing happens at a time. To let LOADs, SAVEs and
ACCs overlap with compute, some slots must be kept free, so they are not
available for reuse. From the residency rule (a tile holds its slot from issue
until its last reader, or its SAVE, finishes):

| Overlap | Extra slots | Why |
|---|---|---|
| LOAD in flight | 1 per busy DMA unit | the destination slot is claimed at issue, 4 ticks before the data lands |
| SAVE overlapping the next block | 1 per busy DMA unit (same units) | the old sum keeps its slot until its SAVE finishes, while the next block opens its own sum |
| COMPUTE and ACC in parallel | 2 per matrix unit, minus the 1 already budgeted | a product waiting for its ACC, plus the ACC result allocated before its inputs are freed |

```
overhead O = Nd + 2*Nm - 1           (foundation.overlap_overhead)
tiling budget: (bm + 1)(bn + 1) <= S - O
```

Every DMA unit does a LOAD or a SAVE, never both, so one slot per DMA unit
covers both. With `--nd 16 --nm 16`, O = 47: 4096³ cannot run below 51 tiles,
needs 2.3× the ideal DMA at S = 64, and is within 10% of ideal from S = 256 up.
The no-reload point moves from S₀ to S₀ + O.

Regenerate with `--size 4096 --nd 16 --nm 16` and
`--size 512 --mem "4;8;...;1024" --nd 8 --nm 8`.
