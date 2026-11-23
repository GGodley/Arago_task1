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
| `outputs/instructions_vs_memory_{64,128,256,512,1024,2048,4096}_overlap_nd8_nm8.png` (+ `.csv`) | same chart for every size, plus the curve after reserving overlap overhead (Nd = 8, Nm = 8, O = 23) |

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
covers both. The no-reload point moves from S₀ to S₀ + O.

All seven sizes use one setting, `--nd 8 --nm 8` (O = 23), so they compare
directly. Any job needs at least 26 tiles with this overlap.

| Size | S₀ | S₀ + O | DMA at S = 32, ideal / overlap | DMA at S = 64 | DMA at S = 256 |
|---|---|---|---|---|---|
| 64³ | 3 | 26 | 3 / 3 | 3 / 3 | 3 / 3 |
| 128³ | 8 | 31 | 12 / 12 | 12 / 12 | 12 / 12 |
| 256³ | 22 | 45 | 48 / 80 | 48 / 48 | 48 / 48 |
| 512³ | 74 | 97 | 320 / 576 | 256 / 256 | 192 / 192 |
| 1024³ | 274 | 297 | 2,304 / 4,352 | 1,536 / 2,048 | 1,024 / 1,024 |
| 2048³ | 1,058 | 1,081 | 16,384 / 33,792 | 11,264 / 14,336 | 6,144 / 6,144 |
| 4096³ | 4,162 | 4,185 | 122,880 / 266,240 | 81,920 / 110,592 | 40,960 / 45,056 |

The overhead is a fixed number of slots, so it hurts most when S is small and
fades as S grows: about 2× the DMA at S = 32, and within 10% by S = 256 for
every size.

Regenerate (memory lists as used):

```
python run.py --size 64   --mem "3;4;8;16;32;64;128;256;512"       --nd 8 --nm 8
python run.py --size 128  --mem "4;8;16;32;64;128;256;512"         --nd 8 --nm 8
python run.py --size 256  --mem "4;8;16;32;64;128;256;512"         --nd 8 --nm 8
python run.py --size 512  --mem "4;8;16;32;64;128;256;512;1024"    --nd 8 --nm 8
python run.py --size 1024 --mem "4;8;16;32;64;128;256;512;1024;2048" --nd 8 --nm 8
python run.py --size 2048 --mem "4;8;16;32;64;128;256;512;1024;2048;4096;8192" --nd 8 --nm 8
python run.py --size 4096 --mem "4;8;16;32;64;128;256;512;1024;2048;4096;8192" --nd 8 --nm 8
```
