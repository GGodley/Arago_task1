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
| `outputs/instructions_vs_memory_64.png`, `_256.png` (+ CSV) | the same for 64³ and 256³ |

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
