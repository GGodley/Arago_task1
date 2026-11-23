# 01: Instruction counts vs matrix size

**Question.** How many MAC (matrix-unit) and DMA instructions does a square
matmul need, and how do they grow with size?

**Method.** Closed-form counts from `foundation/counts.py` for
N = 64, 128, ..., 4096 (M = K = N). The DMA graph also shows the ideal-tiling
total when local memory holds only 64 or 256 tiles. Run `python run.py`
(options: `--sizes "64;128;..."`, `--mem "64;256"`).

| Graph | Shows |
|---|---|
| `outputs/mac_instructions.png` | COMPUTE, ACC and their total |
| `outputs/dma_instructions.png` | LOAD, SAVE, their total, and the total with limited memory |
| `outputs/mac_vs_dma_line.png` | MAC total vs DMA total, line chart |
| `outputs/mac_vs_dma_bar.png` | MAC total vs DMA total, grouped bars |
| `outputs/op_counts.csv` | every number behind the graphs |

**Findings.**

- MAC instructions grow as N³ (8× per doubling), from 1 at 64³ to 520,192 at
  4096³. They do not depend on the hardware.
- DMA instructions grow as N² (4× per doubling) when every tile is moved once,
  reaching 12,288 at 4096³. With limited memory, reloads make them grow faster:
  4096³ needs 81,920 at S = 64 tiles, 6.7× the compulsory count.
- The counts are equal at 128³ (12 each). Beyond that MAC leads, by 42× at 4096³.
  A DMA instruction takes 4× as long as a MAC instruction (512 vs 128 cycles),
  so in time the crossover comes later, around 512³.
