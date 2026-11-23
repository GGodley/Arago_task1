# 05 Independent-work cap and a time target

## The cap (`foundation/parallelism.py`)

A matrix unit can only run a product whose A and B tiles are in memory, and
all products of output tile (i, j) go into that tile's one running sum. A
running sum is a serial chain: one ACC per tick, each consuming one product.
So each open output tile keeps at most 1 + acc matrix units busy (one COMPUTE
feeding one ACC per tick), and

    Nm_eff <= (1 + acc) * open        block bm x bn: open = bm bn;  panel p: open = p

Units above the cap sit idle but still hold overhead slots.

## The memory check: Little's law (default)

A tiling fits in S when the memory it uses at steady state is at most S:

    M = R + (L tau + COMPUTE 2 Lm + SAVE Ld) / (T - 2 Ld)       tau = Ld + 2 Lm = 6

R is the resident set (the bm·bn running sums for a block; the whole panel plus
its sums for a panel). The rest is Little's law: each tile in flight holds its
slot for a known time, so slots = rate × hold time. This replaces the fixed
overhead Nd + 2 Nm − 1, which reserved slots as if every unit were busy all the
time. The old results are kept as `outputs/*_fixed.*` (`--overhead fixed`).

## Part A: fastest setup, with and without the cap (4096³)

Units are free. For each tiling, M <= S gives T >= 2 Ld + X / (S − R), with X the
in-flight slot-ticks; the cap adds T >= 2 Ld + W / cap.

| S | T, cap | in flight (% of S) | Nd, Nm | tiling |
|---|---|---|---|---|
| 128 | 11,351 | 81% | 39, 46 | block 5x5 |
| 1,024 | 988 | 74% | 151, 531 | block 16x17 |
| 8,192 | 115 | 70% | 613, 4,864 | block 38x64 |

Run flat out, about 70 to 80% of memory holds tiles in flight. That is Little's
law: a faster job has more tiles in flight at once. The fastest times are lower
than with the fixed overhead (15,772 → 11,351 ticks at S = 128), because the
fixed formula reserved slots for units that were not busy.

## Part B: cheapest setup under 15,000 ticks (4096³)

The target fixes Nm = ceil(W / (15,000 − 8)) = 35. Each tiling then needs
Nd = ceil((L + SAVE) · 4 / 14,992) DMA units; the cheapest is the tiling with
the fewest that fits (M <= S, with T − 2 Ld = 14,992) and has cap >= 35.

| S | Nd | units | M = R + in flight | tiling | fixed overhead (Nd, tiling) |
|---|---|---|---|---|---|
| < 104 | infeasible | | | | |
| 104 | 30 | 65 | 25 + 78.7 | block 5x5 | infeasible |
| 128 | 19 | 54 | 64 + 62.3 | block 8x8 | infeasible |
| 256 | 13 | 48 | 169 + 52.5 | block 13x13 | 14, block 11x13 |
| 1,024 | 7 | 42 | 704 + 44.3 | block 22x32 | 7 |
| 8,192 | 4 | 39 | 4,096 + 39.3 | block 64x64 | 4 |

- **104 tiles is the least memory that can meet the target** (132 with the
  fixed overhead). The smallest block with enough independent work for 35 matrix
  units is 4x5 or 5x5; its in-flight tiles take about 79 slots.
- **Memory buys DMA units, not matrix units.** Nm stays at 35; Nd falls from 30
  to 4 as reuse improves.
- **The in-flight floor is about 39 slots**, set by the target: 35 for compute
  (2 slots for 1 tick at 35 per tick) plus about 4 for loads and saves at best.
- **Choosing S** needs a price for a DMA unit relative to a memory tile:
  104 → 256 saves 17 units for 152 tiles (9 tiles per unit); 256 → 1,024 saves 6
  for 768 (128 tiles per unit).

Regenerate:

```
python run.py                              # 4096³, target 15,000
python run.py --size 2048 --target 5000
python run.py --overhead fixed           # the old Nd + 2 Nm - 1 check
```
