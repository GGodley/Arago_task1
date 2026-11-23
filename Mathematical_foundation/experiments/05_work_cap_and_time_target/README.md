# 05 Independent-work cap and a time target

## The cap (`foundation/parallelism.py`)

A matrix unit can only run a product whose A and B tiles are in memory, and
all products of output tile (i, j) go into that tile's one running sum. A
running sum is a serial chain: one ACC per tick, each consuming one product.
So each open output tile keeps at most 1 + acc matrix units busy (one COMPUTE
feeding one ACC per tick), and

    Nm_eff <= (1 + acc) * open        block bm x bn: open = bm bn;  panel p: open = p

Units above the cap sit idle but still hold overhead slots.

## Part A: fastest setup, with and without the cap (4096³)

| S | T, no cap | T, cap | overhead, no cap | overhead, cap | Nd, Nm (cap) | tiling (cap) |
|---|---|---|---|---|---|---|
| 128 | 15,308 | 15,772 | 81% | 76% | 32, 33 | block 4x5 |
| 1,024 | 1,407 | 1,497 | 92% | 80% | 121, 350 | block 11x16 |
| 8,192 | 159 | 179 | 93% | 80% | 480, 3,043 | block 24x64 |

The cap is real but weak: the search answers it by switching to wider blocks
that keep more outputs open, at a cost of 3 to 12% in time. About 80% of memory
still goes to overlap when units are free, because each matrix unit needs 2
buffer slots and the work-cap only asks for 1 open output per 2 units.

## Part B: cheapest setup under 15,000 ticks (4096³)

The target fixes Nm = ceil(W / (15,000 − 8)) = 35. Each tiling then needs
Nd = ceil((L + SAVE) · 4 / 14,992) DMA units; the cheapest is the tiling with
the fewest that fits in S with its overhead and has cap >= 35.

| S | feasible | Nd | Nm | units | overhead (% of S) | tiling | T |
|---|---|---|---|---|---|---|---|
| < 132 | no | | | | | | |
| 132 | yes | 33 | 35 | 68 | 102 (77%) | block 4x5 | 14,903 |
| 256 | yes | 14 | 35 | 49 | 83 (32%) | block 11x13 | 14,871 |
| 1,024 | yes | 7 | 35 | 42 | 76 (7%) | block 22x32 | 14,871 |
| 8,192 | yes | 4 | 35 | 39 | 73 (1%) | A panel 64 | 14,871 |

- **132 tiles is the least memory that can meet the target.** There, reuse is
  so low that 33 DMA units are needed, and the overhead is 77% of memory again.
- **Memory buys DMA units, not matrix units.** Nm stays at 35 (set by the
  work); Nd falls from 33 to 4 as reuse improves.
- **With a target, the overhead is small:** about 75 slots, under 8% of memory
  from 1,024 tiles up. The 80 to 93% only happens when chasing the fastest time.

Not yet modelled: a cost per unit and per tile of memory (here units count
equally and memory is free), and the Little's-law slot time for loads.

Regenerate:

```
python run.py                              # 4096³, target 15,000
python run.py --size 2048 --target 5000
```
