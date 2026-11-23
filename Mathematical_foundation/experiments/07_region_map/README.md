# 07 Region map

The optimizer's map of where a workload can run, speed against memory, drawn from
`foundation/optimizer.py` (the same code the dashboard ports to JavaScript).

```
python Mathematical_foundation/experiments/07_region_map/run.py
python Mathematical_foundation/experiments/07_region_map/run.py --dims "64,4096,4096"
```

| Edge | Where it comes from | Derived or chosen |
|---|---|---|
| Bottom | P = 1, one matrix unit | derived |
| Top | P_max = W / ceil(W / ((1 + acc) m n)), the work cap with every output open | derived |
| Left (solid) | least Little's-law memory over tilings with enough independent work for Nm | derived |
| Right (dashed) | memory of the tiling with the fewest DMA units | derived, but jumps where Nd is 1 to 3 |
| Midline (dotted) | sqrt(left × right), splits Tiny from Compact | chosen |
| Fast, Extremely fast | 1/3 and 2/3 of the way up the log speed range | chosen |

Near the top both edges meet at the whole job's tiles (mk + kn + mn): at the
fastest speeds everything has to be resident. Below the Fast band, the right edge
jumps because a single DMA unit more or less changes which tiling is cheapest;
that band is not offered.
