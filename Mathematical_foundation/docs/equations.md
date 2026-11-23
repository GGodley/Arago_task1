# Model equations

All quantities are in tiles (one 64x64 block) or ticks (128 cycles).

## Variables

| Symbol | Meaning |
|---|---|
| M, K, N | matrix sizes: A is MxK, B is KxN |
| m, k, n | sizes in tiles: ceil(M/64), ceil(K/64), ceil(N/64) |
| acc | 1 if k > 1 (partials must be added), else 0 |
| Nd, Nm | number of DMA units, matrix units |
| S | local memory size, in tiles |
| Ld, Lm | DMA transfer time (4 ticks), matrix op time (1 tick) |
| W | matrix commands, COMPUTE + ACC |
| Sv | SAVE commands |
| L0, L(S) | compulsory LOADs; LOADs needed with S tiles (reloads included) |
| R | tiles a tiling keeps resident (running sums or a resident panel) |
| F | slots free for tiles in flight, S - R - 1 |
| tau | slot-time one streamed tile costs, Ld + Lm(1 + acc) |

## Step 1: work (implemented in `foundation/counts.py`)

    COMPUTE = m n k        ACC = m n (k - 1)        W = m n (2k - 1)
    SAVE    = m n          LOAD >= L0 = k (m + n)

## Step 2: memory sets traffic (implemented in `foundation/counts.py`)

    block bm x bn:  L = k (m ceil(n/bn) + n ceil(m/bm)),  needs (bm + 1)(bn + 1) <= S
                    best square side b = floor(sqrt(S)) - 1,  L ~ 2 m n k / sqrt(S)
    A panel of p:   L = m k + n k ceil(m/p),  p = floor((S - 2) / (k + acc))
    L(S) = min over block, A panel, B panel
    no reloads once S >= S0 = min(m, n) (k + acc) + 2

## Step 2b: overlap overhead (implemented in `foundation/counts.py`)

    O = Nd + 2 Nm - 1       slots kept free so LOAD/SAVE (1 per DMA unit) and
                            COMPUTE/ACC (2 per matrix unit) can overlap
    tiling budget:  (bm + 1)(bn + 1) <= S - O,   no reloads once S >= S0 + O

## Step 3: memory sets concurrency (implemented in `foundation/timing.py`)

    Little's law:  DMA rate <= min(Nd / Ld, F / tau)

## Step 4: time (implemented in `foundation/timing.py`)

    T_mat  = 2 Ld + ceil(W / Nm) Lm
    T_dma  = (L + Sv) max(Ld / Nd, tau / F)
    T_path = 2 Ld + Lm (1 + ceil(log2 k))
    T      = min over tilings of max(T_mat, T_dma, T_path),   cycles = 128 T

## Step 5: outputs (planned)

    U_m = W Lm / (Nm T)        U_d = (L + Sv) Ld / (Nd T)

## Step 7: optimization (planned)

    For a target time T*:  Nm* = ceil(W Lm / (T* - 2 Ld)),  Nd* = ceil((L(S) + Sv) Ld / T*)
    feasible if (L(S) + Sv) tau / F(S) <= T*
    S* = (wd Ld m n k / (ws T*))^(2/3), clipped to [flow minimum, S0]
