# Mathematical foundation

A closed-form model of the blocked matmul accelerator, built from first
principles: operation counts, then how local memory sets traffic and
concurrency, then time, then the hardware optimization. The event-driven
simulator in `matmul_sim/` is used only to validate it.

The full derivation lives in [docs/equations.md](docs/equations.md).

## Layout

```
Mathematical_foundation/
├── README.md                 this file
├── docs/
│   └── equations.md          the model's equations and what each variable means
├── foundation/               reusable closed-form code (no plotting, no I/O)
│   ├── counts.py             instruction counts and LOADs under limited memory
│   └── plotting.py           shared figure style for every experiment
├── experiments/              one numbered folder per experiment
│   ├── 01_instruction_counts/
│   │   ├── README.md         question, method, findings
│   │   ├── run.py            regenerates everything in outputs/
│   │   └── outputs/          graphs and CSV (committed so they can be viewed on GitHub)
│   └── 02_memory_vs_instructions/   same structure
└── tests/                    checks the formulas against the simulator's own trace
```

## Running

```bash
pip install -e ".[test]"          # from the repo root: numpy, matplotlib, pytest
python Mathematical_foundation/experiments/01_instruction_counts/run.py
python Mathematical_foundation/experiments/02_memory_vs_instructions/run.py
python -m pytest Mathematical_foundation/tests
```

## Adding to it

- **A new formula** (timing, utilization, the optimizer) goes in a new module
  under `foundation/`, exported from `foundation/__init__.py`, with a test in
  `tests/`. Keep modules free of plotting so experiments can share them.
- **A new experiment** gets the next number, `experiments/02_<name>/`, with its
  own `run.py`, `README.md` and `outputs/`. Import formulas from `foundation`
  and the style from `foundation.plotting`.
- **Validation against the simulator** goes in `tests/`, comparing a formula
  with what `matmul_sim` produces.

## Experiments

| # | Question | Status |
|---|---|---|
| 01 | How do MAC and DMA instruction counts grow with matrix size? | done |
| 02 | How do instruction counts change with local memory size (4096³)? | done |
| 03 | [Timing vs units](experiments/03_timing_vs_units/): heatmaps of time over DMA and matrix unit counts | done |
| 04 | Optimizer: cheapest (Nd, Nm, S) for a workload and time target | planned |
