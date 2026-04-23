# Thomas Replication Experiments

Run one per day from this directory. Results save to `validation_results/`.

## Schedule

| Day | Script | Dataset | Gens | Est. Time | What it proves |
|-----|--------|---------|------|-----------|----------------|
| 1 | `a_5k_fddc_42gen.py` | 5k no BB | 42 | ~2-3 hrs | Our FDDC = Thomas's FDDC (both impls) |
| 2 | `b_5k_fddc_1801gen.py` | 5k no BB | 1801 | ~24 hrs | Our FDDC at his roulette gen count |
| 3 | `c_10k_fddc_56gen.py` | 10k no BB | 56 | ~3-4 hrs | Our FDDC = Thomas's FDDC (both impls) |
| 4 | `d_10k_fddc_1217gen.py` | 10k no BB | 1217 | ~24 hrs | Our FDDC at his roulette gen count |
| 5 | `e_20k_fddc_44gen.py` | 20k no BB | 44 | ~3-4 hrs | Our FDDC = Thomas's FDDC (both impls) |
| 6 | `f_20k_fddc_1192gen.py` | 20k no BB | 1192 | ~24 hrs | Our FDDC at his roulette gen count |
| 7 | `g_30k_fddc_27gen.py` | 30k no BB | 27 | ~2-3 hrs | Our FDDC = Thomas's FDDC (both impls) |
| 8 | `h_30k_fddc_1159gen.py` | 30k no BB | 1159 | ~24 hrs | Our FDDC at his roulette gen count |

## How to run

```bash
cd scripts/experiments
/usr/local/bin/python3 a_5k_fddc_42gen.py
```

## What each does

**Short runs (A, C, E, G):** Run FDDC with BOTH our implementation and Thomas's
original simulation code at Thomas's exact FDDC generation count. Proves
the refactored code produces equivalent costs. Also runs simulation
equivalence test first.

**Long runs (B, D, F, H):** Run our FDDC for Thomas's roulette generation
count (1000-1800 gens). Only uses our implementation. Shows what FDDC
achieves given the same compute budget his basic GA had.

## Config (all experiments)

- Population: 100
- Memory: 10, Encounters: 10, Children: 2
- Mutation: 0.6 slight / 0.4 random
- Rank power: 1.5, Sigma points: 4
- Workers: auto (all cores)
- Seed: 42