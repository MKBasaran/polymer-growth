# Baseline Benchmarks (Pre-Refactor)

**Purpose:** Capture legacy code performance and behavior for thesis comparison.

**Hardware:**
- Machine: [To be filled - check About This Mac]
- CPU: [cores, model]
- RAM: [GB]
- OS: macOS 14.x (Darwin 24.5.0)
- Python: [check with `python --version`]

**Software Versions:**
```bash
python --version
pip list | grep -E "numpy|pandas|matplotlib|scipy"
```

---

## BENCHMARK 1: Core Simulation Speed

**Test:** Single simulation run with fixed parameters

**Configuration:**
```python
params = {
    'time_sim': 1000,
    'number_of_molecules': 100000,
    'monomer_pool': 31600000,
    'p_growth': 0.72,
    'p_death': 0.000084,
    'p_dead_react': 0.73,
    'l_exponent': 0.41,
    'd_exponent': 0.75,
    'l_naked': 0.24,
    'kill_spawns_new': 1
}
seed = 42
```

**Command:**
```bash
cd "/Users/kaanbasaran/Desktop/thesis_try/program code"
time python -c "
import numpy as np
from simulation import polymer
np.random.seed(42)
result = polymer(1000, 100000, 31600000, 0.72, 0.000084, 0.73, 0.41, 0.75, 0.24, 1)
print('Living:', len(result[0]))
print('Dead:', len(result[1]))
print('Coupled:', len(result[2]))
"
```

**Results (TO BE FILLED):**
```
Runtime: ___ seconds
Living chains: ___
Dead chains: ___
Coupled chains: ___

Memory (check Activity Monitor during run): ___ MB
```

**Output Hash (for reproducibility verification):**
```bash
# Save distribution to file
python -c "
import numpy as np
from simulation import polymer
np.random.seed(42)
result = polymer(1000, 100000, 31600000, 0.72, 0.000084, 0.73, 0.41, 0.75, 0.24, 1)
np.save('../baseline_evidence/sim_baseline_dist.npy', {
    'living': result[0],
    'dead': result[1],
    'coupled': result[2]
})
"

# Compute hash
shasum -a 256 ../baseline_evidence/sim_baseline_dist.npy
```

Hash: `_______________________________________________`

---

## BENCHMARK 2: FDDC Optimizer (Short Run)

**Test:** FDDC optimization for 10 generations (short run for baseline)

**Configuration:**
- Population: 50
- Generations: 10
- Target: `fakeData/sim_val0`
- Seed: 42

**Command:**
```bash
cd /Users/kaanbasaran/Desktop/thesis_try
python legacy_reproduce.py
# Modify script to run only 10 generations for baseline
```

**Results (TO BE FILLED):**
```
Total runtime: ___ minutes ___ seconds
Time per generation: ___ seconds (average)
Best cost after 10 gen: ___
Memory peak: ___ GB

Final best parameters:
  time_sim: ___
  number_of_molecules: ___
  ...
```

---

## BENCHMARK 3: GUI Responsiveness

**Test:** Manual interaction testing with legacy GUI

**Procedure:**
1. Launch GUI: `python "program code/user_interface.py"`
2. Select FDDC algorithm
3. Select Min max V2 cost function
4. Load fake data: `fakeData/sim_val0`
5. Set population=50, iterations=5
6. Click "Run"
7. Observe behavior

**Observations (TO BE FILLED):**

**During Computation:**
- Does GUI window respond to clicks? **YES / NO**
- Can you minimize/maximize window? **YES / NO**
- Does window show "(Not Responding)"? **YES / NO**
- Are plots updated in real-time? **YES / NO**
- Can you interact with other applications? **YES / NO**

**Time to First Result:**
- Time from click to first plot update: ___ seconds

**Errors/Crashes:**
- Any errors? **YES / NO**
- If yes, describe: ___

**Overall UX Rating:**
- Usability (1-5): ___
- Responsiveness (1-5): ___
- Visual clarity (1-5): ___

---

## BENCHMARK 4: Multiprocessing Overhead

**Test:** Population evaluation parallelization efficiency

**Method:**
- Run GA_base with Pool(1) vs Pool(6) on same problem
- Measure wall-clock time difference
- Compute parallelization efficiency

**Command:**
```bash
# Modify GA_base.py line 20 to use Pool(1)
# Run and time

# Restore Pool(6)
# Run and time

# Efficiency = (Time_1_worker / Time_6_workers) / 6
```

**Results (TO BE FILLED):**
```
Single worker (Pool(1)): ___ seconds
Six workers (Pool(6)): ___ seconds
Speedup: ___x
Efficiency: ___% (ideal=100%)

Analysis:
[Is there overhead? Is it scaling well?]
```

---

## BENCHMARK 5: Memory Usage

**Test:** Peak memory during FDDC run

**Method:**
- Run FDDC with Activity Monitor open
- Record peak memory usage

**Results (TO BE FILLED):**
```
Peak memory (RSS): ___ MB
Peak memory (Virtual): ___ MB

Python process: ___ MB
Child processes (if any): ___ MB each
```

---

## BENCHMARK 6: Determinism Check

**Test:** Same seed → same output?

**Method:**
```bash
# Run 1
python -c "import numpy as np; from program_code.simulation import polymer; np.random.seed(42); r1 = polymer(100, 10000, 1000000, 0.72, 0.000084, 0.73, 0.41, 0.75, 0.24, 1); import pickle; pickle.dump(r1, open('run1.pkl', 'wb'))"

# Run 2
python -c "import numpy as np; from program_code.simulation import polymer; np.random.seed(42); r2 = polymer(100, 10000, 1000000, 0.72, 0.000084, 0.73, 0.41, 0.75, 0.24, 1); import pickle; pickle.dump(r2, open('run2.pkl', 'wb'))"

# Compare
python -c "import pickle, numpy as np; r1 = pickle.load(open('run1.pkl', 'rb')); r2 = pickle.load(open('run2.pkl', 'rb')); print('Match:', np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]) and np.array_equal(r1[2], r2[2]))"
```

**Results (TO BE FILLED):**
```
Run 1 vs Run 2: **MATCH / DIFFERENT**

If different, diagnosis:
[Why? Global state? Multiprocessing? Random order?]
```

---

## SUMMARY TABLE (After Baseline Capture)

| Metric | Baseline | Target (Post-Refactor) | Achieved |
|--------|----------|------------------------|----------|
| Simulation (1000 steps, 100k mol) | ___ s | <___ s (5x faster) | ___ s |
| FDDC 10 gen | ___ min | <___ min (3x faster) | ___ min |
| GUI freeze during compute | YES/NO | NO | YES/NO |
| GUI response time | ___ s | <1s | ___ s |
| Memory peak | ___ MB | <___ MB | ___ MB |
| Determinism (same seed) | YES/NO | YES | YES/NO |
| Test coverage | 0% | >80% | ___% |

---

## NOTES

**Observations:**
- [Any surprising findings?]
- [Performance bottlenecks identified visually?]
- [GUI issues observed?]

**Blockers:**
- [Anything that prevented full baseline capture?]

---

**Date Captured:** 2026-01-05 (to be updated after actual runs)

**Captured By:** [Your name]

**Next Steps:** Proceed to Phase 1 refactor, re-run these benchmarks after each phase

---

*These numbers will be reported in thesis Chapter X and accompanying paper*