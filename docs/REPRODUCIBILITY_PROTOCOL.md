# Reproducibility Protocol

**Purpose:** Define how to achieve bit-exact reproducible results.

**Thesis Requirement:** Reviewers must be able to reproduce any result in the paper.

---

## 1. SEED HANDLING

### 1.1 Core Principle

**Every source of randomness must be controlled by a seed.**

Sources in this codebase:
1. `numpy.random` (simulation growth/death decisions, vampire attacks)
2. `random` module (GA selection, mutation, crossover)
3. Multiprocessing workers (inherit or spawn seeds)

### 1.2 Implementation Strategy

**Pre-Refactor (Legacy):**
```python
# PROBLEM: Global RNG, no control
np.random.seed(2)  # Commented out in simulation.py!
r = np.random.random(living.shape[0])  # Uses global state
```

**Post-Refactor (v1.0):**
```python
# SOLUTION: Explicit RNG passing
rng = np.random.default_rng(seed=42)
r = rng.random(living.shape[0])  # Uses local RNG
```

**For Python `random` module:**
```python
import random
random.seed(42)  # Set global state explicitly
# OR use random.Random(42) for local instance
```

### 1.3 Multiprocessing Seeds

**Problem:**
Child processes inherit parent RNG state → potential correlation

**Solution:**
Use `SeedSequence` to spawn independent seeds:

```python
import numpy as np
from multiprocessing import Pool

def worker(args):
    idx, params, child_seed = args
    rng = np.random.default_rng(child_seed)
    # Use rng for all random operations
    return simulate(params, rng)

# Main process
main_seed = 42
ss = np.random.SeedSequence(main_seed)
child_seeds = ss.spawn(population_size)

with Pool(6) as pool:
    tasks = [(i, pop[i], child_seeds[i]) for i in range(population_size)]
    results = pool.map(worker, tasks)
```

**Key:** Each worker gets independent, reproducible RNG.

---

## 2. DETERMINISM GUARANTEES

### 2.1 What We Guarantee

**For a given seed:**
- ✅ Same simulation output (living, dead, coupled distributions)
- ✅ Same GA trajectory (population, fitness, best individual)
- ✅ Same FDDC convergence path
- ✅ Bit-exact floating point results (on same architecture)

**Platform notes:**
- Results are deterministic on same CPU architecture (x86, ARM)
- May differ slightly between architectures due to floating point rounding
- Use `atol=1e-10` for cross-platform comparisons

### 2.2 What We Don't Guarantee

**Cross-platform:**
- Exact floating point values may differ between x86 and ARM
- Matplotlib rendering may differ (visual only, data identical)

**Version changes:**
- NumPy version changes may affect random number generation
- Document NumPy version in any published result

### 2.3 Verification Test

**Automated test:**
```python
def test_determinism():
    """Same seed produces identical results."""
    seed = 42
    params = SimulationParams(...)

    # Run 1
    rng1 = np.random.default_rng(seed)
    dist1 = simulate(params, rng1)

    # Run 2
    rng2 = np.random.default_rng(seed)
    dist2 = simulate(params, rng2)

    # Verify exact match
    assert np.array_equal(dist1.living, dist2.living)
    assert np.array_equal(dist1.dead, dist2.dead)
    assert np.array_equal(dist1.coupled, dist2.coupled)
```

---

## 3. REPRODUCIBLE EXPERIMENTS

### 3.1 Experiment Configuration File

**Every experiment must have a YAML config:**

```yaml
# experiments/thesis_fddc_5k.yml
experiment:
  name: "FDDC on 5K synthetic data"
  date: "2026-01-10"
  seed: 42  # CRITICAL: Must be specified

data:
  target_file: "data/synthetic/sim_val0.csv"

optimizer:
  type: "fddc"
  population_size: 50
  max_generations: 200
  convergence_threshold: 10.0

parameters:
  bounds:
    time_sim: [100, 3000]
    number_of_molecules: [10000, 120000]
    # ... etc

execution:
  n_workers: 6

output:
  directory: "results/thesis_fddc_5k"
  save_trajectory: true
  save_distributions: true
```

**To reproduce:**
```bash
polymer-sim reproduce experiments/thesis_fddc_5k.yml
```

**Output includes:**
- `config.yml` (copy of input, for archival)
- `result.json` (best params, cost, convergence info)
- `trajectory.csv` (generation, best_cost, mean_cost)
- `best_distribution.csv` (final simulated distribution)
- `metadata.json` (software versions, runtime, hardware)

### 3.2 Metadata Logging

**Every run automatically logs:**
```json
{
  "software": {
    "polymer_growth_version": "1.0.0",
    "python_version": "3.10.8",
    "numpy_version": "1.24.2",
    "platform": "macOS-14.0-arm64"
  },
  "hardware": {
    "cpu": "Apple M1",
    "cores": 8,
    "ram_gb": 16
  },
  "execution": {
    "seed": 42,
    "start_time": "2026-01-10T14:30:00",
    "end_time": "2026-01-10T15:45:00",
    "runtime_seconds": 4500
  }
}
```

---

## 4. RESULT VERIFICATION

### 4.1 Hash-Based Verification

**For critical results (paper figures):**

```bash
# Generate result
polymer-sim fit --config paper_fig1.yml --seed 42 --out results/fig1/

# Compute hash of output distribution
python -c "
import numpy as np
import hashlib
dist = np.load('results/fig1/best_distribution.npy')
hash_val = hashlib.sha256(dist.tobytes()).hexdigest()
print(f'SHA256: {hash_val}')
" > results/fig1/verification_hash.txt

# Publish hash with paper
```

**Reviewers can:**
1. Clone repository
2. Run same command with same seed
3. Compute hash
4. Verify match

### 4.2 Regression Tests

**Store reference outputs:**
```
tests/fixtures/
├── reference_simulation_seed42.npy
├── reference_fddc_10gen_seed42.json
└── reference_distributions/
    ├── 5k_fit_result.csv
    └── 10k_fit_result.csv
```

**Test compares new code against references:**
```python
def test_regression_simulation():
    """Verify simulation output matches reference."""
    reference = np.load('tests/fixtures/reference_simulation_seed42.npy')

    params = SimulationParams(...)  # Same as reference
    rng = np.random.default_rng(42)
    result = simulate(params, rng)

    assert np.allclose(result.living, reference['living'], atol=1e-10)
```

**Update reference:**
Only when algorithm intentionally changes (document in ENGINEERING_LOG.md).

---

## 5. VERSION PINNING

### 5.1 Python Environment

**Exact environment for paper results:**

```bash
# requirements-exact.txt (generated from successful run)
numpy==1.24.2
pandas==1.5.3
matplotlib==3.7.0
scipy==1.10.1
numba==0.56.4
click==8.1.3
# ... etc
```

**For reproduction:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-exact.txt
pip install -e .
```

### 5.2 Software Versioning

**Git tag for paper:**
```bash
git tag -a v1.0-paper -m "Version used for paper submission"
git push origin v1.0-paper
```

**Archive on Zenodo:**
- Upload git repository
- Get DOI
- Cite in paper: "Software available at DOI:..."

---

## 6. COMMON PITFALLS

### 6.1 "It Doesn't Reproduce!"

**Checklist:**
1. ✅ Same seed?
2. ✅ Same software versions? (`pip list > versions.txt` and compare)
3. ✅ Same Python version?
4. ✅ Same input data file? (check hash)
5. ✅ Same number of workers? (multiprocessing order may matter)
6. ✅ Same architecture? (x86 vs ARM floating point)

**Most common cause:** Forgot to set seed or used different NumPy version.

### 6.2 Floating Point Differences

**If results differ by <1e-6:**
- Acceptable (rounding differences)
- Use `np.allclose(atol=1e-6)` instead of `np.array_equal`

**If results differ by >1e-3:**
- Problem! Different algorithm path taken
- Debug: check for non-deterministic operations (dict iteration, set order)

### 6.3 Multiprocessing Non-Determinism

**Problem:** Even with seeds, results differ.

**Diagnosis:**
```python
# Are results being combined in non-deterministic order?
results = pool.map(worker, tasks)  # ✅ Order preserved
results = pool.imap_unordered(worker, tasks)  # ❌ Non-deterministic!
```

**Fix:** Always use `map` (not `imap_unordered`), process results in order.

---

## 7. REPRODUCIBILITY CHECKLIST

**Before publishing any result:**

- [ ] Config file created and committed to git
- [ ] Seed explicitly set (not random)
- [ ] Software versions logged in metadata
- [ ] Output hash computed and stored
- [ ] Result reproduced locally at least once
- [ ] Regression test added to test suite
- [ ] Instructions added to `examples/reproduce_paper.md`

**For paper submission:**

- [ ] All experiment configs in `experiments/` directory
- [ ] README includes "Reproducing Results" section
- [ ] Git tag created for paper version
- [ ] Repository archived on Zenodo
- [ ] DOI obtained and cited in paper
- [ ] Hardware specs documented
- [ ] Runtime estimates provided

---

## 8. TROUBLESHOOTING GUIDE

**Issue:** "I get different results on my machine."

**Solution:**
1. Check NumPy version: `pip show numpy`
2. Check seed was set: grep for `seed=` in config
3. Check for platform: x86 vs ARM may differ slightly
4. Run verification hash script
5. Contact: [your email]

**Issue:** "Results are close but not exact."

**Solution:**
- If difference <1e-6: expected (floating point)
- If difference >1e-6: check for algorithm changes
- Compare: simulation output, GA trajectory, final parameters

**Issue:** "GUI produces different results than CLI."

**Solution:**
- Both should use same backend → this is a bug
- File issue with: seed used, config file, output diff

---

*This protocol will be referenced in paper's "Methods" section and thesis appendix*