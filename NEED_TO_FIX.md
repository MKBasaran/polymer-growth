# NEED TO FIX

Post-demo fixes discovered during presentation prep (2026-04-24).

## 1. reproduce_pop1: re-simulates 10x instead of reusing cached result

**File:** `src/polymer_growth/optimizers/fddc.py`, `_reproduce_pop1()` (line ~508)

**Bug:** Each of the 10 sigma evaluations runs a full simulation with a different eval_seed. Thomas's code runs the simulation ONCE, then evaluates the cached distribution against 10 different sigma weightings. Ours conflates stochastic noise averaging with sigma evaluation.

**Thomas's behavior (correct FDDC):**
```python
self.fitnessFunction(child)          # simulate ONCE
for i in range(self.memory_size):
    set_sigma(random_pop2_individual)
    cost = compute_cost()            # reuse cached distribution
```

**Our behavior (wrong):**
```python
tasks = [(child, sigma, different_eval_seed) for each sigma]
# each task re-runs the full simulation
```

**Fix:** Run simulation once for the child (with one eval_seed), cache the Distribution, then evaluate it against 10 sigma weightings without re-simulating. This also makes reproduce_pop1 ~10x cheaper per child.

**Impact on demo:** Minimal. Results are still valid for equivalence testing since both columns used the same FDDC code. The 5k benchmark numbers don't change meaningfully. But this must be fixed before final thesis experiments.

---

## 2. Sigma vector length hardcoded to 100

**File:** `src/polymer_growth/optimizers/fddc.py`, `_initialize_populations()` (line ~288)

**Bug:** `base_sigma_length = 100` is a placeholder. The comment says "will be dynamic" but it never gets set from the actual experimental histogram length. Thomas's code dynamically sizes sigma to match `len(distribution_comparison.non_zero_indices[0])` after running one simulation.

**Effect:**
- 5k dataset (~100 non-zero bins): accidentally correct
- 30k dataset (~300 non-zero bins): sigma has 100 entries covering 300 bins, 3x coarser than Thomas intended

**Fix:** After the initial objective call on line 282, retrieve the experimental histogram length and use it as `base_sigma_length`. This requires either:
- (a) Having the objective function return/expose the histogram length, or
- (b) Passing `len(experimental_values)` into the optimizer constructor

Option (b) is simpler -- add an optional `sigma_length` param to `FDDCConfig`.

---

## 3. Slide text: "FDDC implementation is untouched"

**File:** Presentation slides (slide 12)

**Bug:** Not true. The FDDC was reimplemented from scratch. Two algorithmic differences (items 1 and 2 above) plus engineering changes (parallelization, RNG, ProcessPool).

**Fix:** Change to: "Same FDDC algorithm (coevolution + novelty ranking), reimplemented with engineering improvements. One known deviation in child evaluation will be corrected before final experiments."

---

## 4. One broken integration test

**File:** `tests/test_integration.py`, `test_simple_optimization_toy_problem()`

**Bug:** The test's toy objective wrapper doesn't accept the `eval_seed` kwarg added in the determinism fix.

**Fix:** Add `eval_seed=None` to the test's `objective_wrapper` signature. One line.