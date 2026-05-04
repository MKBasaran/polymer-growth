# Integrity Audit Prompt

Feed this entire document to a fresh Claude Code instance. It will audit the project and return structured findings.

---

## Context

You are auditing a BSc thesis project that refactors Thomas van den Broek's (2020) polymer growth simulation software. The thesis research question is:

> "How can legacy scientific simulation code be modernized to improve maintainability, reproducibility, and accessibility for domain scientists while preserving algorithmic correctness?"

The project claims three things:
1. **Software engineering improvements**: tests, packaging, deterministic seeding, modular GUI
2. **Performance improvements**: 1.87x faster at equal workers, 5.56x scaling at 13 workers vs Thomas's 3.33x
3. **Algorithmic equivalence**: both codebases produce statistically equivalent results

## Your Task

Read every file listed below. For each section, output structured findings. Do not skip files. Do not summarize without reading.

## Section 1: Code Integrity

Read these files completely and verify:

### 1a. Does our simulation produce equivalent output to Thomas's?

Files to read:
- `src/polymer_growth/core/simulation.py` -- our simulation engine
- `program code/simulation.py` -- Thomas's simulation engine
- `validation_results/exp_simulation_equivalence.json` -- statistical equivalence test results

Check: Is the simulation logic identical? Same growth/death/vampiric coupling formulas? Same monomer depletion? If there are differences, are they cosmetic (code style) or functional (different results)?

### 1b. Does our FDDC optimizer implement the same algorithm as Thomas's?

Files to read:
- `src/polymer_growth/optimizers/fddc.py` -- our FDDC (focus on `_generation_megabatch`, `_reproduce_batch`, `_evaluate_initial_fitness_parallel`)
- `program code/fddc.py` -- Thomas's FDDC (focus on `run`, `reproduce_pop1`, `reproduce_pop2`, `compute_initial_fitness`, `computeFitness`)

Check:
- Same number of simulations per generation? (Should be 33: 10 encounters + 2 pop1 + 20 pop2 + 1 best)
- Same selection mechanism? (rank-based, power=1.5)
- Same crossover? (two-point for pop1, special logic for pop2)
- Same mutation? (0.6 rate, 0.001 strength)
- Same memory/FIFO update? (size=10, pop after append)
- Same novelty ranking for pop2?
- Any behavioral differences introduced by batching or megabatch dispatch?

### 1c. Does our cost function match Thomas's?

Files to read:
- `src/polymer_growth/objective/min_max_v2.py` -- our MinMaxV2
- `program code/distributionComparison.py` -- Thomas's min_maxV2 (class at line 304)

Check: Same normalization? Same peak alignment? Same partition-based error? Same exponential penalty? Same cost cap at 100000? Any differences in how sigma is applied?

### 1d. Are there any false or misleading claims in code comments?

Read `src/polymer_growth/optimizers/fddc.py` lines 16-22 (BLAS suppression comments). Verify: does any code in `src/polymer_growth/core/simulation.py` or `src/polymer_growth/objective/min_max_v2.py` call BLAS operations (np.dot, np.matmul, np.linalg.*, scipy.linalg.*)? If not, are the comments honest about this?

## Section 2: Experiment Integrity

### 2a. Speed comparison (Experiment A)

Files: `validation_results/expA_*.json` (6 files)

Check:
- Are Thomas and ours both at 6 workers? (look at `workers` field)
- Is the dataset the same? (should be "5k")
- Are generation counts the same? (should be 42)
- Is the speedup ratio consistent across seeds?
- Are the costs in the same range? (both should be ~20-25 for 5k)

### 2b. Worker scaling (Experiment B)

Files: `validation_results/expB_*.json` (36 files)

Check:
- Does each file have the correct worker count in the `workers` or `n_workers_actual` field?
- Is Thomas's code at 13 workers slower than at 8 workers? (claimed plateau)
- Is our speedup at 13 workers realistic? (5.56x from 1 worker baseline)
- Are costs comparable across worker counts for the same seed?

### 2c. Determinism

Files: `validation_results/exp4_determinism_summary.json`, `exp4_determinism_run*.json`

Check: Are the cost histories truly bit-identical across all 3 runs?

### 2d. Cost validation

Files: `validation_results/expD_*.json` (4 files)

Check: Does Thomas's own code on modern hardware produce costs significantly higher than his Table VIII for 10k/20k/30k? This validates that his published results came from extended runs, not the reported generation counts.

## Section 3: SWE Principles

### 3a. Package structure

Check:
- `pyproject.toml` -- valid metadata, classifiers, entry points?
- Can the package be installed with `pip install -e .`?
- Are `__init__.py` files exporting the right public API?
- Is `ParameterBounds` used consistently (no hardcoded bounds elsewhere)?

### 3b. Testing

Files: `tests/test_simulation.py`, `tests/test_objective.py`, `tests/test_integration.py`

Check: Do tests cover the core simulation, cost function, and optimizer? Run `python -m pytest tests/ -v` and report results.

### 3c. Input validation

Check: Do `FDDCOptimizer.__init__` and `MinMaxV2ObjectiveFunction.__init__` validate their inputs? What happens with None, empty array, wrong shape?

### 3d. Thread safety

File: `src/polymer_growth/gui/workers.py`

Check: Is the `_is_cancelled` flag accessed via a lock? Is the `cancel()` method thread-safe?

### 3e. Resource cleanup

File: `src/polymer_growth/optimizers/fddc.py`

Check: Is the process pool cleaned up in a `try/finally` block?

## Section 4: Thesis Claims Audit

Read `METHODOLOGY.md` and `OPEN_QUESTIONS.md`.

For each claim below, state PROVEN, UNPROVEN, or FALSE with evidence:

1. "1.87x faster at equal workers (6 workers)" -- check expA data
2. "5.56x scaling at 13 workers" -- check expB data
3. "Thomas's code plateaus at 3.33x and degrades past 8 workers" -- check expB thomas data
4. "Deterministic: bit-identical across runs" -- check exp4 data
5. "Simulation equivalence: t-test and KS-test pass" -- check exp_simulation_equivalence.json
6. "Cost equivalence at equal generation counts" -- check expD data
7. "0 tests to 29 tests" -- run pytest, count
8. "Pip-installable package" -- check pyproject.toml and try install

## Section 5: Literature Review Pointers

Read the following and note which claims need citation:

1. FDDC algorithm -- cite Paredis (1995) "Coevolutionary Computation" and (1999) "Coevolution, Memory and Balance"
2. Numba JIT compilation -- cite Lam et al. (2015) "Numba: a LLVM-based Python JIT compiler"
3. Amdahl's Law for parallel scaling analysis -- cite Amdahl (1967)
4. Steady-state GA replacement rates -- cite relevant GA textbook (De Jong, Goldberg, or Eiben & Smith)
5. Polymer growth simulation model -- cite the original chemistry papers Thomas references
6. MinMaxV2 cost function -- is this Thomas's invention or from literature? Check his thesis text in `docs/thesis.txt`
7. Multiprocessing Pool with fork vs spawn -- cite Python documentation or relevant systems paper

## Output Format

Return your findings as:

```
## Section 1: Code Integrity
### 1a. Simulation Equivalence
VERDICT: [PROVEN/UNPROVEN/FALSE]
EVIDENCE: [specific file, line numbers, values]
ISSUES: [list any problems found]

### 1b. FDDC Algorithm Equivalence
... (same format)

## Section 2: Experiment Integrity
... (same format for each experiment)

## Section 3: SWE Principles
... (same format)

## Section 4: Thesis Claims
| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|

## Section 5: Literature Gaps
| Topic | Needs Citation? | Suggested Source |
|-------|----------------|-----------------|

## Critical Issues (must fix before submission)
1. ...

## Warnings (should fix)
1. ...

## Confirmed Good
1. ...
```
