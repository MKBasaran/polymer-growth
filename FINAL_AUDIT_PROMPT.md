# Final Integrity Audit -- Post-Fix

Feed this entire document to a fresh Claude Code instance. It will audit the project and return structured findings that can be fed directly back to the development agent.

## Project Context

BSc thesis by Kaan Basaran. Research question:

> "How can legacy scientific simulation code be modernized to improve maintainability, reproducibility, and accessibility for domain scientists while preserving algorithmic correctness?"

Three success criteria:
1. Software engineering modernization (tests, packaging, GUI, reproducibility)
2. Performance improvement (speed, parallelization scaling)
3. Algorithmic equivalence with Thomas van den Broek's (2020) original code

Codebase:
- Original: `program code/` (Thomas's flat Python, Pool(6), Tkinter)
- Refactored: `src/polymer_growth/` (pip package, PySide6, Numba, configurable workers)
- Experiments: `scripts/experiments/`
- Results: `validation_results/`

## SECTION 1: Code Correctness

### 1.1 Simulation Engine Equivalence
Read completely:
- `src/polymer_growth/core/simulation.py` (our simulate() + _simulate_core JIT)
- `program code/simulation.py` (Thomas's polymer())

Verify: same growth formula, same death formula, same vampiric coupling formula, same monomer depletion. Report ANY functional differences (ordering, edge cases, RNG mechanism).

### 1.2 FDDC Optimizer Equivalence
Read completely:
- `src/polymer_growth/optimizers/fddc.py` -- focus on the "FAITHFUL PATH" in optimize() loop and `_reproduce_pop1_single`, `_reproduce_pop2_single`, `_crossover`, `_mutate`, `_crossover_sigma`
- `program code/fddc.py` -- focus on `run()`, `reproduce_pop1`, `reproduce_pop2`, `breed`, `mutate`, `breed_sigma`

Verify line by line:
- Mutation: does ours skip last gene like Thomas (range(len-1))? Does ours have P(small)=0.6, P(large)=0.4, P(unchanged)=0 like Thomas?
- Crossover: does ours randomize which parent contributes inner vs outer like Thomas?
- Sigma crossover: does ours reject duplicate positions like Thomas?
- Re-ranking: does the faithful path re-rank after EACH child like Thomas?
- Reproduction order: does the faithful path interleave pop1/pop2 per child like Thomas?
- Initial fitness: does ours simulate once per individual, cost eval N times, like Thomas's compute_initial_fitness?

Previous audit found 6 HIGH deviations. All were reportedly fixed. VERIFY each fix.

### 1.3 Cost Function Equivalence
Read completely:
- `src/polymer_growth/objective/min_max_v2.py` (our MinMaxV2ObjectiveFunction)
- `program code/distributionComparison.py` lines 304-437 (Thomas's min_maxV2)

Verify: same normalization, same peak alignment, same partition error formula, same exponential penalty, same cost cap. Report ANY differences including off-by-one, bin count, padding/truncation.

### 1.4 Fast Sim / Megabatch Mode
Read `src/polymer_growth/optimizers/fddc.py` -- find `_generation_megabatch`. This is an OPTIMIZED path that deviates from Thomas's algorithm (batched reproduction, single ranking per gen).

Check:
- Is this path ONLY activated when `_sim_hist_fn` and `_cost_from_hist_fn` are set?
- Is the faithful path (interleaved, per-child ranking) the DEFAULT when only `simulate_fn`/`cost_fn` are provided?
- Does any experiment script in `scripts/experiments/` use `use_fast_sim=True`? If so, flag it -- experiments should use the faithful path for correctness claims.

Read `scripts/experiments/shared.py` -- check `run_fddc()`. Does it pass `use_fast_sim` anywhere? Does `exp_final_suite.py` use it?

### 1.5 Dead Code / Unused Functions
Search `src/polymer_growth/` for:
- Functions defined but never called
- Imports not used
- Worker functions that aren't referenced (`_worker_sim_and_cost`, etc.)
- `_simulate_fast_hist` -- is it used or dead?
- `_reproduce_batch` -- is it used or dead (now that faithful path uses `_reproduce_pop1_single`/`_reproduce_pop2_single`)?

### 1.6 False or Misleading Comments
Read `src/polymer_growth/optimizers/fddc.py` lines 16-24. Do the comments about thread suppression honestly state that no BLAS operations exist in this codebase?

## SECTION 2: Experiment Integrity

### 2.1 Experiment A: Speed Comparison
Files: `validation_results/expA_*.json` (6 files expected)
- Both at 6 workers? Same dataset (5k)? Same gens (42)?
- Is `use_fast_sim` used? (It should NOT be for equivalence claims)
- Speedup consistent across 3 seeds?
- Costs in same range?

### 2.2 Experiment B: Worker Scaling  
Files: `validation_results/expB_*.json` (36 files expected)
- Both codebases at [1,2,4,6,8,13] workers?
- Does Thomas plateau/saturate past 8 workers?
- Is our scaling monotonic?

### 2.3 Determinism
Files: `validation_results/exp4_determinism_*.json`
- Are cost histories bit-identical across 3 runs?

### 2.4 Cost Validation
Files: `validation_results/expD_*.json` (4 files expected)
- Does Thomas's own code on modern hardware reproduce his Table VIII costs?
- 5k should match (~21). 10k/20k/30k should be higher (confirming his results came from extended 24hr runs)

### 2.5 Dataset Scaling
Files: `validation_results/expC_*.json` (12 files expected)
- Our code on 5k/10k/20k/30k at 13 workers
- Times reasonable? Costs comparable to Thomas at equal gen counts?

### 2.6 Simulation Equivalence
File: `validation_results/exp_simulation_equivalence.json`
- t-test and KS-test results. All p > 0.05?

### 2.7 SWE Report
File: `validation_results/exp6_swe_report.json`
- Code metrics accurate?

## SECTION 3: Document Integrity

### 3.1 METHODOLOGY.md
- Does it accurately describe the current code behavior (post-fix)?
- Does it mention BLAS suppression as a speed factor? (It shouldn't -- no BLAS ops)
- Does it mention the faithful vs megabatch paths?
- Are experiment procedures reproducible from the descriptions?
- Python version: should be 3.10 via Rosetta, not 3.11

### 3.2 DEFINITIVE_ANALYSIS.md
- Is this document still accurate post-fix? Flag any outdated claims.

### 3.3 OPEN_QUESTIONS.md
- Are the questions still relevant?

## SECTION 4: Repository Hygiene

List EVERY file in the repo root and `scripts/` that should NOT be in a scientific software repository. Examples:
- Screenshots, PNGs, PPTXs, PDFs not part of the software
- Internal planning documents (HANDOFF_PROMPT.md, night.txt, presentation_*.md)
- Test output directories (testingForMetadata/, 3runIdenticals/, 10-04-2026/)
- Stale log files (*.log)
- .venv/ directory
- Archived experiments that should be in .gitignore

For each file/directory, state: DELETE (not needed), GITIGNORE (keep local, exclude from repo), or KEEP (belongs in repo).

## SECTION 5: Thesis Claims Verdicts

For each claim, state PROVEN, UNPROVEN, or FALSE:

1. "The refactored code is faster than Thomas's at equal worker count (6 workers)"
2. "The refactored code scales better with additional workers"  
3. "Thomas's code saturates past 8 workers"
4. "The simulation engine produces statistically equivalent output"
5. "The FDDC optimizer implements the same algorithm as Thomas's"
6. "The cost function produces equivalent results"
7. "The optimization is deterministic (bit-identical across runs with same seed)"
8. "The software is pip-installable"
9. "The refactored code has 29 tests (Thomas has 0)"
10. "Thomas's Table VIII results required more generations than reported"

## SECTION 6: Literature Citations Needed

Check METHODOLOGY.md and the codebase. Which claims need academic citations?
- FDDC algorithm origin
- Numba JIT compilation
- Amdahl's Law
- Steady-state GA theory
- Polymer growth simulation model
- MinMaxV2 cost function origin
- Python multiprocessing fork vs spawn

## Output Format

```
## Section 1: Code Correctness
### 1.1 Simulation Equivalence
VERDICT: [PROVEN/UNPROVEN/FALSE]
DIFFERENCES FOUND: [list with file:line]
IMPACT: [none/cosmetic/functional]

### 1.2 FDDC Equivalence (post-fix)
PREVIOUSLY FLAGGED ISSUE 1 (mutation): [FIXED/STILL BROKEN] -- evidence
PREVIOUSLY FLAGGED ISSUE 2 (re-ranking): [FIXED/STILL BROKEN] -- evidence  
PREVIOUSLY FLAGGED ISSUE 3 (interleaving): [FIXED/STILL BROKEN] -- evidence
PREVIOUSLY FLAGGED ISSUE 4 (crossover): [FIXED/STILL BROKEN] -- evidence
PREVIOUSLY FLAGGED ISSUE 5 (last gene): [FIXED/STILL BROKEN] -- evidence
PREVIOUSLY FLAGGED ISSUE 6 (sigma dupes): [FIXED/STILL BROKEN] -- evidence

### 1.4 Fast Sim Usage in Experiments
FILES USING use_fast_sim=True: [list] -- THESE RESULTS ARE TAINTED IF USED FOR EQUIVALENCE CLAIMS

### 1.5 Dead Code
FILES/FUNCTIONS TO REMOVE: [list with paths]

## Section 2: Experiment Integrity
[per-experiment verdict]

## Section 3: Document Integrity  
[per-document verdict, list of outdated claims]

## Section 4: Repository Hygiene
| Path | Action | Reason |
|------|--------|--------|
| ... | DELETE/GITIGNORE/KEEP | ... |

## Section 5: Thesis Claims
| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|

## Section 6: Literature Gaps
| Topic | Source | Present in docs? |
|-------|--------|-----------------|

## CRITICAL ACTIONS (must do before submission)
1. ...

## WARNINGS (should do)
1. ...

## CONFIRMED GOOD
1. ...
```
