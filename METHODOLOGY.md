# Methodology

This chapter describes the experimental design used to evaluate the refactored polymer growth simulation software against the original implementation by van den Broek (2020). Seven experiments were conducted, each isolating a specific research question. All experiments were executed on an Apple M4 Pro with 14 CPU cores, 24 GB unified memory, and macOS Sequoia 15.x. Python 3.10 (x86_64 via Rosetta 2) was used throughout.

The original codebase is referred to as "Thomas's implementation" and the refactored codebase as "the refactored implementation." Both codebases implement the same FDDC (Fitness-Diversity Driven Co-evolution) optimizer and MinMaxV2 cost function. The experimental datasets (5k, 10k, 20k, 30k molecular weight targets) and generation counts are drawn from Table VIII of van den Broek's thesis. All experiment scripts, raw result files, and analysis code are included in the repository under `scripts/experiments/` and `validation_results/`.


## 4.1 Experiment A: Software Speed Comparison

### Purpose

This experiment answers the question: does the refactored implementation execute faster than the original when all other variables are held constant? The comparison isolates the effect of code-level changes -- Numba JIT compilation of the simulation core, batched parallel dispatch, and histogram-based inter-process communication -- on end-to-end optimization runtime.

### Variables

The independent variable was the codebase implementation, taking two levels: Thomas's original code and the refactored code. The dependent variable was wall-clock execution time in seconds, measured from the start of optimizer construction through the final generation. Controlled variables included the worker count (6 processes), the dataset (5k no BB), the generation count (42), the population size (100), and the hardware platform. The FDDC hyperparameters -- memory size (10), number of encounters (10), number of children (2), mutation rate (0.6), mutation strength (0.001), two-point crossover, sigma points per index (4), and rank selection power (1.5) -- were held constant across both implementations.

### Procedure

Three random seeds (42, 123, 777) were used per implementation, yielding six total runs. For Thomas's implementation, the experiment script instantiated Thomas's `fddc` class and `min_maxV2` cost function directly from the original source files. Matplotlib state was nullified before optimization to prevent pickle errors in worker processes. For the refactored implementation, the `FDDCOptimizer` class was instantiated with the Numba-accelerated `_simulate_fast` function and histogram-based megabatch evaluation mode enabled. Each run recorded the total elapsed time, cost history per generation, and best cost achieved. Results were serialized to JSON files named `expA_{impl}_5k_6w_seed{seed}.json`.

To reproduce this experiment, one executes `python scripts/experiments/exp_final_suite.py --exp A` from the project root.

### Statistical Method

The mean and standard deviation of execution times were computed across the three seeds for each implementation. The speedup ratio was calculated as the mean time of Thomas's implementation divided by the mean time of the refactored implementation. No inferential statistical test was applied because the sample size (n=3 per group) is too small for parametric tests and the effect size was expected to be large enough for descriptive comparison.

### Expected Outcome and Interpretation

The refactored implementation was expected to complete in less time than the original due to Numba JIT compilation of the simulation inner loop and reduced inter-process communication overhead. A speedup ratio greater than 1.0 indicates the refactored code is faster. The observed result was a 1.87x speedup (mean 140 seconds vs. 262 seconds). Both implementations were expected to achieve comparable best costs at equal generation counts, confirming that the speed improvement did not compromise algorithmic behavior.


## 4.2 Experiment B: Parallelization Scaling

### Purpose

This experiment measures how execution time scales with the number of worker processes for both implementations. It answers two questions: how close does each implementation come to ideal linear speedup, and at what point does parallelization overhead dominate the gains?

### Variables

The independent variable was the number of worker processes, tested at six levels: 1, 2, 4, 6, 8, and 13. The dependent variables were wall-clock execution time (seconds), speedup ratio (T_1 / T_n, where T_1 is the single-worker time and T_n is the n-worker time), and parallel efficiency (speedup / n, expressed as a percentage). Controlled variables included the dataset (5k no BB), generation count (42), population size (100), and all FDDC hyperparameters as specified in Experiment A. Both implementations were tested at each worker count.

### Procedure

For each combination of implementation, worker count, and seed (3 seeds per condition), a full FDDC optimization was executed. Thomas's code was modified at runtime by monkey-patching the `multiprocessing.Pool.__init__` method to override the hardcoded pool size of 6 workers. The refactored code accepted the worker count as a configuration parameter (`n_workers`). Each run produced a JSON result file named `expB_{impl}_5k_{w}w_seed{seed}.json`. After all runs completed, the mean execution time across seeds was computed for each (implementation, worker count) pair. The single-worker mean time served as the baseline for computing speedup ratios.

To reproduce this experiment, one executes `python scripts/experiments/exp_final_suite.py --exp B` from the project root.

### Statistical Method

Mean execution time and standard deviation were computed across seeds for each condition. Speedup was computed as T_1 / T_n for each worker count n. Parallel efficiency was computed as (T_1 / T_n) / n. These metrics follow the standard parallel performance analysis framework described by Amdahl (1967). No inferential test was applied; the analysis is descriptive.

### Expected Outcome and Interpretation

Both implementations were expected to show sub-linear speedup due to Amdahl's Law: serial portions of the algorithm (rank computation, selection, crossover, mutation) cannot be parallelized. The refactored implementation was expected to achieve higher speedup at large worker counts because its histogram-based IPC transfers approximately 4 KB per task versus approximately 560 KB for full `Distribution` objects. The observed result was a maximum speedup of 5.56x at 13 workers for the refactored code versus a 3.33x plateau for Thomas's code. An efficiency below 50% at a given worker count indicates diminishing returns from additional processes.


## 4.3 Experiment C: Dataset Generalization

### Purpose

This experiment tests whether the refactored implementation generalizes across different experimental datasets. It verifies that performance gains observed on the 5k dataset in Experiments A and B are not dataset-specific artifacts.

### Variables

The independent variable was the experimental dataset, taking four levels: 5k, 10k, 20k, and 30k molecular weight targets, each stored as an Excel file containing GPC (gel permeation chromatography) distribution data. The dependent variables were execution time and best cost achieved after the number of generations specified in van den Broek's Table VIII (42 for 5k, 56 for 10k, 44 for 20k, 27 for 30k). Controlled variables included the worker count (13), population size (100), and all FDDC hyperparameters. Only the refactored implementation was tested.

### Procedure

For each dataset, three runs were executed with seeds 42, 123, and 777. Each run used the refactored optimizer with Numba JIT simulation and megabatch evaluation. The generation count matched the FDDC column of van den Broek's Table VIII. Results were saved as `expC_ours_{dataset}_13w_seed{seed}.json`. After all runs completed, mean execution time and mean best cost were computed per dataset and compared against the Table VIII reference costs.

To reproduce this experiment, one executes `python scripts/experiments/exp_final_suite.py --exp C` from the project root.

### Statistical Method

Mean and standard deviation of execution time and best cost were computed across seeds for each dataset. The observed costs were compared descriptively to van den Broek's Table VIII FDDC costs. No inferential test was applied because the comparison is against published single-run values rather than a distribution of outcomes.

### Expected Outcome and Interpretation

The refactored implementation was expected to produce costs comparable to Table VIII across all four datasets, demonstrating that the code changes did not introduce dataset-specific regressions. Execution time was expected to scale with the generation count and dataset complexity. Costs higher than Table VIII would suggest either a difference in the number of generations actually used to produce the published results or stochastic variation. The observed results showed consistent performance across datasets, with costs matching or exceeding Table VIII values at the specified generation counts.


## 4.4 Experiment D: Cost Function Validation

### Purpose

This experiment investigates a discrepancy observed in Experiment C: for the 10k, 20k, and 30k datasets, neither implementation achieves the costs reported in Table VIII within the stated generation counts. By running Thomas's unmodified code on modern hardware, this experiment determines whether the published Table VIII results were produced with the reported generation counts or with a longer computational budget (e.g., a 24-hour run with an unknown number of generations).

### Variables

The independent variable was the dataset, taking four levels: 5k, 10k, 20k, and 30k. The dependent variable was the best cost achieved by Thomas's original code after the generation count specified in Table VIII. All other variables were controlled: worker count (6, Thomas's hardcoded default), population size (100), and FDDC hyperparameters matching Thomas's source code defaults.

### Procedure

Thomas's `fddc.py` and `distributionComparison.py` modules were executed directly, without modification, for each dataset at the generation count from Table VIII. The optimizer was initialized with Thomas's `min_maxV2` cost function. The cost history was recorded at each generation. The minimum cost across all generations was compared to the Table VIII FDDC cost. Results were saved as `expD_thomas_{dataset}_6w.json`.

To reproduce this experiment, one executes `python scripts/experiments/exp_cost_validation.py` from the project root.

### Statistical Method

No statistical test was applied. The comparison was between a single observed run and a single published reference value. The ratio of the observed cost to the Table VIII cost was computed: a ratio near 1.0 indicates agreement; a ratio substantially greater than 1.0 indicates the published result required more generations than reported.

### Expected Outcome and Interpretation

If Thomas's code produced costs matching Table VIII at the stated generation counts, the discrepancy observed in Experiment C would indicate a difference between the implementations. If Thomas's code also produced higher costs than Table VIII, the conclusion would be that the published results came from extended runs. The observed result confirmed the latter: Thomas's code on modern hardware produced costs higher than Table VIII for the 10k, 20k, and 30k datasets, indicating that the published results were obtained from 24-hour budget runs rather than the stated generation counts. The 5k dataset was the exception, where costs matched Table VIII within expected stochastic variation.


## 4.5 Experiment E: Determinism Verification

### Purpose

This experiment verifies that the refactored implementation produces bit-identical results when executed repeatedly with the same random seed and configuration. The original implementation lacked deterministic seeding: it relied on NumPy's global random state, and the FDDC optimizer did not propagate seeds to worker processes. The refactored implementation introduced an `eval_seed` parameter that derives a unique per-evaluation seed from the optimizer's master RNG, ensuring reproducibility.

### Variables

The independent variable was the run number (1, 2, 3), each using identical configuration. The dependent variable was the cost history vector -- the sequence of best-cost values across all generations. Controlled variables included all configuration parameters: seed (42), dataset (5k), population size (50), generation count (10), and worker count (default). Every aspect of the configuration was held constant across runs.

### Procedure

The refactored optimizer was executed three times with seed 42, population size 50, and 10 generations on the 5k dataset. After each run, the complete cost history (a list of floating-point values, one per generation) and the best parameter vector were saved to `exp4_determinism_run{n}.json`. After all three runs completed, the cost histories were compared element-wise using exact equality (Python `==` on lists of floats). The best parameter vectors were also compared element-wise.

To reproduce this experiment, one executes `python scripts/experiments/exp_determinism.py` from the project root.

### Statistical Method

No statistical test was applied. The criterion was exact bitwise equality of cost histories across runs. Any deviation in any generation would constitute a failure.

### Expected Outcome and Interpretation

All three runs were expected to produce identical cost histories and identical best parameter vectors. A passing result confirms that the `eval_seed` mechanism correctly eliminates non-determinism from the optimization loop. A failing result would indicate a source of randomness not controlled by the seed -- for example, process scheduling order affecting floating-point accumulation, or an unseeded random call in a worker process. The observed result was a pass: all three runs produced bit-identical cost histories.


## 4.6 Experiment F: Simulation Engine Equivalence

### Purpose

This experiment tests whether the refactored simulation function (`simulate`) produces statistically equivalent output to Thomas's original simulation function (`polymer`). The refactored function uses NumPy's `Generator` API for random number generation and includes a Numba JIT-compiled fast path, while Thomas's function uses NumPy's legacy global random state. Both implementations encode the same agent-based polymer growth model: living chain growth, chain death, and vampiric coupling reactions.

### Variables

The independent variable was the simulation implementation (Thomas's `polymer` vs. the refactored `simulate`). The dependent variables were three polymer characterization metrics computed from each simulation's output: number-average molecular weight (Mn), weight-average molecular weight (Mw), and polydispersity index (PDI = Mw/Mn). The controlled variables were the simulation parameters, which were held constant within each parameter set.

Four parameter sets were tested to cover different simulation regimes: (1) `thomas_default` -- parameters from van den Broek's published optimized result with 100,000 molecules; (2) `high_growth` -- a regime with elevated growth probability (0.90) and 50,000 molecules; (3) `low_death` -- a regime equivalent to the default parameters but tested for consistency; and (4) `no_respawn` -- the same default parameters with `kill_spawns_new` set to False, testing the branch where dead chains are removed rather than respawned.

### Procedure

For each parameter set, 10 runs were executed per implementation. Each run used a deterministic seed (seeds 1000 through 1009). For the refactored implementation, `np.random.default_rng(seed)` was used. For Thomas's implementation, `np.random.seed(seed)` was called before each simulation to set the global state. From each run, the complete chain length distribution (living, dead, and coupled arrays) was collected. Mn, Mw, and PDI were computed from the combined chain lengths using standard polymer characterization formulas: Mn = mean(DP * M_monomer + M_initiator), Mw = sum(MW^2) / sum(MW), PDI = Mw / Mn.

To reproduce this experiment, one executes `python scripts/validate_thomas_model.py --runs 10` from the project root.

### Statistical Method

Two statistical tests were applied to each (parameter set, metric) combination at a significance level of alpha = 0.05.

First, Welch's t-test (two-sample t-test with unequal variances) was used to test whether the means of Mn, Mw, and PDI differed between implementations. The null hypothesis H0 was that the population means are equal. Welch's t-test was chosen over Student's t-test because equal variance between implementations cannot be assumed: the two RNG backends may produce different variance characteristics.

Second, the two-sample Kolmogorov-Smirnov test was applied to each metric to test whether the two samples were drawn from the same continuous distribution. This non-parametric test is sensitive to differences in shape, spread, and location, providing a complementary check to the t-test.

For each metric, a p-value below 0.05 from either test would indicate a statistically detectable difference between implementations.

### Expected Outcome and Interpretation

Both implementations encode the same mathematical model, so no significant differences were expected. The RNG backends differ (legacy `np.random` vs. `np.random.Generator`), meaning individual runs with the same seed will not produce identical chains. But the statistical distributions of output metrics across runs should be indistinguishable. A result where all p-values exceed 0.05 supports the conclusion that the refactored simulation is equivalent to the original. The observed result was exactly this: all p-values exceeded 0.05 across all parameter sets and all metrics, and the null hypothesis was not rejected in any case.


## 4.7 Experiment G: Software Engineering Metrics

### Purpose

This experiment provides a structured quantitative and qualitative comparison of the two codebases along software engineering dimensions. It answers the question: has the refactoring improved maintainability, testability, and accessibility as measured by standard code metrics?

### Variables

This is not a statistical experiment. There is no independent variable in the experimental sense. The comparison is between two fixed codebases. The quantitative metrics collected were: number of Python files, total lines of code, code lines (excluding blanks and comments), average lines per file, number of test files, number of test functions, and number of declared dependencies. Cyclomatic complexity (mean, median, maximum) was computed using the `radon` static analysis tool where available. The qualitative dimensions assessed were: modularity (package structure), testability (test infrastructure), installability (dependency management), reproducibility (deterministic seeding), parallelization (worker configurability), and GUI architecture (framework and coupling).

### Procedure

A Python script (`scripts/experiments/generate_swe_report.py`) traversed both codebases and computed line counts by category (code, blank, comment) for each file. Test files matching `test_*.py` in the `tests/` directory were scanned for functions matching `def test_*`. Dependencies were parsed from `pyproject.toml` for the refactored code; Thomas's code has no dependency manifest and dependencies were enumerated manually from import statements. Cyclomatic complexity was computed by invoking `radon cc` on each codebase directory. The qualitative comparison was assembled from direct code inspection.

To reproduce this experiment, one executes `python scripts/experiments/generate_swe_report.py` from the project root.

### Statistical Method

No statistical method was applied. Metrics were reported descriptively. Line counts and file counts are exact values, not samples from a distribution.

### Expected Outcome and Interpretation

The refactored codebase was expected to have more files (due to modular decomposition) but lower average lines per file. The refactored codebase was expected to have a non-zero test count, while Thomas's codebase has zero tests. The presence of `pyproject.toml` with declared dependencies, a pip-installable package structure, and a PySide6-based GUI decoupled from optimization logic were expected as qualitative improvements. These metrics do not prove that the refactored code is "better" in any absolute sense; they provide evidence that specific engineering practices (modularity, testing, dependency management, reproducibility) were adopted during the refactoring.

The results confirmed these expectations. Thomas's codebase consisted of a flat directory of Python scripts with no test infrastructure, no dependency manifest, hardcoded parallelization, and a Tkinter GUI coupled to optimization logic. The refactored codebase is a pip-installable package organized into `core/`, `objective/`, `optimizers/`, `gui/`, and `cli/` subpackages, with declared dependencies in `pyproject.toml`, a pytest-based test suite, configurable worker counts, and a PySide6 GUI operating on a threaded worker architecture. Full results are presented in Table X.
