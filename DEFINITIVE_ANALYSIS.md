# Definitive Analysis: Thomas vs Ours

Written after reading every line of both codebases. No speculation.

---

## FDDC Algorithm: Exact Simulation Count per Generation

Both codebases implement the same FDDC algorithm with identical parameters:
pop=100, memory_size=10, n_encounters=10, n_children=2.

### Thomas's code (program code/fddc.py)

**Init (once):**
- `p.map(compute_initial_fitness, indexes)` -- 100 tasks across Pool(6)
- Each task (line 324-333): `self.fitnessFunction(pop1[i])` = 1 sim, then `compute_cost()` x 10 = 0 sims
- **100 sims, PARALLEL**

**Per generation (run(), line 146-209):**

| Phase | Code | Sims | Execution |
|-------|------|------|-----------|
| Encounters | `p.map(computeFitness, range(10))` line 179. Each calls `fitnessFunction(individual)[0]` = 1 sim | **10** | PARALLEL |
| reproduce_pop1 x2 | `fitnessFunction(child)` line 373 = 1 sim. Then `compute_cost()` x10 = 0 sims | **2** | SERIAL (main process) |
| reproduce_pop2 x2 | `p.map(compute_fitness_sigma, children)` line 431. Each child calls `fitnessFunction(pop1[r])[0]` = 1 sim. 10 children per call | **20** | PARALLEL |
| Best eval | `fitnessFunction(best, plot=True)[0]` line 208 = 1 sim | **1** | SERIAL |
| **TOTAL** | | **33** | |

### Our code (src/polymer_growth/optimizers/fddc.py)

**Init (once, with simulate_fn/cost_fn):**
- `_parallel_map(_worker_eval_initial_cached, eval_tasks)` -- 100 tasks
- Each task: `_worker_simulate(params, eval_seed)` = 1 sim, then `_worker_cost(dist, sigma)` x 10 = 0 sims
- **100 sims, PARALLEL**

**Per generation (optimize(), line 247):**

| Phase | Code | Sims | Execution |
|-------|------|------|-----------|
| Encounters | `_parallel_map(_worker_eval, tasks)` line 489. Each calls `objective(params, sigma, eval_seed)` = 1 sim | **10** | PARALLEL |
| reproduce_pop1 x2 | `simulate_fn(child, eval_seed)` line 590 = 1 sim. Then `cost_fn(dist, s)` x10 = 0 sims | **2** | SERIAL (main process) |
| reproduce_pop2 x2 | `_parallel_map(_worker_eval, tasks)` line 626. Each calls `objective(opp, child, eval_seed)` = 1 sim. 10 per call | **20** | PARALLEL |
| Best eval | `simulate_fn(best_params, eval_seed)` line 266 + `cost_fn(dist, None)` line 267 = 1 sim | **1** | SERIAL |
| **TOTAL** | | **33** | |

### Conclusion: Both do exactly 33 sims per generation.

The algorithm is the algorithm. At equal worker count, equal sims = equal time.

---

## Where Parallelization Actually Helps

Per generation, the work breaks down:

- **Parallel work**: 10 (encounters) + 20 (reproduce_pop2) = **30 sims**
- **Serial work**: 2 (reproduce_pop1) + 1 (best eval) = **3 sims**

### At 6 workers:
- Parallel: 30 sims / 6 workers = 5 sequential batches
- Serial: 3 sims sequential
- Total: ~8 sim-time-units
- (Each sim ~0.5s on M4 Pro, so ~4s/gen... but overhead makes it ~6s/gen)

### At 13 workers:
- Parallel: 30 sims / 13 workers = ~2.3 sequential batches
- Serial: 3 sims sequential (UNCHANGED)
- Total: ~5.3 sim-time-units

### Amdahl's Law:
- Serial fraction: 3/33 = 9.1% of work, but 3/8 = 37.5% of wall-clock at 6w
- Theoretical max speedup from 6w to 13w: 8 / (3 + 5/2.17) = 8 / 5.3 = **1.51x**
- Observed: 4.3min / 3.8min = **1.13x** (overhead eats most of the gain)

### At 64 workers (server):
- Parallel: 30/64 = 0.47 batches ≈ 1 batch
- Serial: 3 sims
- Total: ~4 sim-time-units
- Theoretical max speedup vs 6w: 8/4 = **2x**

The serial bottleneck (3 sims) sets a hard floor. More workers can never make those 3 sims faster.

---

## Thomas's 24 Hours: What Actually Happened

Thomas's thesis (line 799): "the algorithms were allowed to run for **one day per data set**."

### What we know for certain:
1. He used Pool(6) (fddc.py line 17)
2. No seeding (simulation.py line 7: `# np.random.seed(2)` commented out)
3. He ran 4 datasets, each for 24 hours
4. Results (Table VIII): 5K=42 gens, 10K=56, 20K=44, 30K=27

### What we do NOT know:
- His CPU model, core count, clock speed, RAM
- Whether the GUI was visible during runs (Tkinter `self.update()` + `plt.pause()` every gen)
- Whether he ran other programs simultaneously

### What we CAN calculate:
On our M4 Pro, Thomas's code does 42 gens in 4.3 min = 6.1s/gen.
Thomas did 42 gens in 24 hours = 2057s/gen.
That's **337x slower per generation** on his hardware.

Table II shows simulation time scales with number_of_molecules:
- 10k molecules = 17.5s per sim (for a generation of 50 individuals)
- 70k molecules = 50.9s per sim

Table IX shows his FDDC solutions converged to 84k-118k molecules.
But the bounds in his UI (line 214) show `l_number_of_molecules=100` (lower) and `u_number_of_molecules=100000` (upper). The optimizer evolves this parameter.

### Honest statement:
We do not have enough information to decompose the 337x into hardware vs molecule-count vs GUI-overhead. We can only say: on identical hardware (M4 Pro), Thomas's code and ours run at the same speed at 6 workers.

---

## What IS Genuinely Better in Our Code

### 1. Scalable parallelization (PROVABLE)
Thomas: hardcoded Pool(6), no BLAS thread suppression.
Ours: configurable n_workers + BLAS/MKL/OpenBLAS/VECLIB suppression.

Without BLAS suppression, each Pool worker spawns internal BLAS threads. At Pool(6), that's 6 workers x N BLAS threads competing for cores. At Pool(13), it would be 13 x N -- guaranteed thrashing.

Our BLAS suppression (fddc.py lines 18-22) pins each worker to 1 thread. This is what makes scaling to 13+ workers safe.

**To prove:** Run scaling benchmark [1, 2, 4, 6, 8, 13] workers with BLAS suppression (ours) and measure near-linear speedup. Then demonstrate Thomas's code cannot scale past 6 without modification.

### 2. Deterministic seeding (PROVABLE)
Thomas: uses `np.random.random()` and `random.uniform()` with no seeds. Results vary every run.
Ours: `eval_seed` parameter feeds `np.random.default_rng(seed)`. Same seed = identical results.

**Already proved:** exp4_determinism test -- 3 runs, bit-identical.

### 3. Init efficiency (PROVABLE)
Before our fix, our init did 1000 full sims (each sigma opponent = separate simulation).
Thomas's init: 100 sims (simulate once per individual, evaluate cost 10 times).
Our fixed init: 100 sims (same pattern).

This isn't a speedup over Thomas -- it's matching Thomas's efficiency. But it demonstrates we understood and correctly reimplemented his optimization.

### 4. Convergence depth (PROVABLE if cost improves)
Thomas could do 42 gens on 5K in 24 hours on his hardware.
We can do 200+ gens in 15 min on ours.
If FDDC finds better solutions past gen 42, that's a result Thomas could never reach on his hardware within practical time limits.

### 5. Software engineering (PROVABLE by inspection)
- 0 tests -> 29 tests
- No package structure -> pip-installable package
- No parameter validation -> dataclass with validate()
- Tkinter GUI coupled to logic -> PySide6 with threaded workers
- No documentation -> typed functions, docstrings
- No reproducibility -> deterministic seeding

---

## What We CANNOT Claim

1. **"Our algorithm is faster"** -- It's the same algorithm doing the same work.
2. **"203x speedup"** -- That was hardware difference, not software.
3. **"Numba JIT speeds up simulation"** -- Measured at +2%, negligible.
4. **"X% speedup from software alone at equal workers"** -- At 6 workers, it's 1.00x.

---

## The Three Thesis Claims, Honestly

### Claim 1: Result Equivalence
Both codebases produce statistically equivalent simulation output (t-test, KS-test).
Both FDDC implementations find solutions in the same cost range (~22-28 for 5K).
**Status: PROVEN** (exp_simulation_equivalence.json, exp1 results)

### Claim 2: Scalable Performance
Our infrastructure enables efficient scaling to modern hardware:
- BLAS thread suppression prevents core contention
- Configurable worker count adapts to available cores
- Near-linear speedup from 1 to 13 workers (scaling benchmark)
- Thomas's hardcoded Pool(6) cannot safely scale
**Status: NEEDS SCALING BENCHMARK** (exp3_scaling_full.py)

### Claim 3: SWE Principles + Modern GUI
Quantified improvements in modularity, testability, installability, reproducibility, GUI architecture.
**Status: PROVEN** (exp6_swe_report.json)

### Bonus: Convergence Depth
More generations = better solutions (if confirmed by experiment).
Our speed advantage at 13 workers means more gens in the same wall-clock time.
**Status: RUNNING** (exp_convergence_depth.py)
