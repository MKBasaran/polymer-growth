"""
FDDC: Fitness-Diversity Driven Co-evolution optimizer.

Two-population co-evolutionary genetic algorithm:
- Population 1: Simulation parameters (minimize cost)
- Population 2: Cost function weights (maximize diversity)

Based on algorithm from previous research:
    - Converges in ~20 generations vs ~77 for baseline GA
    - Uses novelty ranking for population 2 to maintain diversity
    - Memory-based fitness to handle stochastic evaluations
"""

import os

# Prevent numpy/BLAS from spawning internal threads that compete with our
# process-level parallelism. Must be set before numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import multiprocessing as mp
from multiprocessing.pool import Pool
import time

# Force fork start method for true parallelism (workers inherit globals)
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

# Module-level function references for multiprocessing workers.
# Workers forked from the main process inherit these via fork().
_worker_objective = None
_worker_simulate = None
_worker_cost = None


def _worker_eval(args):
    """Evaluate a single (params, sigma, eval_seed) tuple in a worker process."""
    params, sigma, eval_seed = args
    return _worker_objective(params, sigma=sigma, eval_seed=eval_seed)


def _worker_eval_indexed(args):
    """Evaluate with index tracking for initial population."""
    pop1_idx, pop2_idx, params, sigma, eval_seed = args
    cost = _worker_objective(params, sigma=sigma, eval_seed=eval_seed)
    return (pop1_idx, pop2_idx, cost)


def _worker_sim(args):
    """Simulate only, return distribution. For batched reproduce_pop1."""
    params, eval_seed = args
    return _worker_simulate(params, eval_seed)


def _worker_sim_and_cost(args):
    """Simulate + compute cost in worker. Returns float, not Distribution."""
    params, sigma, eval_seed = args
    dist = _worker_simulate(params, eval_seed)
    return _worker_cost(dist, sigma)


_worker_sim_hist_fn = None  # _simulate_fast_hist function ref
_worker_cost_from_hist = None  # cost function that takes histogram


def _worker_megabatch(args):
    """ONE worker function for entire generation. Minimal IPC.

    Each task is (params_tuple, sigma_or_none, seed, hist_length).
    - sigma is ndarray: sim -> hist -> cost with that sigma -> return float
    - sigma is scalar (0): sim -> hist -> cost with default sigma -> return float
    - sigma is None: sim -> hist -> return histogram (for multi-sigma eval)
    """
    params_tuple, sigma, seed, hist_length = args
    hist = _worker_sim_hist_fn(params_tuple, seed, hist_length)
    if sigma is None:
        return ('h', hist)
    elif np.ndim(sigma) == 0:
        return ('c', _worker_cost_from_hist(hist, None))
    else:
        return ('c', _worker_cost_from_hist(hist, sigma))


def _worker_eval_initial_cached(args):
    """Simulate once, evaluate cost against multiple sigmas.

    Matches Thomas's compute_initial_fitness: one sim per pop1 individual,
    then cheap cost evals against each pop2 sigma.
    """
    pop1_idx, params, sigma_list, eval_seed = args
    dist = _worker_simulate(params, eval_seed)
    results = []
    for pop2_idx, sigma in sigma_list:
        cost = _worker_cost(dist, sigma)
        results.append((pop2_idx, cost))
    return (pop1_idx, results)


@dataclass
class FDDCConfig:
    """
    Configuration for FDDC optimizer.

    Attributes:
        population_size: Number of individuals per population (default: 50)
        max_generations: Maximum generations to run (default: 20)
        memory_size: Fitness memory length for each individual (default: 10)
        n_encounters: Number of pop1-pop2 encounters per generation (default: 10)
        n_children: Number of offspring per generation (default: 2)
        mutation_rate: Probability of mutation per gene (default: 0.6)
        mutation_strength: Mutation strength as fraction (default: 0.001)
        crossover_type: Crossover method ('two_point', 'uniform') (default: 'two_point')
        n_workers: Number of parallel workers (default: 6)
                   Set to 1 for serial execution, None for CPU count
        enable_fddc: Use FDDC novelty ranking (default: True)
                     If False, becomes regular co-evolution
        sigma_points_to_distribute: Number of sigma indices to modify (default: auto = 20% of length)
        sigma_points_per_index: Weight increase per modified sigma (default: 4)
        rank_selection_power: Power for rank-based selection (default: 1.5)
                             Higher = stronger selection pressure
    """
    population_size: int = 50
    max_generations: int = 20
    memory_size: int = 10
    n_encounters: int = 10
    n_children: int = 2
    mutation_rate: float = 0.6
    mutation_strength: float = 0.001
    crossover_type: str = 'two_point'
    n_workers: Optional[int] = None  # None = auto (cpu_count - 1)
    enable_fddc: bool = True
    sigma_length: Optional[int] = None  # None = auto (from experimental data)
    sigma_points_to_distribute: Optional[int] = None
    sigma_points_per_index: int = 4
    rank_selection_power: float = 1.5


@dataclass
class OptimizationResult:
    """Result from FDDC optimization."""
    best_params: np.ndarray
    best_cost: float
    generation: int
    cost_history: List[float]
    convergence_generation: Optional[int] = None


class FDDCOptimizer:
    """
    FDDC: Fitness-Diversity Driven Co-evolution optimizer.

    Example:
        >>> from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
        >>> from polymer_growth.core import SimulationParams
        >>> from polymer_growth.objective import MinMaxV2ObjectiveFunction
        >>>
        >>> # Setup
        >>> bounds = np.array([[lower1, upper1], [lower2, upper2], ...])
        >>> objective = MinMaxV2ObjectiveFunction(experimental_data)
        >>>
        >>> # Optimize
        >>> config = FDDCConfig(max_generations=20)
        >>> optimizer = FDDCOptimizer(bounds, objective, config=config)
        >>> result = optimizer.optimize(seed=42)
        >>> print(f"Best cost: {result.best_cost}")
    """

    def __init__(
        self,
        bounds: np.ndarray,
        objective_function: Callable,
        config: Optional[FDDCConfig] = None,
        callback: Optional[Callable[[int, float], None]] = None,
        console_callback: Optional[Callable[[str], None]] = None,
        simulate_fn: Optional[Callable] = None,
        cost_fn: Optional[Callable] = None
    ):
        """
        Initialize FDDC optimizer.

        Args:
            bounds: Parameter bounds (N x 2 array: [[lower, upper], ...])
            objective_function: Function that takes params array, returns cost
            config: FDDC configuration (uses defaults if None)
            callback: Optional callback(generation, best_cost) for progress updates
            console_callback: Optional callback(message) for console output
            simulate_fn: Optional function(params, eval_seed) -> distribution.
                         If provided with cost_fn, reproduce_pop1 simulates once
                         and evaluates cost with multiple sigmas (matching Thomas).
            cost_fn: Optional function(distribution, sigma) -> float cost.
        """
        self.bounds = bounds
        self.n_params = bounds.shape[0]
        self.objective = objective_function
        self.config = config if config is not None else FDDCConfig()
        self.callback = callback
        self.console_callback = console_callback
        self.simulate_fn = simulate_fn
        self.cost_fn = cost_fn
        self._sim_hist_fn = None  # Set externally for megabatch mode
        self._cost_from_hist_fn = None  # Set externally for megabatch mode
        self._hist_length = 0  # Set during init from experimental data

        # Validate config
        if self.config.population_size % self.config.memory_size != 0:
            raise ValueError("population_size must be divisible by memory_size")

        # Will be initialized in optimize()
        self.rng = None
        self.pop1 = None  # Parameter population
        self.pop2 = None  # Sigma weight population
        self.fitness_memory_pop1 = None
        self.fitness_memory_pop2 = None
        self.rank_probabilities = None
        self.cost_history = []
        self._last_best_dist = None  # Cache for best-cost eval
        self._megabatch_best_cost = None  # Cache for megabatch best eval

    def _parallel_map(self, func, tasks):
        """Run tasks in parallel using ProcessPool, or sequential if 1 worker."""
        if self._pool:
            return list(self._pool.map(func, tasks))
        return [func(t) for t in tasks]

    def _log(self, message: str):
        """Emit message to console callback and print to terminal."""
        print(message)
        if self.console_callback:
            self.console_callback(message)

    def optimize(self, seed: int = 42) -> OptimizationResult:
        """
        Run FDDC optimization.

        Args:
            seed: Random seed for reproducibility

        Returns:
            OptimizationResult with best parameters and cost history
        """
        # Initialize RNG
        self.rng = np.random.default_rng(seed)

        # Pre-warm Numba JIT before forking so children inherit compiled code
        try:
            from polymer_growth.core.simulation import _compute_vampiric_success_prob
            _dummy = np.array([1.0, 2.0])
            _compute_vampiric_success_prob(_dummy, _dummy, 0.5, 0.5, 0.5, 0.5)
        except Exception:
            pass

        # Create reusable pool for parallel evaluations
        global _worker_objective, _worker_simulate, _worker_cost
        global _worker_sim_hist_fn, _worker_cost_from_hist
        _worker_objective = self.objective
        _worker_simulate = self.simulate_fn
        _worker_cost = self.cost_fn
        _worker_sim_hist_fn = self._sim_hist_fn
        _worker_cost_from_hist = self._cost_from_hist_fn

        n_workers = self.config.n_workers if self.config.n_workers else None
        self._use_parallel = n_workers is None or n_workers != 1

        # Always use ProcessPool for true parallelism (no GIL).
        # fork() after Qt init is safe because workers only run numpy/simulation
        # code -- they never touch Qt/Cocoa internals.
        if self._use_parallel:
            self._pool = Pool(processes=n_workers)
        else:
            self._pool = None

        # Initialize populations
        self._initialize_populations()

        # Compute rank probabilities (used for parent selection)
        self.rank_probabilities = self._compute_rank_probabilities()

        # Initial fitness evaluation
        self._evaluate_initial_fitness()

        # Main evolution loop
        for gen in range(self.config.max_generations):
            self._log(f"\n=== Generation {gen + 1}/{self.config.max_generations} ===")

            if self._sim_hist_fn and self._cost_from_hist_fn:
                # FASTEST PATH: histogram-based IPC, 1 pool.map per gen
                self._generation_megabatch(gen)
                best_cost = self._megabatch_best_cost
            elif self.simulate_fn and self.cost_fn:
                # FAST PATH: Distribution-based, 1 pool.map per gen
                if gen > 0:
                    self._run_encounters()
                rank_pop1, rank_pop2 = self._compute_ranks()
                self._reproduce_batch(rank_pop1, rank_pop2)
                rank_pop1, _ = self._compute_ranks()
                best_params = rank_pop1[-1]
                eval_seed = int(self.rng.integers(0, 2**63))
                dist = self.simulate_fn(best_params, eval_seed)
                best_cost = self.cost_fn(dist, None)
            else:
                if gen > 0:
                    self._run_encounters()
                rank_pop1, rank_pop2 = self._compute_ranks()
                self._reproduce_batch(rank_pop1, rank_pop2)
                rank_pop1, _ = self._compute_ranks()
                best_params = rank_pop1[-1]
                eval_seed = int(self.rng.integers(0, 2**63))
                best_cost = self.objective(best_params, eval_seed=eval_seed)

            self.cost_history.append(best_cost)
            self._log(f"Best cost: {best_cost:.6f}")

            # Callback
            if self.callback:
                self.callback(gen + 1, best_cost)

        # Shutdown process pool
        if self._pool:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

        # Final result - use best cost from history (not re-evaluation)
        best_cost_in_history = min(self.cost_history)
        best_generation = self.cost_history.index(best_cost_in_history)

        rank_pop1, _ = self._compute_ranks()
        best_params = rank_pop1[-1]

        return OptimizationResult(
            best_params=best_params,
            best_cost=best_cost_in_history,  # Use minimum from history
            generation=self.config.max_generations,
            cost_history=self.cost_history,
            convergence_generation=best_generation + 1  # +1 for 1-indexed display
        )

    def _initialize_populations(self):
        """Initialize both populations randomly."""
        pop_size = self.config.population_size

        # Population 1: Parameter vectors
        self.pop1 = []
        for _ in range(pop_size):
            individual = self.rng.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.n_params
            )
            self.pop1.append(individual)

        # Population 2: Sigma weights
        # Run objective once (needed for sigma length detection if not set)
        _ = self.objective(self.pop1[0],
                           eval_seed=int(self.rng.integers(0, 2**63)))

        # Sigma length: use config value, or fall back to 100
        base_sigma_length = self.config.sigma_length or 100

        # Auto-compute sigma_points_to_distribute if not set
        if self.config.sigma_points_to_distribute is None:
            self.config.sigma_points_to_distribute = max(1, base_sigma_length // 5)

        self.pop2 = []
        for _ in range(pop_size):
            # Start with all 1s
            sigma = np.ones(base_sigma_length)

            # Randomly increase some indices
            n_modify = self.config.sigma_points_to_distribute
            modify_indices = self.rng.choice(
                base_sigma_length,
                size=n_modify,
                replace=False
            )
            sigma[modify_indices] += self.config.sigma_points_per_index

            self.pop2.append(sigma)

        # Initialize fitness memories
        self.fitness_memory_pop1 = [[] for _ in range(pop_size)]
        self.fitness_memory_pop2 = [[] for _ in range(pop_size)]

    def _compute_rank_probabilities(self) -> np.ndarray:
        """Compute selection probabilities for each rank."""
        n = self.config.population_size
        power = self.config.rank_selection_power
        probs = np.array([(i + 1) / n for i in range(n)]) ** power
        return probs

    def _evaluate_initial_fitness(self):
        """Evaluate initial fitness for all individuals."""
        pop_size = self.config.population_size
        mem_size = self.config.memory_size
        n_workers = self.config.n_workers

        # Determine if we should use parallel evaluation
        use_parallel = n_workers is None or n_workers > 1
        total_evals = pop_size * mem_size

        if use_parallel:
            self._evaluate_initial_fitness_parallel()
        else:
            self._evaluate_initial_fitness_sequential()

    def _evaluate_initial_fitness_sequential(self):
        """Sequential initial eval: simulate once per individual, cost N times."""
        pop_size = self.config.population_size
        mem_size = self.config.memory_size
        can_cache = self.simulate_fn and self.cost_fn

        self._log(f"Evaluating initial population ({pop_size} individuals, sequential)...")

        for i in range(pop_size):
            pop1_ind = self.pop1[i]
            start_idx = (i // mem_size) * mem_size
            pop2_indices = range(start_idx, start_idx + mem_size)
            eval_seed = int(self.rng.integers(0, 2**63))

            if can_cache:
                # Thomas's pattern: simulate once, cost eval N times
                dist = self.simulate_fn(pop1_ind, eval_seed)
                for pop2_idx in pop2_indices:
                    cost = self.cost_fn(dist, self.pop2[pop2_idx])
                    self.fitness_memory_pop1[i].append(-cost)
                    self.fitness_memory_pop2[pop2_idx].append(cost)
            else:
                for pop2_idx in pop2_indices:
                    es = int(self.rng.integers(0, 2**63))
                    cost = self.objective(pop1_ind, sigma=self.pop2[pop2_idx],
                                          eval_seed=es)
                    self.fitness_memory_pop1[i].append(-cost)
                    self.fitness_memory_pop2[pop2_idx].append(cost)

            if (i + 1) % max(1, pop_size // 10) == 0 or (i + 1) == pop_size:
                self._log(f"Progress: {i + 1}/{pop_size}")

    def _evaluate_initial_fitness_parallel(self):
        """Parallel initial eval: simulate once per individual, cost N times.

        Sends pop_size tasks to workers (not pop_size * mem_size).
        Each worker simulates one pop1 individual, then evaluates cost
        against its mem_size pop2 opponents. Matches Thomas's
        compute_initial_fitness (program code/fddc.py:324-333).
        """
        pop_size = self.config.population_size
        mem_size = self.config.memory_size
        n_workers = self.config.n_workers if self.config.n_workers else None
        can_cache = self.simulate_fn and self.cost_fn

        if can_cache:
            # Build 1 task per pop1 individual (100 tasks, not 1000)
            eval_tasks = []
            for i in range(pop_size):
                start_idx = (i // mem_size) * mem_size
                sigma_list = [(pop2_idx, self.pop2[pop2_idx])
                              for pop2_idx in range(start_idx, start_idx + mem_size)]
                eval_seed = int(self.rng.integers(0, 2**63))
                eval_tasks.append((i, self.pop1[i], sigma_list, eval_seed))

            self._log(f"Evaluating initial population ({pop_size} sims, "
                      f"{pop_size * mem_size} cost evals, {n_workers or 'auto'} workers)...")
            start_time = time.time()

            results = self._parallel_map(_worker_eval_initial_cached, eval_tasks)

            for pop1_idx, cost_pairs in results:
                for pop2_idx, cost in cost_pairs:
                    self.fitness_memory_pop1[pop1_idx].append(-cost)
                    self.fitness_memory_pop2[pop2_idx].append(cost)

                completed = pop1_idx + 1
                if completed % max(1, pop_size // 10) == 0 or completed == pop_size:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    self._log(f"Progress: {completed}/{pop_size} ({rate:.1f} individuals/sec)")
        else:
            # Fallback: full objective per eval (old behavior)
            eval_tasks = []
            for i in range(pop_size):
                start_idx = (i // mem_size) * mem_size
                for pop2_idx in range(start_idx, start_idx + mem_size):
                    eval_seed = int(self.rng.integers(0, 2**63))
                    eval_tasks.append((i, pop2_idx, self.pop1[i],
                                       self.pop2[pop2_idx], eval_seed))

            total_evals = len(eval_tasks)
            self._log(f"Evaluating initial population ({total_evals} evaluations, "
                      f"{n_workers or 'auto'} workers)...")
            start_time = time.time()

            results = self._parallel_map(_worker_eval_indexed, eval_tasks)

            for pop1_idx, pop2_idx, cost in results:
                self.fitness_memory_pop1[pop1_idx].append(-cost)
                self.fitness_memory_pop2[pop2_idx].append(cost)

            elapsed = time.time() - start_time
            self._log(f"Done: {total_evals} evals in {elapsed:.1f}s")

    def _generation_megabatch(self, gen: int):
        """Entire generation in ONE pool.map. Minimal IPC.

        Every task returns either a float (~8 bytes) or a histogram (~4KB).
        No Distribution objects cross process boundaries (~560KB each).
        33 tasks, 1 pool.map call, ~33 x avg 2KB = ~66KB total IPC vs ~18MB before.
        """
        n_enc = self.config.n_encounters
        n_children = self.config.n_children
        mem_size = self.config.memory_size
        pop_size = self.config.population_size
        hist_len = self._hist_length

        rank_pop1, rank_pop2 = self._compute_ranks()

        # Build ONE task list for everything
        tasks = []  # Each: (params_tuple, sigma_or_None, seed, hist_len)
        task_roles = []  # Track what each task is for

        # --- Encounters: return cost float ---
        enc_pop1_indices = []
        enc_pop2_indices = []
        enc_pop2_sigmas = []
        if gen > 0:
            enc_pop1 = self._rank_select(rank_pop1, n_enc)
            enc_pop2 = self._rank_select(rank_pop2, n_enc)
            for i in range(n_enc):
                enc_pop1_indices.append(
                    next(idx for idx, ind in enumerate(self.pop1)
                         if np.array_equal(ind, enc_pop1[i])))
                enc_pop2_indices.append(
                    next(idx for idx, ind in enumerate(self.pop2)
                         if np.array_equal(ind, enc_pop2[i])))
                seed = int(self.rng.integers(0, 2**63))
                params_tuple = tuple(enc_pop1[i].tolist())
                tasks.append((params_tuple, enc_pop2[i], seed, hist_len))
                task_roles.append(('enc', i))

        # --- Pop1 children: return histogram (need multi-sigma eval) ---
        pop1_children = []
        pop1_sigmas = []
        for c in range(n_children):
            parents = self._rank_select(rank_pop1, 2)
            child = self._mutate(self._crossover(parents[0], parents[1]))
            p2_idx = self.rng.integers(0, pop_size, size=mem_size)
            pop1_children.append(child)
            pop1_sigmas.append([self.pop2[i] for i in p2_idx])
            seed = int(self.rng.integers(0, 2**63))
            params_tuple = tuple(child.tolist())
            tasks.append((params_tuple, None, seed, hist_len))
            task_roles.append(('pop1', c))

        # --- Pop2 opponent evals: return cost float ---
        pop2_children = []
        pop2_task_ranges = []  # (start_idx, count) per child
        for c in range(n_children):
            parents = self._rank_select(rank_pop2, 2)
            child = self._crossover_sigma(parents[0], parents[1])
            pop2_children.append(child)
            p1_idx = self.rng.integers(0, pop_size, size=mem_size)
            opponents = [self.pop1[i] for i in p1_idx]
            start = len(tasks)
            for opp in opponents:
                seed = int(self.rng.integers(0, 2**63))
                params_tuple = tuple(opp.tolist())
                tasks.append((params_tuple, child, seed, hist_len))
                task_roles.append(('pop2', c))
            pop2_task_ranges.append((start, mem_size))

        # --- Best eval: return cost float (pass sigma=0 as "use default" flag) ---
        best_params = rank_pop1[-1]
        best_seed = int(self.rng.integers(0, 2**63))
        best_idx = len(tasks)
        tasks.append((tuple(best_params.tolist()), np.float64(0), best_seed, hist_len))
        task_roles.append(('best', 0))

        # ---- ONE pool.map ----
        results = self._parallel_map(_worker_megabatch, tasks)

        # ---- Process results ----

        # Encounters
        enc_count = 0
        for i, (role, idx) in enumerate(task_roles):
            if role == 'enc':
                _, cost = results[i]
                self.fitness_memory_pop1[enc_pop1_indices[enc_count]].append(-cost)
                self.fitness_memory_pop1[enc_pop1_indices[enc_count]].pop(0)
                self.fitness_memory_pop2[enc_pop2_indices[enc_count]].append(cost)
                self.fitness_memory_pop2[enc_pop2_indices[enc_count]].pop(0)
                enc_count += 1

        # Pop1 children (got histograms back, eval cost with multiple sigmas)
        pop1_idx = 0
        for i, (role, idx) in enumerate(task_roles):
            if role == 'pop1':
                _, hist = results[i]
                child = pop1_children[idx]
                sigmas = pop1_sigmas[idx]
                child_fitness = [
                    -self._cost_from_hist_fn(hist, s) for s in sigmas]
                worst = np.argmin(
                    [np.mean(m) for m in self.fitness_memory_pop1])
                if np.mean(child_fitness) > np.mean(
                        self.fitness_memory_pop1[worst]):
                    self.pop1[worst] = child
                    self.fitness_memory_pop1[worst] = child_fitness

        # Pop2 children
        for c_idx in range(n_children):
            start, count = pop2_task_ranges[c_idx]
            child = pop2_children[c_idx]
            child_fitness = [results[start + j][1] for j in range(count)]
            worst = np.argmin(
                [np.mean(m) for m in self.fitness_memory_pop2])
            if np.mean(child_fitness) > np.mean(
                    self.fitness_memory_pop2[worst]):
                self.pop2[worst] = child
                self.fitness_memory_pop2[worst] = child_fitness

        # Best cost
        self._megabatch_best_cost = results[best_idx][1]

    def _run_encounters(self):
        """Run random encounters between pop1 and pop2 (parallelized)."""
        n_enc = self.config.n_encounters

        # Select random individuals from each population
        rank_pop1, rank_pop2 = self._compute_ranks()

        selected_pop1 = self._rank_select(rank_pop1, n_enc)
        selected_pop2 = self._rank_select(rank_pop2, n_enc)

        # Resolve indices before parallel execution
        pop1_indices = []
        pop2_indices = []
        for i in range(n_enc):
            pop1_idx = next(idx for idx, ind in enumerate(self.pop1)
                           if np.array_equal(ind, selected_pop1[i]))
            pop2_idx = next(idx for idx, ind in enumerate(self.pop2)
                           if np.array_equal(ind, selected_pop2[i]))
            pop1_indices.append(pop1_idx)
            pop2_indices.append(pop2_idx)

        # Parallel evaluation of all encounters
        eval_seeds = [int(self.rng.integers(0, 2**63)) for _ in range(n_enc)]
        if self._use_parallel and n_enc > 1:
            tasks = [(selected_pop1[i], selected_pop2[i], eval_seeds[i])
                     for i in range(n_enc)]
            costs = self._parallel_map(_worker_eval, tasks)
        else:
            costs = [self.objective(selected_pop1[i], sigma=selected_pop2[i],
                                    eval_seed=eval_seeds[i])
                     for i in range(n_enc)]

        # Update fitness memories with results
        for i in range(n_enc):
            self.fitness_memory_pop1[pop1_indices[i]].append(-costs[i])
            self.fitness_memory_pop1[pop1_indices[i]].pop(0)
            self.fitness_memory_pop2[pop2_indices[i]].append(costs[i])
            self.fitness_memory_pop2[pop2_indices[i]].pop(0)

    def _compute_ranks(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute ranks for both populations.

        Returns:
            (rank_pop1, rank_pop2) - lists sorted from worst to best
        """
        # Rank pop1 by average fitness
        avg_fitness_pop1 = [np.mean(mem) for mem in self.fitness_memory_pop1]
        sorted_indices = np.argsort(avg_fitness_pop1)
        rank_pop1 = [self.pop1[i] for i in sorted_indices]

        # Rank pop2
        if self.config.enable_fddc:
            # FDDC: rank by novelty
            rank_pop2 = self._rank_by_novelty()
        else:
            # Regular: rank by fitness
            avg_fitness_pop2 = [np.mean(mem) for mem in self.fitness_memory_pop2]
            sorted_indices = np.argsort(avg_fitness_pop2)
            rank_pop2 = [self.pop2[i] for i in sorted_indices]

        return rank_pop1, rank_pop2

    def _rank_by_novelty(self) -> List[np.ndarray]:
        """Rank population 2 by novelty (FDDC component)."""
        # First rank by fitness
        avg_fitness = [np.mean(mem) for mem in self.fitness_memory_pop2]
        fitness_sorted_indices = np.argsort(avg_fitness)
        fitness_ranks = [self.pop2[i] for i in fitness_sorted_indices]

        # Compute novelty for each individual
        novelty_scores = []
        for i in range(len(fitness_ranks)):
            if i == 0:
                # First individual: compare to next
                novelty = abs(avg_fitness[fitness_sorted_indices[i]] -
                            avg_fitness[fitness_sorted_indices[i + 1]])
            elif i == len(fitness_ranks) - 1:
                # Last individual: compare to previous
                novelty = abs(avg_fitness[fitness_sorted_indices[i]] -
                            avg_fitness[fitness_sorted_indices[i - 1]])
            else:
                # Middle: minimum distance to neighbors
                dist_prev = abs(avg_fitness[fitness_sorted_indices[i]] -
                              avg_fitness[fitness_sorted_indices[i - 1]])
                dist_next = abs(avg_fitness[fitness_sorted_indices[i]] -
                              avg_fitness[fitness_sorted_indices[i + 1]])
                novelty = min(dist_prev, dist_next)

            novelty_scores.append(novelty)

        # Rank by novelty
        novelty_sorted_indices = np.argsort(novelty_scores)
        novelty_ranks = [fitness_ranks[i] for i in novelty_sorted_indices]

        return novelty_ranks

    def _rank_select(self, ranked_population: List[np.ndarray], n: int) -> List[np.ndarray]:
        """Select n individuals using rank-based selection."""
        selected = []
        for _ in range(n):
            r = self.rng.uniform(0, np.sum(self.rank_probabilities))
            cumsum = 0
            for i, prob in enumerate(self.rank_probabilities):
                cumsum += prob
                if cumsum >= r:
                    selected.append(ranked_population[i])
                    break
        return selected

    def _reproduce_batch(self, rank_pop1: List[np.ndarray],
                         rank_pop2: List[np.ndarray]):
        """Batched reproduction: all children generated, then evaluated in parallel.

        Pop1 children: simulate ALL in one parallel batch (was serial).
        Pop2 children: evaluate ALL in one parallel batch (was 2 separate batches).
        This eliminates the serial bottleneck in reproduce_pop1.
        """
        n_children = self.config.n_children
        mem_size = self.config.memory_size
        pop_size = self.config.population_size

        # --- Pop1: generate all children, batch simulate ---
        pop1_children = []
        pop1_sigmas = []  # list of sigma-lists per child
        for _ in range(n_children):
            parents = self._rank_select(rank_pop1, 2)
            child = self._mutate(self._crossover(parents[0], parents[1]))
            pop2_indices = self.rng.integers(0, pop_size, size=mem_size)
            sigmas = [self.pop2[i] for i in pop2_indices]
            pop1_children.append(child)
            pop1_sigmas.append(sigmas)

        if self.simulate_fn and self.cost_fn:
            # Batch simulate all pop1 children in parallel
            eval_seeds = [int(self.rng.integers(0, 2**63))
                          for _ in range(n_children)]
            if self._use_parallel and n_children > 1:
                sim_tasks = [(child, seed) for child, seed in
                             zip(pop1_children, eval_seeds)]
                dists = self._parallel_map(_worker_sim, sim_tasks)
            else:
                dists = [self.simulate_fn(c, s)
                         for c, s in zip(pop1_children, eval_seeds)]

            # Cost eval is cheap -- do it in main process
            for i, (child, dist, sigmas) in enumerate(
                    zip(pop1_children, dists, pop1_sigmas)):
                child_fitness = [-self.cost_fn(dist, s) for s in sigmas]
                worst_idx = np.argmin(
                    [np.mean(m) for m in self.fitness_memory_pop1])
                if np.mean(child_fitness) > np.mean(
                        self.fitness_memory_pop1[worst_idx]):
                    self.pop1[worst_idx] = child
                    self.fitness_memory_pop1[worst_idx] = child_fitness

            # Cache last child's distribution for best-cost eval
            self._last_best_dist = dists[-1] if dists else None
        else:
            # Fallback: original sequential behavior
            for i in range(n_children):
                child = pop1_children[i]
                sigmas = pop1_sigmas[i]
                eval_seeds = [int(self.rng.integers(0, 2**63))
                              for _ in range(len(sigmas))]
                if self._use_parallel and mem_size > 1:
                    tasks = [(child, s, es)
                             for s, es in zip(sigmas, eval_seeds)]
                    child_fitness = [-c for c in self._parallel_map(
                        _worker_eval, tasks)]
                else:
                    child_fitness = [
                        -self.objective(child, sigma=s, eval_seed=es)
                        for s, es in zip(sigmas, eval_seeds)]
                worst_idx = np.argmin(
                    [np.mean(m) for m in self.fitness_memory_pop1])
                if np.mean(child_fitness) > np.mean(
                        self.fitness_memory_pop1[worst_idx]):
                    self.pop1[worst_idx] = child
                    self.fitness_memory_pop1[worst_idx] = child_fitness
            self._last_best_dist = None

        # --- Pop2: generate all children, batch evaluate ---
        all_pop2_tasks = []
        pop2_children = []
        pop2_opponents_info = []  # (child_idx, opponent_list, eval_seeds)

        for c_idx in range(n_children):
            parents = self._rank_select(rank_pop2, 2)
            child = self._crossover_sigma(parents[0], parents[1])
            pop2_children.append(child)

            pop1_indices = self.rng.integers(0, pop_size, size=mem_size)
            opponents = [self.pop1[i] for i in pop1_indices]
            seeds = [int(self.rng.integers(0, 2**63)) for _ in range(mem_size)]
            pop2_opponents_info.append((opponents, seeds))

            for opp, es in zip(opponents, seeds):
                all_pop2_tasks.append((c_idx, opp, child, es))

        if self._use_parallel and len(all_pop2_tasks) > 1:
            worker_tasks = [(t[1], t[2], t[3]) for t in all_pop2_tasks]
            all_costs = self._parallel_map(_worker_eval, worker_tasks)
        else:
            all_costs = [self.objective(t[1], sigma=t[2], eval_seed=t[3])
                         for t in all_pop2_tasks]

        # Distribute costs back to children
        cost_idx = 0
        for c_idx in range(n_children):
            child = pop2_children[c_idx]
            child_fitness = all_costs[cost_idx:cost_idx + mem_size]
            cost_idx += mem_size

            worst_idx = np.argmin(
                [np.mean(m) for m in self.fitness_memory_pop2])
            if np.mean(child_fitness) > np.mean(
                    self.fitness_memory_pop2[worst_idx]):
                self.pop2[worst_idx] = child
                self.fitness_memory_pop2[worst_idx] = child_fitness

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Two-point crossover for parameter vectors."""
        child = parent1.copy()
        pt1 = self.rng.integers(0, len(parent1))
        pt2 = self.rng.integers(0, len(parent1))

        if pt1 > pt2:
            pt1, pt2 = pt2, pt1

        child[pt1:pt2] = parent2[pt1:pt2]
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate parameter vector."""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if self.rng.random() < self.config.mutation_rate:
                # Small perturbation or large jump
                if self.rng.random() < 0.5:
                    # Small: += strength * value
                    delta = mutated[i] * self.config.mutation_strength
                    mutated[i] += delta if self.rng.random() < 0.5 else -delta
                else:
                    # Large: sample uniform
                    mutated[i] = self.rng.uniform(self.bounds[i, 0], self.bounds[i, 1])

            # Clip to bounds
            mutated[i] = np.clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1])

        return mutated

    def _crossover_sigma(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Crossover for sigma weights (special logic)."""
        # Start with base sigma (all 1s)
        child = np.ones_like(parent1)

        # Find modified positions in both parents
        modified_p1 = np.where(parent1 > 1)[0]
        modified_p2 = np.where(parent2 > 1)[0]

        # Combine and sample
        all_modified = np.concatenate([modified_p1, modified_p2])

        if len(all_modified) > 0:
            # Randomly select positions to modify
            n_select = self.config.sigma_points_to_distribute
            selected = self.rng.choice(all_modified, size=min(n_select, len(all_modified)), replace=False)

            child[selected] += self.config.sigma_points_per_index

        return child