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
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from multiprocessing.pool import Pool
import time

# Force fork start method for true parallelism (workers inherit globals)
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass  # Already set

# Detect if running inside a Qt application (fork after Qt init crashes on macOS)
def _is_qt_running():
    try:
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() is not None
    except ImportError:
        return False


# Module-level objective function reference for multiprocessing workers.
# Workers forked from the main process inherit this via fork().
_worker_objective = None


def _worker_eval(args):
    """Evaluate a single (params, sigma) pair in a worker process."""
    params, sigma = args
    return _worker_objective(params, sigma=sigma)


def _worker_eval_indexed(args):
    """Evaluate with index tracking for initial population."""
    pop1_idx, pop2_idx, params, sigma = args
    cost = _worker_objective(params, sigma=sigma)
    return (pop1_idx, pop2_idx, cost)


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
        console_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize FDDC optimizer.

        Args:
            bounds: Parameter bounds (N x 2 array: [[lower, upper], ...])
            objective_function: Function that takes params array, returns cost
            config: FDDC configuration (uses defaults if None)
            callback: Optional callback(generation, best_cost) for progress updates
            console_callback: Optional callback(message) for console output
        """
        self.bounds = bounds
        self.n_params = bounds.shape[0]
        self.objective = objective_function
        self.config = config if config is not None else FDDCConfig()
        self.callback = callback
        self.console_callback = console_callback

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

    def _parallel_map(self, func, tasks):
        """Run tasks in parallel using whichever pool is available."""
        if self._use_process_pool and self._pool:
            return list(self._pool.map(func, tasks))
        elif self._thread_pool:
            return list(self._thread_pool.map(func, tasks))
        else:
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
        global _worker_objective
        _worker_objective = self.objective

        n_workers = self.config.n_workers if self.config.n_workers else None
        self._use_parallel = n_workers is None or n_workers != 1

        # Use ProcessPool for CLI (true parallelism, no GIL).
        # Fall back to ThreadPool inside Qt GUI (fork after Qt crashes on macOS).
        self._use_process_pool = self._use_parallel and not _is_qt_running()
        if self._use_parallel:
            if self._use_process_pool:
                self._pool = Pool(processes=n_workers)
            else:
                self._pool = None
                self._thread_pool = ThreadPoolExecutor(max_workers=n_workers)
        else:
            self._pool = None
            self._thread_pool = None

        # Initialize populations
        self._initialize_populations()

        # Compute rank probabilities (used for parent selection)
        self.rank_probabilities = self._compute_rank_probabilities()

        # Initial fitness evaluation
        self._evaluate_initial_fitness()

        # Main evolution loop
        for gen in range(self.config.max_generations):
            self._log(f"\n=== Generation {gen + 1}/{self.config.max_generations} ===")

            # Co-evolution encounters (except first generation)
            if gen > 0:
                self._run_encounters()

            # Reproduction
            for _ in range(self.config.n_children):
                rank_pop1, rank_pop2 = self._compute_ranks()
                self._reproduce_pop1(rank_pop1)
                self._reproduce_pop2(rank_pop2)

            # Get best individual
            rank_pop1, _ = self._compute_ranks()
            best_params = rank_pop1[-1]  # Highest ranked
            best_cost = self.objective(best_params)

            self.cost_history.append(best_cost)
            self._log(f"Best cost: {best_cost:.6f}")

            # Callback
            if self.callback:
                self.callback(gen + 1, best_cost)

        # Shutdown pools
        if self._pool:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        if hasattr(self, '_thread_pool') and self._thread_pool:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None

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
        # First, run objective once to get sigma length
        _ = self.objective(self.pop1[0])

        # Determine sigma length (will be set by objective function)
        # For now, use default length based on experimental distribution
        # This will be overridden by actual objective function
        base_sigma_length = 100  # Placeholder, will be dynamic

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
        """Sequential evaluation of initial population (original behavior)."""
        self._log("Evaluating initial population (sequential)...")

        pop_size = self.config.population_size
        mem_size = self.config.memory_size

        for i in range(pop_size):
            pop1_ind = self.pop1[i]

            # Select pop2 opponents
            start_idx = (i // mem_size) * mem_size
            pop2_indices = range(start_idx, start_idx + mem_size)

            for pop2_idx in pop2_indices:
                pop2_ind = self.pop2[pop2_idx]
                cost = self.objective(pop1_ind, sigma=pop2_ind)

                self.fitness_memory_pop1[i].append(-cost)
                self.fitness_memory_pop2[pop2_idx].append(cost)

            if (i + 1) % max(1, pop_size // 10) == 0 or (i + 1) == pop_size:
                self._log(f"Progress: {i + 1}/{pop_size}")
            elif not self.console_callback:
                print(f"Progress: {i + 1}/{pop_size}", end='\r')

    def _evaluate_initial_fitness_parallel(self):
        """Parallel evaluation of initial population using ProcessPoolExecutor."""
        pop_size = self.config.population_size
        mem_size = self.config.memory_size
        n_workers = self.config.n_workers if self.config.n_workers else None

        # Build list of all evaluations needed
        eval_tasks = []
        for i in range(pop_size):
            start_idx = (i // mem_size) * mem_size
            for pop2_idx in range(start_idx, start_idx + mem_size):
                eval_tasks.append((i, pop2_idx, self.pop1[i], self.pop2[pop2_idx]))

        total_evals = len(eval_tasks)
        self._log(f"Evaluating initial population ({total_evals} evaluations, {n_workers or 'auto'} workers)...")

        start_time = time.time()
        completed = 0

        # Submit all tasks via parallel map
        results = self._parallel_map(_worker_eval_indexed, eval_tasks)

        for pop1_idx, pop2_idx, cost in results:
            self.fitness_memory_pop1[pop1_idx].append(-cost)
            self.fitness_memory_pop2[pop2_idx].append(cost)

            completed += 1
            if completed % max(1, total_evals // 10) == 0 or completed == total_evals:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                self._log(f"Progress: {completed}/{total_evals} ({rate:.1f} evals/sec)")

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
        if self._use_parallel and n_enc > 1:
            tasks = [(selected_pop1[i], selected_pop2[i]) for i in range(n_enc)]
            costs = self._parallel_map(_worker_eval, tasks)
        else:
            costs = [self.objective(selected_pop1[i], sigma=selected_pop2[i])
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

    def _reproduce_pop1(self, rank_pop1: List[np.ndarray]):
        """Reproduce population 1 (parameter vectors), parallelized child eval."""
        parents = self._rank_select(rank_pop1, 2)
        child = self._mutate(self._crossover(parents[0], parents[1]))

        # Pick random sigma opponents
        pop2_indices = self.rng.integers(0, self.config.population_size,
                                         size=self.config.memory_size)
        sigmas = [self.pop2[i] for i in pop2_indices]

        # Parallel evaluation of child against all sigmas
        if self._use_parallel and self.config.memory_size > 1:
            tasks = [(child, s) for s in sigmas]
            child_fitness = [-c for c in self._parallel_map(_worker_eval, tasks)]
        else:
            child_fitness = [-self.objective(child, sigma=s) for s in sigmas]

        # Replace worst if child is better
        worst_idx = np.argmin([np.mean(mem) for mem in self.fitness_memory_pop1])
        if np.mean(child_fitness) > np.mean(self.fitness_memory_pop1[worst_idx]):
            self.pop1[worst_idx] = child
            self.fitness_memory_pop1[worst_idx] = child_fitness

    def _reproduce_pop2(self, rank_pop2: List[np.ndarray]):
        """Reproduce population 2 (sigma weights), parallelized child eval."""
        parents = self._rank_select(rank_pop2, 2)
        child = self._crossover_sigma(parents[0], parents[1])

        # Pick random pop1 opponents
        pop1_indices = self.rng.integers(0, self.config.population_size,
                                         size=self.config.memory_size)
        opponents = [self.pop1[i] for i in pop1_indices]

        # Parallel evaluation of child sigma against all opponents
        if self._use_parallel and self.config.memory_size > 1:
            tasks = [(opp, child) for opp in opponents]
            child_fitness = self._parallel_map(_worker_eval, tasks)
        else:
            child_fitness = [self.objective(opp, sigma=child) for opp in opponents]

        # Replace worst if better
        worst_idx = np.argmin([np.mean(mem) for mem in self.fitness_memory_pop2])
        worst_fitness = np.mean(self.fitness_memory_pop2[worst_idx])
        child_avg = np.mean(child_fitness)

        if child_avg > worst_fitness:
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