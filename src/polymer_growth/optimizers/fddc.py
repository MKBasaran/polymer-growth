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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List
import multiprocessing as mp
from multiprocessing.pool import Pool
import time

import sys as _sys
_is_frozen = getattr(_sys, 'frozen', False)
if not _is_frozen and _sys.platform != 'win32':
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

_worker_objective = None
_worker_simulate = None
_worker_cost = None


def _worker_eval(args):
    params, sigma, eval_seed = args
    return _worker_objective(params, sigma=sigma, eval_seed=eval_seed)


def _worker_eval_indexed(args):
    pop1_idx, pop2_idx, params, sigma, eval_seed = args
    cost = _worker_objective(params, sigma=sigma, eval_seed=eval_seed)
    return (pop1_idx, pop2_idx, cost)


def _worker_eval_initial_cached(args):
    pop1_idx, params, sigma_list, eval_seed = args
    dist = _worker_simulate(params, eval_seed)
    results = []
    for pop2_idx, sigma in sigma_list:
        cost = _worker_cost(dist, sigma)
        results.append((pop2_idx, cost))
    return (pop1_idx, results)


@dataclass
class FDDCConfig:
    """Configuration for the FDDC optimiser.

    Holds GA hyperparameters (population size, generation cap, mutation rate,
    sigma weighting), the worker count for the multiprocessing pool, and the
    toggle that disables FDDC novelty ranking to fall back to a plain
    co-evolutionary GA.
    """

    population_size: int = 50
    max_generations: int = 20
    memory_size: int = 10
    n_encounters: int = 10
    n_children: int = 2
    mutation_rate: float = 0.6
    mutation_strength: float = 0.001
    crossover_type: str = 'two_point'
    n_workers: Optional[int] = None
    enable_fddc: bool = True
    sigma_length: Optional[int] = None
    sigma_points_to_distribute: Optional[int] = None
    sigma_points_per_index: int = 4
    rank_selection_power: float = 1.5


@dataclass
class OptimizationResult:
    """Result of an FDDC optimisation run.

    ``best_params`` is the lowest-cost parameter vector observed in the
    recorded history, ``best_cost`` is its cost, ``cost_history`` is the
    per-generation best cost, ``generation`` is the number of generations
    actually run, and ``convergence_generation`` is the index of the
    generation that produced ``best_cost``.
    """

    best_params: np.ndarray
    best_cost: float
    generation: int
    cost_history: List[float]
    convergence_generation: Optional[int] = None


class FDDCOptimizer:
    """Fitness-Diversity Driven Co-evolution genetic algorithm.

    Two interacting populations co-evolve: population 1 holds candidate
    parameter vectors, population 2 holds cost-function sigma weights.
    Population 2 is selected by novelty rather than fitness, which keeps
    the search varied. The optimiser dispatches evaluations to a process
    pool when ``n_workers != 1``.
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
        """Construct an optimiser.

        Args:
            bounds: ``(n_params, 2)`` array of lower/upper bounds for each
                parameter.
            objective_function: Callable ``(params, sigma, eval_seed) -> float``
                returning a scalar cost.
            config: Optional FDDC configuration. Defaults are used if omitted.
            callback: Optional ``(generation, best_cost)`` UI hook called once
                per generation.
            console_callback: Optional ``(message)`` UI hook for log lines.
            simulate_fn: Optional callable ``(params, eval_seed) -> Distribution``
                that, paired with ``cost_fn``, lets the optimiser avoid
                re-simulating for every sigma evaluation.
            cost_fn: Optional callable ``(distribution, sigma) -> float`` used
                with ``simulate_fn`` to score a cached distribution against
                multiple sigmas.
        """
        if not isinstance(bounds, np.ndarray) or bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be a numpy array with shape (N, 2)")
        if not callable(objective_function):
            raise TypeError("objective_function must be callable")

        self.bounds = bounds
        self.n_params = bounds.shape[0]
        self.objective = objective_function
        self.config = config if config is not None else FDDCConfig()
        self.callback = callback
        self.console_callback = console_callback
        self.simulate_fn = simulate_fn
        self.cost_fn = cost_fn

        if self.config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.config.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.config.memory_size < 1:
            raise ValueError("memory_size must be at least 1")
        if self.config.population_size % self.config.memory_size != 0:
            raise ValueError("population_size must be divisible by memory_size")

        self.rng = None
        self.pop1 = None
        self.pop2 = None
        self.fitness_memory_pop1 = None
        self.fitness_memory_pop2 = None
        self.rank_probabilities = None
        self.cost_history = []
        self.rank_pop1 = []
        self.rank_pop2 = []
        self.novelty_rank_pop2 = []
        self._prob_sum = 0.0

    def _parallel_map(self, func, tasks):
        if self._pool:
            return list(self._pool.map(func, tasks))
        return [func(t) for t in tasks]

    def _log(self, message: str):
        print(message)
        if self.console_callback:
            self.console_callback(message)

    def optimize(self, seed: int = 42) -> OptimizationResult:
        """Run FDDC for ``config.max_generations`` generations from ``seed``.

        The process pool, when used, is created on entry and torn down in a
        ``finally`` block. Determinism at fixed seed is verified by Experiment E.

        Args:
            seed: Seed for the ``numpy.random.Generator`` that drives every
                stochastic decision in the optimiser.

        Returns:
            OptimizationResult: best parameters, best cost, full cost history,
            number of generations run, and the generation index that produced
            the best cost.
        """
        self.rng = np.random.default_rng(seed)

        global _worker_objective, _worker_simulate, _worker_cost
        _worker_objective = self.objective
        _worker_simulate = self.simulate_fn
        _worker_cost = self.cost_fn

        n_workers = self.config.n_workers if self.config.n_workers else None
        self._use_parallel = n_workers is None or n_workers != 1

        if self._use_parallel:
            self._pool = Pool(processes=n_workers)
        else:
            self._pool = None

        self._initialize_populations()

        pop_size = self.config.population_size
        self.rank_probabilities = np.array(
            [((i + 1) / pop_size) ** self.config.rank_selection_power
             for i in range(pop_size)])
        self._prob_sum = float(np.sum(self.rank_probabilities))

        self._evaluate_initial_fitness()

        try:
            for gen in range(self.config.max_generations):
                self._log(f"\n=== Generation {gen + 1}/{self.config.max_generations} ===")

                # --- Thomas's run() structure exactly ---

                # Encounters: use ranks from PREVIOUS generation (stored on self)
                if gen > 0:
                    self._run_encounters()

                # For each child: rank, reproduce_pop1, reproduce_pop2
                for _ in range(self.config.n_children):
                    self._compute_ranks()
                    self._reproduce_pop1()
                    self._reproduce_pop2()

                # Final ranking (stored for next gen's encounters)
                self._compute_ranks()

                # Best eval
                best_params = self.rank_pop1[-1]
                eval_seed = int(self.rng.integers(0, 2**63))
                if self.simulate_fn and self.cost_fn:
                    dist = self.simulate_fn(best_params, eval_seed)
                    best_cost = self.cost_fn(dist, None)
                else:
                    best_cost = self.objective(best_params, eval_seed=eval_seed)

                self.cost_history.append(best_cost)
                self._log(f"Best cost: {best_cost:.6f}")

                if self.callback:
                    self.callback(gen + 1, best_cost)
        finally:
            if self._pool:
                self._pool.terminate()
                self._pool.join()
                self._pool = None

        best_cost_in_history = min(self.cost_history)
        best_generation = self.cost_history.index(best_cost_in_history)

        self._compute_ranks()
        best_params = self.rank_pop1[-1]

        return OptimizationResult(
            best_params=best_params,
            best_cost=best_cost_in_history,
            generation=self.config.max_generations,
            cost_history=self.cost_history,
            convergence_generation=best_generation + 1
        )

    def _initialize_populations(self):
        pop_size = self.config.population_size

        self.pop1 = []
        for _ in range(pop_size):
            individual = self.rng.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1],
                size=self.n_params
            )
            self.pop1.append(individual)

        _ = self.objective(self.pop1[0],
                           eval_seed=int(self.rng.integers(0, 2**63)))

        base_sigma_length = self.config.sigma_length or 100

        if self.config.sigma_points_to_distribute is None:
            self.config.sigma_points_to_distribute = max(1, base_sigma_length // 5)

        self.pop2 = []
        for _ in range(pop_size):
            sigma = np.ones(base_sigma_length)
            n_modify = self.config.sigma_points_to_distribute
            modify_indices = self.rng.choice(
                base_sigma_length,
                size=n_modify,
                replace=False
            )
            sigma[modify_indices] += self.config.sigma_points_per_index
            self.pop2.append(sigma)

        self.fitness_memory_pop1 = [[] for _ in range(pop_size)]
        self.fitness_memory_pop2 = [[] for _ in range(pop_size)]

    def _evaluate_initial_fitness(self):
        n_workers = self.config.n_workers
        use_parallel = n_workers is None or n_workers > 1

        if use_parallel:
            self._evaluate_initial_fitness_parallel()
        else:
            self._evaluate_initial_fitness_sequential()

    def _evaluate_initial_fitness_sequential(self):
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
        pop_size = self.config.population_size
        mem_size = self.config.memory_size
        n_workers = self.config.n_workers if self.config.n_workers else None
        can_cache = self.simulate_fn and self.cost_fn

        if can_cache:
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

    # ------------------------------------------------------------------ #
    # ENCOUNTERS — matches Thomas's run() lines 150-184
    # Uses self.rank_pop1 / self.rank_pop2 from the PREVIOUS generation's
    # final _compute_ranks() call — no extra ranking here.
    # ------------------------------------------------------------------ #

    def _run_encounters(self):
        n_enc = self.config.n_encounters

        selected_pop1 = self._rank_select(self.rank_pop1, n_enc)
        selected_pop2 = self._rank_select(self.rank_pop2, n_enc)

        pop1_id_map = {id(ind): idx for idx, ind in enumerate(self.pop1)}
        pop2_id_map = {id(ind): idx for idx, ind in enumerate(self.pop2)}
        pop1_indices = [pop1_id_map[id(selected_pop1[i])] for i in range(n_enc)]
        pop2_indices = [pop2_id_map[id(selected_pop2[i])] for i in range(n_enc)]

        eval_seeds = [int(self.rng.integers(0, 2**63)) for _ in range(n_enc)]
        if self._use_parallel and n_enc > 1:
            tasks = [(selected_pop1[i], selected_pop2[i], eval_seeds[i])
                     for i in range(n_enc)]
            costs = self._parallel_map(_worker_eval, tasks)
        else:
            costs = [self.objective(selected_pop1[i], sigma=selected_pop2[i],
                                    eval_seed=eval_seeds[i])
                     for i in range(n_enc)]

        for i in range(n_enc):
            self.fitness_memory_pop1[pop1_indices[i]].append(-costs[i])
            self.fitness_memory_pop1[pop1_indices[i]].pop(0)
            self.fitness_memory_pop2[pop2_indices[i]].append(costs[i])
            self.fitness_memory_pop2[pop2_indices[i]].pop(0)

    # ------------------------------------------------------------------ #
    # RANKING — stores results on self like Thomas does
    # ------------------------------------------------------------------ #

    def _compute_ranks(self):
        pop_size = self.config.population_size
        fm1 = self.fitness_memory_pop1
        fm2 = self.fitness_memory_pop2

        sums1 = [sum(fm1[i]) for i in range(pop_size)]
        sorted_idx = sorted(range(pop_size), key=lambda i: sums1[i])
        self.rank_pop1 = [self.pop1[i] for i in sorted_idx]

        sums2 = [sum(fm2[i]) for i in range(pop_size)]
        sorted_idx2 = sorted(range(pop_size), key=lambda i: sums2[i])
        self.rank_pop2 = [self.pop2[i] for i in sorted_idx2]

        if not self.config.enable_fddc:
            self.novelty_rank_pop2 = self.rank_pop2
            return

        sorted_sums2 = [sums2[i] for i in sorted_idx2]
        novelty_scores = []
        n = len(sorted_sums2)
        for i in range(n):
            if i == 0:
                v = abs(sorted_sums2[i] - sorted_sums2[i + 1])
            elif i == n - 1:
                v = abs(sorted_sums2[i] - sorted_sums2[i - 1])
            else:
                v1 = abs(sorted_sums2[i] - sorted_sums2[i + 1])
                v2 = abs(sorted_sums2[i] - sorted_sums2[i - 1])
                v = min(v1, v2)
            novelty_scores.append(v)

        novelty_sorted = sorted(range(n), key=lambda i: novelty_scores[i])
        self.novelty_rank_pop2 = [self.rank_pop2[i] for i in novelty_sorted]

    # ------------------------------------------------------------------ #
    # SELECTION — matches Thomas's rank-based selection
    # ------------------------------------------------------------------ #

    def _rank_select(self, ranked_population, n):
        selected = []
        probs = self.rank_probabilities
        prob_sum = self._prob_sum
        for _ in range(n):
            r = float(self.rng.uniform(0, prob_sum))
            cumsum = 0.0
            for i in range(len(probs)):
                cumsum += probs[i]
                if cumsum >= r:
                    selected.append(ranked_population[i])
                    break
        return selected

    # ------------------------------------------------------------------ #
    # REPRODUCE POP1 — matches Thomas's reproduce_pop1 (lines 336-393)
    # 1 sim in main process, 10 cheap cost evals
    # ------------------------------------------------------------------ #

    def _reproduce_pop1(self):
        parents = self._rank_select(self.rank_pop1, 2)
        child = self._mutate(self._crossover(parents[0], parents[1]))

        eval_seed = int(self.rng.integers(0, 2**63))

        if self.simulate_fn and self.cost_fn:
            dist = self.simulate_fn(child, eval_seed)
            fitnesses = []
            for _ in range(self.config.memory_size):
                r = int(self.rng.integers(0, self.config.population_size))
                cost = -self.cost_fn(dist, self.pop2[r])
                fitnesses.append(cost)
        else:
            eval_seeds = [int(self.rng.integers(0, 2**63))
                          for _ in range(self.config.memory_size)]
            sigmas = [self.pop2[int(self.rng.integers(0, self.config.population_size))]
                      for _ in range(self.config.memory_size)]
            if self._use_parallel and self.config.memory_size > 1:
                tasks = [(child, s, es) for s, es in zip(sigmas, eval_seeds)]
                fitnesses = [-c for c in self._parallel_map(_worker_eval, tasks)]
            else:
                fitnesses = [-self.objective(child, sigma=s, eval_seed=es)
                             for s, es in zip(sigmas, eval_seeds)]

        worst = self.rank_pop1[0]
        worst_idx = next(i for i in range(len(self.pop1)) if self.pop1[i] is worst)
        if sum(fitnesses) > sum(self.fitness_memory_pop1[worst_idx]):
            self.fitness_memory_pop1.pop(worst_idx)
            self.fitness_memory_pop1.append(fitnesses)
            self.pop1.pop(worst_idx)
            self.pop1.append(child)

    # ------------------------------------------------------------------ #
    # REPRODUCE POP2 — matches Thomas's reproduce_pop2 (lines 396-451)
    # p.map(compute_fitness_sigma, [child]*10)
    # ------------------------------------------------------------------ #

    def _reproduce_pop2(self):
        parents = self._rank_select(self.novelty_rank_pop2, 2)
        child = self._crossover_sigma(parents[0], parents[1])

        eval_seeds = [int(self.rng.integers(0, 2**63))
                      for _ in range(self.config.memory_size)]
        pop1_indices = [int(self.rng.integers(0, self.config.population_size))
                        for _ in range(self.config.memory_size)]

        if self._use_parallel and self.config.memory_size > 1:
            tasks = [(self.pop1[pop1_indices[i]], child, eval_seeds[i])
                     for i in range(self.config.memory_size)]
            fitnesses = list(self._parallel_map(_worker_eval, tasks))
        else:
            fitnesses = [self.objective(self.pop1[pop1_indices[i]], sigma=child,
                                        eval_seed=eval_seeds[i])
                         for i in range(self.config.memory_size)]

        worst = self.rank_pop2[0]
        worst_idx = next(i for i in range(len(self.pop2)) if self.pop2[i] is worst)
        if sum(fitnesses) > sum(self.fitness_memory_pop2[worst_idx]):
            self.fitness_memory_pop2.pop(worst_idx)
            self.fitness_memory_pop2.append(fitnesses)
            self.pop2.pop(worst_idx)
            self.pop2.append(child)

    # ------------------------------------------------------------------ #
    # CROSSOVER / MUTATION — unchanged
    # ------------------------------------------------------------------ #

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        pt1 = self.rng.integers(0, len(parent1))
        pt2 = self.rng.integers(0, len(parent1))

        child = np.empty_like(parent1)
        if pt1 < pt2:
            child[:pt1] = parent2[:pt1]
            child[pt1:pt2] = parent1[pt1:pt2]
            child[pt2:] = parent2[pt2:]
        else:
            child[:pt2] = parent1[:pt2]
            child[pt2:pt1] = parent2[pt2:pt1]
            child[pt1:] = parent1[pt1:]
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()

        for i in range(len(mutated) - 1):
            if self.rng.random() < self.config.mutation_rate:
                delta = mutated[i] * self.config.mutation_strength
                if self.rng.random() < 0.5:
                    mutated[i] += delta
                else:
                    mutated[i] -= delta
            else:
                mutated[i] = self.rng.uniform(self.bounds[i, 0], self.bounds[i, 1])

            mutated[i] = np.clip(mutated[i], self.bounds[i, 0], self.bounds[i, 1])

        return mutated

    def _crossover_sigma(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = np.ones_like(parent1)

        modified_p1 = set(np.where(parent1 > 1)[0].tolist())
        modified_p2 = set(np.where(parent2 > 1)[0].tolist())
        all_modified = list(modified_p1 | modified_p2)

        if len(all_modified) > 0:
            n_select = min(self.config.sigma_points_to_distribute,
                           len(all_modified))
            selected = self.rng.choice(all_modified, size=n_select,
                                       replace=False)
            child[selected] += self.config.sigma_points_per_index

        return child
