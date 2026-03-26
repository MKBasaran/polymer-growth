"""
MODEL COMPARISON: Thomas van den Broek 2020 Thesis Algorithms
==============================================================

This standalone script implements the different optimization models from the thesis:
1. Basic Genetic Algorithm (with various selection methods)
2. Island Model (distributed GA with migration)
3. FDDC (Fitness-Diversity Driven Co-evolution) - CURRENT IMPLEMENTATION

From thesis findings:
- Basic GA: ~77 generations average
- Island Model: ~41 generations average
- FDDC: ~20 generations average (BEST - what we use)

This is a STANDALONE prototype - does NOT modify production code.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional
from enum import Enum
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# =============================================================================
# SELECTION METHODS (from thesis section E)
# =============================================================================

class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    BOLTZMANN = "boltzmann"


def tournament_selection(population: List, fitness: List[float], n: int = 2) -> int:
    """
    Tournament selection: N individuals compete, best wins.
    Higher N = higher selection pressure.

    From thesis: "The more individuals that participate the higher the
    selection pressure becomes"
    """
    contestants = random.sample(range(len(population)), min(n, len(population)))
    best_idx = max(contestants, key=lambda i: fitness[i])
    return best_idx


def roulette_selection(population: List, fitness: List[float]) -> int:
    """
    Roulette wheel selection: probability proportional to fitness.

    From thesis equation (7): pi = fi / sum(F)
    """
    total = sum(fitness)
    if total == 0:
        return random.randint(0, len(population) - 1)

    probs = [f / total for f in fitness]
    return np.random.choice(len(population), p=probs)


def rank_selection(population: List, fitness: List[float]) -> int:
    """
    Rank selection: probability based on rank, not fitness magnitude.
    Low selection pressure - promotes exploration.

    From thesis equation (8): pi = Rank(i) / (N * (N-1))
    """
    n = len(population)
    # Sort indices by fitness (ascending - worst first)
    sorted_indices = sorted(range(n), key=lambda i: fitness[i])

    # Assign ranks (1 to N, where N is best)
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank

    # Compute selection probabilities
    total = n * (n + 1) / 2  # Sum of 1 to N
    probs = [r / total for r in ranks]
    return np.random.choice(n, p=probs)


def boltzmann_selection(population: List, fitness: List[float],
                        generation: int, max_generations: int = 50) -> int:
    """
    Boltzmann selection: temperature-based selection pressure.
    Low pressure early (exploration), high pressure late (exploitation).

    From thesis equations (9-11):
    pi = exp(-(max(f) - fi) / T)
    T = T0 * (1 - a)^k
    k = 1 + 100 * (g / G)
    """
    max_fitness = max(fitness)

    # Temperature schedule
    a = 0.05  # Cooling rate
    k = 1 + 100 * (generation / max_generations)
    T = 1.0 * ((1 - a) ** k)
    T = max(T, 0.001)  # Prevent division by zero

    # Compute selection probabilities
    probs = []
    for f in fitness:
        prob = np.exp(-(max_fitness - f) / T)
        probs.append(prob)

    total = sum(probs)
    probs = [p / total for p in probs]
    return np.random.choice(len(population), p=probs)


# =============================================================================
# MODEL 1: BASIC GENETIC ALGORITHM
# =============================================================================

@dataclass
class BasicGAConfig:
    population_size: int = 50
    max_generations: int = 200
    mutation_rate_slight: float = 0.6  # Slightly adjust parameter
    mutation_rate_random: float = 0.4  # Randomly regenerate
    selection_method: SelectionMethod = SelectionMethod.ROULETTE
    tournament_size: int = 2
    n_params: int = 10
    param_bounds: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.param_bounds is None:
            # Default bounds for polymer simulation parameters
            self.param_bounds = [
                (100, 3000),      # simulation_time
                (10000, 150000),  # n_molecules
                (1000000, 5000000),  # monomer_pool
                (0.1, 1.0),       # p_growth
                (0.0001, 0.01),   # p_death
                (0.1, 1.0),       # p_death_react
                (0.1, 1.0),       # living_exp
                (0.1, 1.0),       # death_exp
                (0.1, 1.0),       # living_naked
                (0, 1),           # death_spawns_new (binary)
            ]


class BasicGeneticAlgorithm:
    """
    Basic GA as described in thesis section III.

    Process:
    1. Initialize population
    2. Evaluate fitness
    3. Select parents (using selection method)
    4. Crossover + mutation to create children
    5. Replace worst individuals
    6. Repeat until solution found or max generations
    """

    def __init__(self, config: BasicGAConfig, objective_fn: Callable):
        self.config = config
        self.objective = objective_fn
        self.population = []
        self.fitness = []
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_individual = None

    def _initialize_population(self):
        """Create random initial population."""
        self.population = []
        for _ in range(self.config.population_size):
            individual = []
            for low, high in self.config.param_bounds:
                if low == 0 and high == 1 and isinstance(low, int):
                    # Binary parameter
                    individual.append(random.randint(0, 1))
                else:
                    individual.append(random.uniform(low, high))
            self.population.append(individual)

    def _evaluate_fitness(self):
        """Evaluate fitness for all individuals."""
        self.fitness = []
        for individual in self.population:
            # Objective returns cost, we want to maximize fitness
            cost = self.objective(individual)
            fitness = -cost  # Convert to maximization
            self.fitness.append(fitness)

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()

    def _select_parent(self) -> int:
        """Select parent using configured selection method."""
        method = self.config.selection_method

        if method == SelectionMethod.TOURNAMENT:
            return tournament_selection(
                self.population, self.fitness, self.config.tournament_size
            )
        elif method == SelectionMethod.ROULETTE:
            # Shift fitness to be positive for roulette
            min_fit = min(self.fitness)
            shifted = [f - min_fit + 1 for f in self.fitness]
            return roulette_selection(self.population, shifted)
        elif method == SelectionMethod.RANK:
            return rank_selection(self.population, self.fitness)
        elif method == SelectionMethod.BOLTZMANN:
            min_fit = min(self.fitness)
            shifted = [f - min_fit + 1 for f in self.fitness]
            return boltzmann_selection(
                self.population, shifted,
                self.generation, self.config.max_generations
            )

    def _crossover(self, parent1: List, parent2: List) -> List:
        """Two-point crossover as described in thesis."""
        n = len(parent1)
        if n < 3:
            return parent1.copy()

        # Select two crossover points
        pt1, pt2 = sorted(random.sample(range(n), 2))

        # Create child with segments from both parents
        child = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        return child

    def _mutate(self, individual: List) -> List:
        """
        Mutation as described in thesis:
        - 60% chance: slightly adjust parameter
        - 40% chance: randomly regenerate
        """
        mutated = individual.copy()

        for i, (low, high) in enumerate(self.config.param_bounds):
            if random.random() < 0.1:  # Mutation probability per gene
                if random.random() < self.config.mutation_rate_slight:
                    # Slight adjustment (10% of range)
                    delta = (high - low) * 0.1 * random.gauss(0, 1)
                    mutated[i] = np.clip(mutated[i] + delta, low, high)
                else:
                    # Random regeneration
                    if low == 0 and high == 1 and isinstance(low, int):
                        mutated[i] = random.randint(0, 1)
                    else:
                        mutated[i] = random.uniform(low, high)

        return mutated

    def run(self, target_cost: float = 10.0, verbose: bool = True) -> Tuple[List, float, int]:
        """
        Run the basic GA.

        Returns: (best_individual, best_cost, generations)
        """
        self._initialize_population()
        self._evaluate_fitness()

        for gen in range(self.config.max_generations):
            self.generation = gen

            # Check termination
            best_cost = -self.best_fitness
            if best_cost < target_cost:
                if verbose:
                    print(f"Solution found at generation {gen}!")
                return self.best_individual, best_cost, gen

            # Create new children (half the population)
            n_children = self.config.population_size // 2
            children = []

            for _ in range(n_children):
                # Select parents
                p1_idx = self._select_parent()
                p2_idx = self._select_parent()

                # Crossover and mutate
                child = self._crossover(
                    self.population[p1_idx],
                    self.population[p2_idx]
                )
                child = self._mutate(child)
                children.append(child)

            # Replace worst individuals
            sorted_indices = sorted(
                range(len(self.fitness)),
                key=lambda i: self.fitness[i]
            )

            for i, child in enumerate(children):
                worst_idx = sorted_indices[i]
                self.population[worst_idx] = child

            # Re-evaluate
            self._evaluate_fitness()

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best cost = {-self.best_fitness:.4f}")

        return self.best_individual, -self.best_fitness, self.config.max_generations


# =============================================================================
# MODEL 2: ISLAND MODEL
# =============================================================================

@dataclass
class IslandModelConfig:
    n_islands: int = 4
    population_per_island: int = 25
    migration_interval: int = 7
    migration_size: int = 2
    connection_type: str = "circular"  # circular, fully_connected, chain, star
    max_generations: int = 200
    # Each island uses a different selection method
    island_selection_methods: List[SelectionMethod] = None

    def __post_init__(self):
        if self.island_selection_methods is None:
            self.island_selection_methods = [
                SelectionMethod.ROULETTE,
                SelectionMethod.RANK,
                SelectionMethod.TOURNAMENT,
                SelectionMethod.BOLTZMANN,
            ][:self.n_islands]


class IslandModel:
    """
    Island Model as described in thesis section F.

    Multiple GAs run simultaneously with periodic migration.
    "Improvements had a tendency to take place 0-3 iterations after migration"

    Connection types determine migration patterns:
    - circular: A->B->C->D->A
    - fully_connected: All exchange with all
    - chain: A->B->C->D (no wrap)
    - star: All send to center
    """

    def __init__(self, config: IslandModelConfig, objective_fn: Callable,
                 param_bounds: List[Tuple[float, float]]):
        self.config = config
        self.objective = objective_fn
        self.param_bounds = param_bounds
        self.islands = []
        self.best_global_fitness = float('-inf')
        self.best_global_individual = None

    def _create_islands(self):
        """Create GA islands with different selection methods."""
        self.islands = []

        for i in range(self.config.n_islands):
            ga_config = BasicGAConfig(
                population_size=self.config.population_per_island,
                max_generations=self.config.max_generations,
                selection_method=self.config.island_selection_methods[i],
                param_bounds=self.param_bounds
            )
            ga = BasicGeneticAlgorithm(ga_config, self.objective)
            ga._initialize_population()
            ga._evaluate_fitness()
            self.islands.append(ga)

    def _get_migration_targets(self, island_idx: int) -> List[int]:
        """Get target islands for migration based on connection type."""
        n = self.config.n_islands

        if self.config.connection_type == "circular":
            return [(island_idx + 1) % n]
        elif self.config.connection_type == "fully_connected":
            return [i for i in range(n) if i != island_idx]
        elif self.config.connection_type == "chain":
            if island_idx < n - 1:
                return [island_idx + 1]
            return []
        elif self.config.connection_type == "star":
            if island_idx != 0:
                return [0]  # All send to center
            return []
        return []

    def _migrate(self):
        """Perform migration between islands."""
        # Collect migrants from each island (best individuals)
        migrants = {}

        for i, island in enumerate(self.islands):
            # Get best individuals
            sorted_indices = sorted(
                range(len(island.fitness)),
                key=lambda idx: island.fitness[idx],
                reverse=True
            )

            best_indices = sorted_indices[:self.config.migration_size]
            migrants[i] = [island.population[idx].copy() for idx in best_indices]

        # Send migrants to targets
        for src_idx, island_migrants in migrants.items():
            targets = self._get_migration_targets(src_idx)

            for target_idx in targets:
                target_island = self.islands[target_idx]

                # Replace worst individuals in target
                sorted_indices = sorted(
                    range(len(target_island.fitness)),
                    key=lambda i: target_island.fitness[i]
                )

                for i, migrant in enumerate(island_migrants):
                    if i < len(sorted_indices):
                        worst_idx = sorted_indices[i]
                        target_island.population[worst_idx] = migrant

    def run(self, target_cost: float = 10.0, verbose: bool = True) -> Tuple[List, float, int]:
        """Run the island model."""
        self._create_islands()

        for gen in range(self.config.max_generations):
            # Run one generation on each island
            for island in self.islands:
                island.generation = gen

                # Create children
                n_children = island.config.population_size // 2
                children = []

                for _ in range(n_children):
                    p1_idx = island._select_parent()
                    p2_idx = island._select_parent()
                    child = island._crossover(
                        island.population[p1_idx],
                        island.population[p2_idx]
                    )
                    child = island._mutate(child)
                    children.append(child)

                # Replace worst
                sorted_indices = sorted(
                    range(len(island.fitness)),
                    key=lambda i: island.fitness[i]
                )

                for i, child in enumerate(children):
                    island.population[sorted_indices[i]] = child

                island._evaluate_fitness()

                # Track global best
                if island.best_fitness > self.best_global_fitness:
                    self.best_global_fitness = island.best_fitness
                    self.best_global_individual = island.best_individual.copy()

            # Migrate at intervals
            if gen > 0 and gen % self.config.migration_interval == 0:
                self._migrate()
                if verbose:
                    print(f"Gen {gen}: Migration occurred")

            # Check termination
            best_cost = -self.best_global_fitness
            if best_cost < target_cost:
                if verbose:
                    print(f"Solution found at generation {gen}!")
                return self.best_global_individual, best_cost, gen

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best cost = {-self.best_global_fitness:.4f}")

        return self.best_global_individual, -self.best_global_fitness, self.config.max_generations


# =============================================================================
# MODEL 3: FDDC (What we currently use - reference implementation)
# =============================================================================

@dataclass
class FDDCConfig:
    """Configuration for Fitness-Diversity Driven Co-evolution."""
    population_size: int = 50
    memory_size: int = 10
    max_generations: int = 200
    n_encounters_per_gen: int = 10
    n_sigma_bins: int = 10
    param_bounds: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.param_bounds is None:
            self.param_bounds = [
                (100, 3000), (10000, 150000), (1000000, 5000000),
                (0.1, 1.0), (0.0001, 0.01), (0.1, 1.0),
                (0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0, 1),
            ]


class FDDCPrototype:
    """
    Fitness-Diversity Driven Co-evolution (from thesis section G)

    This is the BEST performing algorithm from the thesis!

    Key features:
    - Two co-evolving populations (solutions and problems/sigma)
    - Memory-based fitness (10 encounters per individual)
    - Novelty-based ranking for diversity
    - Prevents premature convergence

    From thesis: "By using a form of novelty search it actively seeks out
    individuals from the opposing population to challenge the solutions
    using problems that they have not encountered before"
    """

    def __init__(self, config: FDDCConfig, objective_fn: Callable):
        self.config = config
        self.objective = objective_fn

        # Solution population (parameters)
        self.solutions = []
        self.solution_memories = []  # List of fitness memories

        # Problem population (sigma weights)
        self.problems = []
        self.problem_memories = []

        self.best_solution = None
        self.best_cost = float('inf')

    def _initialize_populations(self):
        """Initialize both populations with random individuals."""
        n = self.config.population_size

        # Solutions (parameter values)
        self.solutions = []
        self.solution_memories = []
        for _ in range(n):
            individual = []
            for low, high in self.config.param_bounds:
                if low == 0 and high == 1 and isinstance(low, int):
                    individual.append(random.randint(0, 1))
                else:
                    individual.append(random.uniform(low, high))
            self.solutions.append(individual)
            self.solution_memories.append([])

        # Problems (sigma weights for each bin)
        self.problems = []
        self.problem_memories = []
        for _ in range(n):
            # Sigma values between 0 and 2 (1 is neutral)
            sigma = [random.uniform(0.5, 1.5) for _ in range(self.config.n_sigma_bins)]
            self.problems.append(sigma)
            self.problem_memories.append([])

    def _encounter(self, sol_idx: int, prob_idx: int) -> float:
        """
        Single encounter between solution and problem.
        Returns cost (solution wants low, problem wants high).
        """
        solution = self.solutions[sol_idx]
        sigma = self.problems[prob_idx]

        # Evaluate with sigma weighting
        cost = self.objective(solution, sigma=sigma)

        # Update memories (FIFO)
        mem_size = self.config.memory_size

        # Solution memory (stores negative cost as fitness)
        self.solution_memories[sol_idx].append(-cost)
        if len(self.solution_memories[sol_idx]) > mem_size:
            self.solution_memories[sol_idx].pop(0)

        # Problem memory (stores positive cost as fitness)
        self.problem_memories[prob_idx].append(cost)
        if len(self.problem_memories[prob_idx]) > mem_size:
            self.problem_memories[prob_idx].pop(0)

        # Track best
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = solution.copy()

        return cost

    def _get_fitness(self, memories: List[List[float]]) -> List[float]:
        """Average fitness from memory."""
        return [np.mean(mem) if mem else 0 for mem in memories]

    def _novelty_rank(self, fitness: List[float]) -> List[float]:
        """
        Compute novelty-based fitness (from thesis).

        New fitness = minimum distance to neighbors in fitness ranking.
        This promotes diversity by favoring "different" individuals.
        """
        n = len(fitness)
        if n < 3:
            return fitness

        # Sort by fitness
        sorted_indices = sorted(range(n), key=lambda i: fitness[i])

        # Compute novelty (distance to nearest neighbor in ranking)
        novelty = [0.0] * n
        for rank, idx in enumerate(sorted_indices):
            # Distance to neighbors
            distances = []
            if rank > 0:
                prev_idx = sorted_indices[rank - 1]
                distances.append(abs(fitness[idx] - fitness[prev_idx]))
            if rank < n - 1:
                next_idx = sorted_indices[rank + 1]
                distances.append(abs(fitness[idx] - fitness[next_idx]))

            novelty[idx] = min(distances) if distances else 0

        return novelty

    def _select_by_rank(self, fitness: List[float]) -> int:
        """Rank-based selection."""
        return rank_selection(list(range(len(fitness))), fitness)

    def run(self, target_cost: float = 10.0, verbose: bool = True) -> Tuple[List, float, int]:
        """Run FDDC optimization."""
        self._initialize_populations()

        # Initial encounters to fill memories
        if verbose:
            print("Initializing memories...")

        for sol_idx in range(self.config.population_size):
            for _ in range(self.config.memory_size):
                prob_idx = random.randint(0, self.config.population_size - 1)
                self._encounter(sol_idx, prob_idx)

        # Main loop
        for gen in range(self.config.max_generations):
            # Check termination
            if self.best_cost < target_cost:
                if verbose:
                    print(f"Solution found at generation {gen}!")
                return self.best_solution, self.best_cost, gen

            # Perform encounters
            sol_fitness = self._get_fitness(self.solution_memories)
            prob_fitness = self._get_fitness(self.problem_memories)

            # Apply novelty ranking to problem population (promotes diversity)
            prob_novelty = self._novelty_rank(prob_fitness)

            for _ in range(self.config.n_encounters_per_gen):
                sol_idx = self._select_by_rank(sol_fitness)
                prob_idx = self._select_by_rank(prob_novelty)
                self._encounter(sol_idx, prob_idx)

            # Create one child for each population
            # Solution child
            p1 = self._select_by_rank(sol_fitness)
            p2 = self._select_by_rank(sol_fitness)
            child = self._crossover(self.solutions[p1], self.solutions[p2])
            child = self._mutate_solution(child)

            # Evaluate child
            prob_idx = random.randint(0, len(self.problems) - 1)
            child_cost = self.objective(child, sigma=self.problems[prob_idx])

            # Replace worst if better
            worst_idx = min(range(len(sol_fitness)), key=lambda i: sol_fitness[i])
            if -child_cost > sol_fitness[worst_idx]:
                self.solutions[worst_idx] = child
                self.solution_memories[worst_idx] = [-child_cost]

            # Problem child (sigma weights)
            p1 = self._select_by_rank(prob_novelty)
            p2 = self._select_by_rank(prob_novelty)
            sigma_child = self._crossover_sigma(self.problems[p1], self.problems[p2])
            sigma_child = self._mutate_sigma(sigma_child)

            # Evaluate sigma child
            sol_idx = random.randint(0, len(self.solutions) - 1)
            sigma_cost = self.objective(self.solutions[sol_idx], sigma=sigma_child)

            # Replace worst problem if better (problems want HIGH cost)
            worst_prob_idx = min(range(len(prob_fitness)), key=lambda i: prob_fitness[i])
            if sigma_cost > prob_fitness[worst_prob_idx]:
                self.problems[worst_prob_idx] = sigma_child
                self.problem_memories[worst_prob_idx] = [sigma_cost]

            if verbose and gen % 25 == 0:
                # FDDC uses 25 gens = 1 "effective" generation (per thesis)
                print(f"Gen {gen}: Best cost = {self.best_cost:.4f}")

        return self.best_solution, self.best_cost, self.config.max_generations

    def _crossover(self, p1: List, p2: List) -> List:
        """Two-point crossover for solutions."""
        n = len(p1)
        if n < 3:
            return p1.copy()
        pt1, pt2 = sorted(random.sample(range(n), 2))
        return p1[:pt1] + p2[pt1:pt2] + p1[pt2:]

    def _crossover_sigma(self, p1: List, p2: List) -> List:
        """Crossover for sigma weights."""
        return [(a + b) / 2 for a, b in zip(p1, p2)]

    def _mutate_solution(self, individual: List) -> List:
        """Mutate solution parameters."""
        mutated = individual.copy()
        for i, (low, high) in enumerate(self.config.param_bounds):
            if random.random() < 0.1:
                if random.random() < 0.6:
                    delta = (high - low) * 0.1 * random.gauss(0, 1)
                    mutated[i] = np.clip(mutated[i] + delta, low, high)
                else:
                    mutated[i] = random.uniform(low, high)
        return mutated

    def _mutate_sigma(self, sigma: List) -> List:
        """Mutate sigma weights."""
        mutated = sigma.copy()
        for i in range(len(mutated)):
            if random.random() < 0.1:
                mutated[i] = np.clip(mutated[i] + random.gauss(0, 0.2), 0.1, 2.0)
        return mutated


# =============================================================================
# DEMO / COMPARISON
# =============================================================================

def dummy_objective(params, sigma=None):
    """
    Dummy objective for testing.
    In real use, this would run the polymer simulation.
    """
    # Simulate some cost based on distance from "target"
    target = [1000, 50000, 2500000, 0.5, 0.001, 0.5, 0.5, 0.5, 0.5, 1]

    cost = 0
    for i, (p, t) in enumerate(zip(params, target)):
        diff = abs(p - t) / max(abs(t), 1)
        weight = sigma[i % len(sigma)] if sigma else 1.0
        cost += diff * weight

    # Add some noise (stochastic simulation)
    cost += random.gauss(0, cost * 0.1)
    return max(0, cost)


def compare_models():
    """
    Compare all three models on dummy objective.

    Expected results (from thesis):
    - Basic GA: ~77 generations
    - Island Model: ~41 generations
    - FDDC: ~20 generations
    """
    print("=" * 60)
    print("MODEL COMPARISON: Thomas van den Broek 2020")
    print("=" * 60)

    results = {}
    n_runs = 3  # For demo, use more for statistical significance

    # 1. Basic GA with Roulette selection
    print("\n[1] Basic Genetic Algorithm (Roulette)")
    print("-" * 40)

    ga_gens = []
    for run in range(n_runs):
        config = BasicGAConfig(
            population_size=50,
            max_generations=200,
            selection_method=SelectionMethod.ROULETTE
        )
        ga = BasicGeneticAlgorithm(config, dummy_objective)
        _, cost, gens = ga.run(target_cost=0.5, verbose=False)
        ga_gens.append(gens)
        print(f"  Run {run+1}: {gens} generations, cost={cost:.4f}")

    results['Basic GA'] = np.mean(ga_gens)
    print(f"  Average: {results['Basic GA']:.1f} generations")

    # 2. Island Model
    print("\n[2] Island Model (4 islands, circular)")
    print("-" * 40)

    bounds = BasicGAConfig().param_bounds
    island_gens = []
    for run in range(n_runs):
        config = IslandModelConfig(
            n_islands=4,
            population_per_island=25,
            max_generations=200
        )
        island = IslandModel(config, dummy_objective, bounds)
        _, cost, gens = island.run(target_cost=0.5, verbose=False)
        island_gens.append(gens)
        print(f"  Run {run+1}: {gens} generations, cost={cost:.4f}")

    results['Island Model'] = np.mean(island_gens)
    print(f"  Average: {results['Island Model']:.1f} generations")

    # 3. FDDC
    print("\n[3] FDDC (Fitness-Diversity Driven Co-evolution)")
    print("-" * 40)

    fddc_gens = []
    for run in range(n_runs):
        config = FDDCConfig(
            population_size=50,
            max_generations=200
        )
        fddc = FDDCPrototype(config, dummy_objective)
        _, cost, gens = fddc.run(target_cost=0.5, verbose=False)
        # Normalize: 25 FDDC gens = 1 "effective" gen (per thesis)
        effective_gens = gens / 25
        fddc_gens.append(effective_gens)
        print(f"  Run {run+1}: {gens} raw / {effective_gens:.1f} effective generations, cost={cost:.4f}")

    results['FDDC'] = np.mean(fddc_gens)
    print(f"  Average: {results['FDDC']:.1f} effective generations")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Avg Generations':<20} {'Thesis Reference'}")
    print("-" * 60)
    print(f"{'Basic GA':<25} {results['Basic GA']:<20.1f} ~77")
    print(f"{'Island Model':<25} {results['Island Model']:<20.1f} ~41")
    print(f"{'FDDC':<25} {results['FDDC']:<20.1f} ~20")
    print()
    print("NOTE: FDDC is what we use in production - it's the BEST!")
    print("The other models are included for comparison/thesis documentation.")


if __name__ == "__main__":
    compare_models()
