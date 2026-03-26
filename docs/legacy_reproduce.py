"""
Phase 0: Legacy Code Reproducibility Script
============================================

This script demonstrates how to run the thesis FDDC algorithm with controlled seeds
for reproducibility. It serves as a bridge between the legacy code and the future
refactored package.

Run: python legacy_reproduce.py

Output: results/YYYYMMDD_HHMMSS/result.json
"""

import numpy as np
import random
import sys
import os
from datetime import datetime
import json

# Add program code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'program code'))

from fddc import fddc
from distributionComparison import min_maxV2
from simulation import polymer


def main():
    # Configuration
    SEED = 42
    MAX_GENERATIONS = 200
    CONVERGENCE_THRESHOLD = 10.0
    POPULATION_SIZE = 50
    TARGET_DATA = "sim_val0"

    print("=" * 70)
    print("POLYMER GROWTH PARAMETER FITTING - LEGACY REPRODUCIBILITY RUN")
    print("=" * 70)
    print(f"Target data: fakeData/{TARGET_DATA}")
    print(f"Optimizer: FDDC (Fitness-Diversity Driven Co-evolution)")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Max generations: {MAX_GENERATIONS}")
    print(f"Seed: {SEED}")
    print("=" * 70)
    print()

    # Set seeds for reproducibility
    print("Setting random seeds...")
    np.random.seed(SEED)
    random.seed(SEED)

    # Define parameter bounds (from thesis)
    bounds = np.array([
        [100, 3000],          # time_sim
        [10000, 120000],      # number_of_molecules
        [1000000, 5000000],   # monomer_pool
        [0.1, 0.99],          # p_growth
        [0.0001, 0.002],      # p_death
        [0.1, 0.9],           # p_dead_react
        [0.1, 0.9],           # l_exponent
        [0.1, 0.9],           # d_exponent
        [0.1, 1.0],           # l_naked
        [0, 1]                # kill_spawns_new (binary)
    ])

    print("Parameter bounds:")
    param_names = [
        "time_sim", "number_of_molecules", "monomer_pool",
        "p_growth", "p_death", "p_dead_react",
        "l_exponent", "d_exponent", "l_naked", "kill_spawns_new"
    ]
    for name, bound in zip(param_names, bounds):
        print(f"  {name:25s}: [{bound[0]:>10g}, {bound[1]:>10g}]")
    print()

    # Create fitness function
    print("Initializing cost function (min_maxV2)...")
    cost_func = min_maxV2(f"program code/fakeData/{TARGET_DATA}", polymer)

    # Create optimizer
    print("Initializing FDDC optimizer...")
    optimizer = fddc(
        bounds=bounds,
        fitnessFunction=cost_func.costFunction,
        distribution_comparison=cost_func,
        populationSize=POPULATION_SIZE,
        read_from_file=True,
        read_from_file_name=f"program code/fakeData/{TARGET_DATA}"
    )

    print("\n" + "=" * 70)
    print("STARTING OPTIMIZATION")
    print("=" * 70)
    print()

    # Run optimization
    start_time = datetime.now()
    converged = False

    for gen in range(MAX_GENERATIONS):
        gen_start = datetime.now()
        optimizer.run()
        gen_elapsed = (datetime.now() - gen_start).total_seconds()

        print(f"Generation {gen+1:3d}/{MAX_GENERATIONS} | "
              f"Best cost: {optimizer.best_score:10.4f} | "
              f"Time: {gen_elapsed:6.2f}s")

        if optimizer.best_score < CONVERGENCE_THRESHOLD:
            print(f"\n*** Converged at generation {gen+1} ***")
            converged = True
            break

    total_elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Total generations: {gen + 1}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Converged: {converged}")
    print()

    # Display best parameters
    print("Best parameters found:")
    for name, value in zip(param_names, optimizer.best):
        print(f"  {name:25s}: {value:15.6g}")
    print(f"\nBest cost: {optimizer.best_score:.6f}")
    print()

    # Create output directory
    output_dir = os.path.join("results", datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "seed": SEED,
            "target_data": TARGET_DATA,
            "optimizer": "FDDC",
            "population_size": POPULATION_SIZE,
            "converged": converged,
            "total_generations": gen + 1,
            "total_time_seconds": total_elapsed
        },
        "bounds": {name: [float(b[0]), float(b[1])] for name, b in zip(param_names, bounds)},
        "best_parameters": {name: float(value) for name, value in zip(param_names, optimizer.best)},
        "best_cost": float(optimizer.best_score)
    }

    result_file = os.path.join(output_dir, "result.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {result_file}")

    # Save a simple CSV for easy import
    csv_file = os.path.join(output_dir, "best_parameters.csv")
    with open(csv_file, "w") as f:
        f.write("parameter,value\n")
        for name, value in zip(param_names, optimizer.best):
            f.write(f"{name},{value}\n")
    print(f"Parameters saved to: {csv_file}")

    print("\n" + "=" * 70)
    print("To reproduce this run, use the same seed:", SEED)
    print("=" * 70)


if __name__ == "__main__":
    main()