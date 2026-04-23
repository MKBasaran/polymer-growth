#!/usr/bin/env python3
"""
Speed benchmark: scaling curve for OUR implementation.
Runs 5k no BB with pop=50, 10 gens at different worker counts.
3 seeds each for mean + std.

Run NOW for tomorrow's presentation (~2-3 hours).

Usage:
    cd scripts/experiments
    /usr/local/bin/python3 speed_benchmark.py
"""

import sys
import time
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polymer_growth.core.simulation import SimulationParams, simulate
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig

DATA_DIR = PROJECT_ROOT / "program code" / "Data"
OUTPUT_DIR = PROJECT_ROOT / "validation_results"

BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])

FDDC_KWARGS = dict(
    memory_size=10, n_encounters=10, n_children=2,
    mutation_rate=0.6, mutation_strength=0.001,
    crossover_type='two_point', enable_fddc=True,
    sigma_points_per_index=4, rank_selection_power=1.5,
)

POP_SIZE = 50
N_GENS = 10
DATASET = "5k no BB.xlsx"
SEEDS = [42, 123, 777]


def make_objective(exp_values):
    objective = MinMaxV2ObjectiveFunction(exp_values)
    def wrapper(params_array, sigma=None):
        p = np.asarray(params_array).flatten().tolist()
        sp = SimulationParams(
            time_sim=int(p[0]), number_of_molecules=int(p[1]),
            monomer_pool=int(p[2]), p_growth=p[3], p_death=p[4],
            p_dead_react=p[5], l_exponent=p[6], d_exponent=p[7],
            l_naked=p[8], kill_spawns_new=bool(round(p[9]))
        )
        dist = simulate(sp, np.random.default_rng())
        return objective.compute_cost(dist, sigma=sigma)
    return wrapper


def run_single(n_workers, seed, exp_values):
    obj_fn = make_objective(exp_values)
    config = FDDCConfig(
        population_size=POP_SIZE,
        max_generations=N_GENS,
        n_workers=n_workers,
        **FDDC_KWARGS,
    )
    start = time.time()
    optimizer = FDDCOptimizer(
        bounds=BOUNDS, objective_function=obj_fn, config=config,
    )
    result = optimizer.optimize(seed=seed)
    elapsed = time.time() - start
    return {"elapsed_sec": elapsed, "best_cost": float(result.best_cost)}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    _, exp_values = load_experimental_data(str(DATA_DIR / DATASET))

    cpu_count = os.cpu_count() or 14
    worker_counts = [1, 2, 4, 6, 8]
    if cpu_count - 1 not in worker_counts:
        worker_counts.append(cpu_count - 1)
    worker_counts.sort()

    print("=" * 60)
    print("SPEED BENCHMARK: Scaling Curve")
    print(f"Dataset: {DATASET} | Pop: {POP_SIZE} | Gens: {N_GENS}")
    print(f"Worker counts: {worker_counts}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} reps each)")
    print(f"Total runs: {len(worker_counts) * len(SEEDS)}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    all_results = []

    for wi, nw in enumerate(worker_counts):
        times = []
        costs = []
        print(f"\n[{wi+1}/{len(worker_counts)}] {nw} workers")

        for si, seed in enumerate(SEEDS):
            print(f"  Seed {seed}...", end=" ", flush=True)
            r = run_single(nw, seed, exp_values)
            times.append(r["elapsed_sec"])
            costs.append(r["best_cost"])
            print(f"{r['elapsed_sec']:.1f}s")

        mean_t = np.mean(times)
        std_t = np.std(times)
        all_results.append({
            "workers": nw,
            "times": times,
            "mean_time": float(mean_t),
            "std_time": float(std_t),
            "mean_cost": float(np.mean(costs)),
        })
        print(f"  -> {mean_t:.1f}s +/- {std_t:.1f}s")

    # Compute speedups relative to 1 worker
    baseline = all_results[0]["mean_time"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Workers':>8} {'Mean Time':>10} {'Std':>8} {'Speedup':>8} {'Efficiency':>10}")
    print("-" * 50)

    for r in all_results:
        speedup = baseline / r["mean_time"]
        efficiency = speedup / r["workers"] * 100
        r["speedup"] = float(speedup)
        r["efficiency_pct"] = float(efficiency)
        print(f"{r['workers']:>8} {r['mean_time']:>9.1f}s {r['std_time']:>7.1f}s "
              f"{speedup:>7.2f}x {efficiency:>8.1f}%")

    print(f"\nIdeal scaling would be {worker_counts[-1]:.0f}x at {worker_counts[-1]} workers.")
    print(f"Actual: {all_results[-1]['speedup']:.2f}x")
    print(f"Parallel efficiency: {all_results[-1]['efficiency_pct']:.1f}%")

    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset": DATASET,
        "pop_size": POP_SIZE,
        "n_gens": N_GENS,
        "seeds": SEEDS,
        "cpu_count": cpu_count,
        "results": all_results,
    }
    out_path = OUTPUT_DIR / "speed_benchmark.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()