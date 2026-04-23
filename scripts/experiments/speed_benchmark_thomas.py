#!/usr/bin/env python3
"""
Speed benchmark: Thomas's code vs ours on SAME machine.
Both at 6 workers (his config) + ours at full cores.
Same dataset, same pop, same gens, 3 seeds.

FOR THESIS (takes ~24+ hours due to Thomas's code being slow).

Usage:
    cd scripts/experiments
    /usr/local/bin/python3 speed_benchmark_thomas.py
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
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

from polymer_growth.core.simulation import SimulationParams, Distribution, simulate
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig

import simulation as thomas_sim

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


def make_new_objective(exp_values):
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


def make_thomas_objective(exp_values):
    objective = MinMaxV2ObjectiveFunction(exp_values)
    def wrapper(params_array, sigma=None):
        p = np.asarray(params_array).flatten().tolist()
        np.random.seed(None)
        living, dead, coupled = thomas_sim.polymer(
            time_sim=int(p[0]), number_of_molecules=int(p[1]),
            monomer_pool=int(p[2]), p_growth=p[3], p_death=p[4],
            p_dead_react=p[5], l_exponent=p[6], d_exponent=p[7],
            l_naked=p[8],
            kill_spawns_new=1 if bool(round(p[9])) else 0,
            video=0, coloured=0, final_plot=0
        )
        dist = Distribution(
            living=np.asarray(living, dtype=np.float64),
            dead=np.asarray(dead, dtype=np.float64),
            coupled=np.asarray(coupled, dtype=np.float64)
        )
        return objective.compute_cost(dist, sigma=sigma)
    return wrapper


def run_single(obj_fn, n_workers, seed):
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
    return {"elapsed_sec": time.time() - start, "best_cost": float(result.best_cost)}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    _, exp_values = load_experimental_data(str(DATA_DIR / DATASET))

    cpu_count = os.cpu_count() or 14

    configs = [
        {"name": "thomas_6w",           "impl": "thomas", "workers": 6},
        {"name": "ours_6w",             "impl": "new",    "workers": 6},
        {"name": f"ours_{cpu_count-1}w", "impl": "new",   "workers": cpu_count - 1},
    ]

    print("=" * 60)
    print("SPEED BENCHMARK: Thomas vs Ours (same machine)")
    print(f"Dataset: {DATASET} | Pop: {POP_SIZE} | Gens: {N_GENS}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    all_results = []

    for ci, cfg in enumerate(configs):
        times, costs = [], []
        print(f"\n[{ci+1}/{len(configs)}] {cfg['name']}")

        for si, seed in enumerate(SEEDS):
            print(f"  Seed {seed}...", end=" ", flush=True)
            if cfg["impl"] == "new":
                obj_fn = make_new_objective(exp_values)
            else:
                obj_fn = make_thomas_objective(exp_values)
            r = run_single(obj_fn, cfg["workers"], seed)
            times.append(r["elapsed_sec"])
            costs.append(r["best_cost"])
            print(f"{r['elapsed_sec']:.1f}s")

        mean_t = np.mean(times)
        all_results.append({
            "name": cfg["name"], "impl": cfg["impl"],
            "workers": cfg["workers"],
            "times": times, "mean_time": float(mean_t),
            "std_time": float(np.std(times)),
            "mean_cost": float(np.mean(costs)),
        })
        print(f"  -> {mean_t:.1f}s +/- {np.std(times):.1f}s")

    thomas_t = all_results[0]["mean_time"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Mean Time':>10} {'Speedup':>8}")
    print("-" * 40)
    for r in all_results:
        s = thomas_t / r["mean_time"]
        r["speedup"] = float(s)
        print(f"{r['name']:<20} {r['mean_time']:>9.1f}s {s:>7.2f}x")

    numba_effect = thomas_t / all_results[1]["mean_time"]
    total = thomas_t / all_results[2]["mean_time"]

    print(f"\nNumba JIT alone (6w vs 6w): {numba_effect:.2f}x faster")
    print(f"Total (thomas 6w vs ours {cpu_count-1}w): {total:.2f}x faster")

    out_path = OUTPUT_DIR / "speed_benchmark_thomas.json"
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "numba_speedup": float(numba_effect),
            "total_speedup": float(total),
        }, f, indent=2)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()