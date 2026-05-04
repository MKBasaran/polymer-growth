"""Shared utilities for Thomas replication experiments."""

import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

from polymer_growth.core.simulation import (
    SimulationParams, Distribution, simulate, MONOMER_MASS, INITIATOR_MASS
)
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

THOMAS_TABLE_VIII = {
    "5k":  {"file": "5k no BB.xlsx",  "fddc_gens": 42,   "fddc_cost": 21.7358,
             "roulette_gens": 1801, "roulette_cost": 13.3815},
    "10k": {"file": "10k no BB.xlsx", "fddc_gens": 56,   "fddc_cost": 108.5352,
             "roulette_gens": 1217, "roulette_cost": 105.5283},
    "20k": {"file": "20k no BB.xlsx", "fddc_gens": 44,   "fddc_cost": 131.7344,
             "roulette_gens": 1192, "roulette_cost": 129.7423},
    "30k": {"file": "30k no BB.xlsx", "fddc_gens": 27,   "fddc_cost": 340.2589,
             "roulette_gens": 1159, "roulette_cost": 285.9435},
}


def _make_sim_params(params_array):
    p = np.asarray(params_array).flatten().tolist()
    return SimulationParams(
        time_sim=int(p[0]), number_of_molecules=int(p[1]),
        monomer_pool=int(p[2]), p_growth=p[3], p_death=p[4],
        p_dead_react=p[5], l_exponent=p[6], d_exponent=p[7],
        l_naked=p[8], kill_spawns_new=bool(round(p[9]))
    )


def make_objective_new(exp_values):
    objective = MinMaxV2ObjectiveFunction(exp_values)
    def wrapper(params_array, sigma=None, eval_seed=None):
        rng = np.random.default_rng(eval_seed)
        dist = simulate(_make_sim_params(params_array), rng)
        return objective.compute_cost(dist, sigma=sigma)
    return wrapper


def make_simulate_new():
    def sim_fn(params_array, eval_seed):
        rng = np.random.default_rng(eval_seed)
        return simulate(_make_sim_params(params_array), rng)
    return sim_fn


def make_objective_thomas(exp_values):
    objective = MinMaxV2ObjectiveFunction(exp_values)
    def wrapper(params_array, sigma=None, eval_seed=None):
        np.random.seed(eval_seed % (2**32) if eval_seed is not None else None)
        p = np.asarray(params_array).flatten().tolist()
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


def make_simulate_thomas():
    def sim_fn(params_array, eval_seed):
        np.random.seed(eval_seed % (2**32) if eval_seed is not None else None)
        p = np.asarray(params_array).flatten().tolist()
        living, dead, coupled = thomas_sim.polymer(
            time_sim=int(p[0]), number_of_molecules=int(p[1]),
            monomer_pool=int(p[2]), p_growth=p[3], p_death=p[4],
            p_dead_react=p[5], l_exponent=p[6], d_exponent=p[7],
            l_naked=p[8],
            kill_spawns_new=1 if bool(round(p[9])) else 0,
            video=0, coloured=0, final_plot=0
        )
        return Distribution(
            living=np.asarray(living, dtype=np.float64),
            dead=np.asarray(dead, dtype=np.float64),
            coupled=np.asarray(coupled, dtype=np.float64)
        )
    return sim_fn


def make_cost_fn(exp_values):
    objective = MinMaxV2ObjectiveFunction(exp_values)
    def cost_fn(dist, sigma=None):
        return objective.compute_cost(dist, sigma=sigma)
    return cost_fn


def run_fddc(dataset_key: str, gen_count: int, pop_size: int,
             impl: str, seed: int = 42, workers: int = None):
    """Run one FDDC optimization. Returns results dict."""
    ref = THOMAS_TABLE_VIII[dataset_key]
    data_path = str(DATA_DIR / ref["file"])
    _, exp_values = load_experimental_data(data_path)

    obj_fn = make_objective_new(exp_values) if impl == "new" else make_objective_thomas(exp_values)
    sim_fn = make_simulate_new() if impl == "new" else make_simulate_thomas()
    cost_fn = make_cost_fn(exp_values)

    import os
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)

    config = FDDCConfig(
        population_size=pop_size,
        max_generations=gen_count,
        n_workers=workers,
        sigma_length=int(np.count_nonzero(exp_values)),
        **FDDC_KWARGS,
    )

    cost_history = []
    gen_times = []

    def progress_cb(gen, cost):
        cost_history.append(float(cost))
        now = time.time()
        if gen_times:
            gen_time = now - gen_times[-1]
            remaining = (gen_count - gen) * gen_time
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  Gen {gen:4d}/{gen_count} | Cost: {cost:.4f} | "
                  f"{gen_time:.0f}s/gen | ETA: {eta.strftime('%H:%M')}")
        else:
            print(f"  Gen {gen:4d}/{gen_count} | Cost: {cost:.4f}")
        gen_times.append(now)

    def console_cb(msg):
        print(f"  {msg}")

    optimizer = FDDCOptimizer(
        bounds=BOUNDS, objective_function=obj_fn, config=config,
        callback=progress_cb, console_callback=console_cb,
        simulate_fn=sim_fn, cost_fn=cost_fn,
    )

    start = time.time()
    result = optimizer.optimize(seed=seed)
    elapsed = time.time() - start

    return {
        "dataset": dataset_key,
        "impl": impl,
        "pop": pop_size,
        "gens": gen_count,
        "seed": seed,
        "best_cost": float(result.best_cost),
        "best_from_history": float(min(cost_history)) if cost_history else float(result.best_cost),
        "cost_history": cost_history,
        "best_params": result.best_params.tolist(),
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "elapsed_hours": elapsed / 3600,
        "thomas_fddc_cost": ref["fddc_cost"],
        "thomas_roulette_cost": ref["roulette_cost"],
    }


def run_simulation_equivalence(n_runs: int = 30):
    """Quick sanity check: both simulations produce same distributions."""
    print("SIMULATION EQUIVALENCE TEST")
    print(f"Runs per param set: {n_runs}")

    param_sets = {
        "thomas_published": [1000, 100000, 32000000, 0.256761375,
                             0.0000806, 0.00494705224, 0.872555086,
                             0.406013255, 0.384144228, 1.0],
        "thesis_default":   [1000, 10000, 1000000, 0.72, 0.000084,
                             0.73, 0.41, 0.75, 0.24, 1.0],
    }

    all_pass = True
    for name, params in param_sets.items():
        new_mn, thomas_mn = [], []
        for i in range(n_runs):
            seed = 1000 + i
            sp = SimulationParams(*params[:9], kill_spawns_new=bool(round(params[9])))
            dist = simulate(sp, np.random.default_rng(seed))
            new_mn.append(float(np.mean(dist.all_chains() * MONOMER_MASS + INITIATOR_MASS)))

            np.random.seed(seed)
            l, d, c = thomas_sim.polymer(*params[:9],
                kill_spawns_new=1 if bool(round(params[9])) else 0,
                video=0, coloured=0, final_plot=0)
            thomas_mn.append(float(np.mean(np.concatenate([l, d, c]) * MONOMER_MASS + INITIATOR_MASS)))

        _, t_p = stats.ttest_ind(new_mn, thomas_mn, equal_var=False)
        _, ks_p = stats.ks_2samp(new_mn, thomas_mn)
        sig = t_p < 0.05 or ks_p < 0.05
        if sig: all_pass = False
        marker = "FAIL" if sig else "PASS"
        print(f"  {name}: t-p={t_p:.4f} ks-p={ks_p:.4f} -> {marker}")

    print(f"RESULT: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


def save_and_report(results: dict, output_file: str):
    """Save results JSON and print summary."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / output_file
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    r = results
    ref = THOMAS_TABLE_VIII[r["dataset"]]
    print(f"\n{'='*50}")
    print(f"RESULTS: {r['dataset']} no BB")
    print(f"{'='*50}")
    print(f"  Implementation: {r['impl']}")
    print(f"  Config: pop={r['pop']}, gens={r['gens']}, seed={r['seed']}")
    print(f"  Our best cost:       {r['best_cost']:.4f}")
    print(f"  Thomas FDDC cost:    {ref['fddc_cost']:.4f} ({ref['fddc_gens']} gens)")
    print(f"  Thomas Roulette cost:{ref['roulette_cost']:.4f} ({ref['roulette_gens']} gens)")
    print(f"  Wall-clock time:     {r['elapsed_hours']:.2f} hours")
    print(f"  Saved to: {path}")