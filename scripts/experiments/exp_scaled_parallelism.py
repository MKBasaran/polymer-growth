#!/usr/bin/env python3
"""Test: scale FDDC parameters to utilize all 13 workers.

Thomas used n_encounters=10, n_children=2 because he was compute-constrained.
The original Paredis paper used 20 encounters. We test scaling these up
to fill 13 workers and measure whether we get faster convergence.

Compares:
  A) Thomas config: n_encounters=10, n_children=2 (33 sims/gen)
  B) Scaled config:  n_encounters=26, n_children=6 (93 sims/gen)

Both at 13 workers, same seed, 42 gens.
If B converges to the same cost in fewer gens, that's wall-clock savings.
If B finds better cost in the same gens, that's quality improvement.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp_scaled_parallelism.py
"""
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from shared import (BOUNDS, THOMAS_TABLE_VIII, FDDC_KWARGS,
                    make_objective_new, make_simulate_new, make_cost_fn)

DATASET = "5k"
POP_SIZE = 100
GENS = 42
SEED = 42
WORKERS = max(1, (os.cpu_count() or 4) - 1)

CONFIGS = {
    "thomas_default": {
        "n_encounters": 10,
        "n_children": 2,
        "label": f"Thomas default (10 enc, 2 children, {WORKERS}w)",
    },
    "scaled": {
        "n_encounters": 26,
        "n_children": 6,
        "label": f"Scaled (26 enc, 6 children, {WORKERS}w)",
    },
}


def run_config(config_name, overrides):
    ref = THOMAS_TABLE_VIII[DATASET]
    data_path = str(PROJECT_ROOT / "program code" / "Data" / ref["file"])
    _, exp_values = load_experimental_data(data_path)

    obj_fn = make_objective_new(exp_values)
    sim_fn = make_simulate_new()
    cost_fn = make_cost_fn(exp_values)

    kwargs = dict(FDDC_KWARGS)
    kwargs["n_encounters"] = overrides["n_encounters"]
    kwargs["n_children"] = overrides["n_children"]

    config = FDDCConfig(
        population_size=POP_SIZE,
        max_generations=GENS,
        n_workers=WORKERS,
        sigma_length=int(np.count_nonzero(exp_values)),
        **kwargs,
    )

    cost_history = []
    gen_times = []

    def progress_cb(gen, cost):
        cost_history.append(float(cost))
        now = time.time()
        if gen_times:
            dt = now - gen_times[-1]
            print(f"  Gen {gen:3d}/{GENS} | Cost: {cost:.4f} | {dt:.1f}s/gen")
        else:
            print(f"  Gen {gen:3d}/{GENS} | Cost: {cost:.4f}")
        gen_times.append(now)

    optimizer = FDDCOptimizer(
        bounds=BOUNDS, objective_function=obj_fn, config=config,
        callback=progress_cb, simulate_fn=sim_fn, cost_fn=cost_fn,
    )

    print(f"\n{'='*60}")
    print(f"  {overrides['label']}")
    print(f"  n_encounters={overrides['n_encounters']}, "
          f"n_children={overrides['n_children']}, workers={WORKERS}")
    print(f"  Sims/gen: {overrides['n_encounters']} + {overrides['n_children']} + "
          f"{overrides['n_children']}*{kwargs['memory_size']} + 1 = "
          f"{overrides['n_encounters'] + overrides['n_children'] + overrides['n_children'] * kwargs['memory_size'] + 1}")
    print(f"{'='*60}")

    start = time.time()
    result = optimizer.optimize(seed=SEED)
    elapsed = time.time() - start

    return {
        "config_name": config_name,
        "n_encounters": overrides["n_encounters"],
        "n_children": overrides["n_children"],
        "workers": WORKERS,
        "gens": GENS,
        "pop_size": POP_SIZE,
        "seed": SEED,
        "best_cost": float(result.best_cost),
        "cost_history": cost_history,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "sims_per_gen": (overrides["n_encounters"] + overrides["n_children"]
                         + overrides["n_children"] * kwargs["memory_size"] + 1),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("SCALED PARALLELISM TEST")
    print(f"Hardware: {os.cpu_count()} cores, {WORKERS} workers")
    print(f"Config: {DATASET} | pop={POP_SIZE} | gens={GENS} | seed={SEED}")
    print("=" * 60)

    results = {}
    for name, cfg in CONFIGS.items():
        out_file = OUTPUT_DIR / f"exp_scaled_{name}.json"
        if out_file.exists():
            print(f"\nLoading: {out_file.name}")
            with open(out_file) as f:
                results[name] = json.load(f)
            continue

        r = run_config(name, cfg)
        results[name] = r
        with open(out_file, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {out_file.name}")

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Time':>8} {'Best Cost':>10} {'Sims/gen':>10}")
    print("-" * 62)
    for name, r in results.items():
        label = CONFIGS[name]["label"] if name in CONFIGS else name
        print(f"{label:<30} {r['elapsed_min']:>6.1f}m {r['best_cost']:>10.4f} "
              f"{r['sims_per_gen']:>10}")

    if "thomas_default" in results and "scaled" in results:
        t_def = results["thomas_default"]
        t_scl = results["scaled"]
        speedup = t_def["elapsed_sec"] / t_scl["elapsed_sec"]
        cost_diff = (t_def["best_cost"] - t_scl["best_cost"]) / t_def["best_cost"] * 100
        print(f"\nScaled vs Default: {speedup:.2f}x time, {cost_diff:+.1f}% cost")
