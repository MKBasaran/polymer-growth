#!/usr/bin/env python3
"""Experiment C: Dataset generalization (IV = dataset).
Our code on [5k, 10k, 20k, 30k], 10 workers, Table VIII gen counts, 10 seeds.
Each run in a fresh subprocess -- zero shared state.

Skips any result file that already exists on disk.

Estimated time: ~2.5 hours (40 runs, gen counts vary)

Usage:
    python3 exp_C_datasets.py
"""
import json
import numpy as np
from scipy import stats
from _runner import run_ours, save_result, OUTPUT_DIR, TABLE_VIII

DATASETS = ["5k", "10k", "20k", "30k"]
SEEDS = [42, 123, 777, 2024, 9999, 101, 202, 303, 404, 505]

if __name__ == '__main__':
    print("=" * 60)
    print("EXPERIMENT C: DATASET GENERALIZATION (IV = dataset)")
    print("Fixed: our code, 10 workers, pop=100, Table VIII gen counts")
    print(f"Datasets: {DATASETS} | Seeds: {len(SEEDS)}")
    print("=" * 60)

    for ds in DATASETS:
        for seed in SEEDS:
            fname = f"expC_ours_{ds}_10w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip (exists): {fname}")
            else:
                r = run_ours(ds, workers=10, seed=seed)
                save_result(r, fname)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT C RESULTS")
    print(f"{'='*60}")
    print(f"{'Dataset':>8} {'Gens':>5} {'Mean Time':>10} {'Mean Cost':>11} "
          f"{'Cost 95% CI':>20} {'Table VIII':>11}")
    print("-" * 70)
    for ds in DATASETS:
        times, costs = [], []
        for seed in SEEDS:
            p = OUTPUT_DIR / f"expC_ours_{ds}_10w_seed{seed}.json"
            if p.exists():
                d = json.load(open(p))
                times.append(d["elapsed_min"])
                costs.append(d["best_cost"])
        if len(costs) >= 2:
            ref = TABLE_VIII[ds]
            ci = stats.t.interval(0.95, len(costs)-1, loc=np.mean(costs),
                                  scale=stats.sem(costs))
            print(f"{ds:>8} {ref['gens']:>5} {np.mean(times):>9.1f}m {np.mean(costs):>11.2f} "
                  f"[{ci[0]:>8.2f}, {ci[1]:>8.2f}] {ref['cost']:>11.4f}")
