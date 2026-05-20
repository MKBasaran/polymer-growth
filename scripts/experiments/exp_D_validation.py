#!/usr/bin/env python3
"""Experiment D: Cost validation (IV = dataset).
Thomas's unmodified code on [5k, 10k, 20k, 30k], 6 workers, Table VIII gen counts.
10 runs per dataset (Thomas is non-deterministic).
Each run in a fresh subprocess -- zero shared state.

Skips any result file that already exists on disk.

Estimated time: ~3 hours (40 runs x ~5 min, minus already-completed)

Usage:
    python3 exp_D_validation.py
"""
import json
import numpy as np
from scipy import stats
from _runner import run_thomas, save_result, OUTPUT_DIR, TABLE_VIII

DATASETS = ["5k", "10k", "20k", "30k"]
RUNS = list(range(1, 11))

if __name__ == '__main__':
    print("=" * 60)
    print("EXPERIMENT D: COST VALIDATION (IV = dataset)")
    print("Thomas's code, Pool(6), pop=100, Table VIII gen counts")
    print(f"Datasets: {DATASETS} | Runs: {len(RUNS)} per dataset")
    print("=" * 60)

    for ds in DATASETS:
        for run in RUNS:
            fname = f"expD_thomas_{ds}_6w_run{run}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip (exists): {fname}")
            else:
                r = run_thomas(ds, workers=6)
                save_result(r, fname)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT D RESULTS")
    print(f"{'='*60}")
    print(f"{'Dataset':>8} {'n':>3} {'Mean Cost':>11} {'Cost 95% CI':>20} "
          f"{'Table VIII':>12} {'Pct diff':>9}")
    print("-" * 68)
    for ds in DATASETS:
        costs, times = [], []
        for run in RUNS:
            p = OUTPUT_DIR / f"expD_thomas_{ds}_6w_run{run}.json"
            if p.exists():
                d = json.load(open(p))
                costs.append(d["best_cost"])
                times.append(d["elapsed_min"])
        if len(costs) >= 2:
            ref = TABLE_VIII[ds]
            ci = stats.t.interval(0.95, len(costs)-1, loc=np.mean(costs),
                                  scale=stats.sem(costs))
            pct = (np.mean(costs) - ref['cost']) / ref['cost'] * 100
            print(f"{ds:>8} {len(costs):>3} {np.mean(costs):>11.2f} "
                  f"[{ci[0]:>8.2f}, {ci[1]:>8.2f}] "
                  f"{ref['cost']:>12.4f} {pct:>+8.1f}%")

    # Cross-validation with Exp C
    print(f"\nCross-validation (Exp C ours vs Exp D thomas):")
    print(f"{'Dataset':>8} {'Ours mean':>11} {'Thomas mean':>13} {'t-test p':>10} {'Verdict':>12}")
    print("-" * 58)
    for ds in DATASETS:
        ours_costs = []
        for seed in [42, 123, 777, 2024, 9999, 101, 202, 303, 404, 505]:
            p = OUTPUT_DIR / f"expC_ours_{ds}_10w_seed{seed}.json"
            if p.exists():
                ours_costs.append(json.load(open(p))["best_cost"])
        thomas_costs = []
        for run in RUNS:
            p = OUTPUT_DIR / f"expD_thomas_{ds}_6w_run{run}.json"
            if p.exists():
                thomas_costs.append(json.load(open(p))["best_cost"])
        if len(ours_costs) >= 2 and len(thomas_costs) >= 2:
            t_stat, p_val = stats.ttest_ind(ours_costs, thomas_costs, equal_var=False)
            verdict = "EQUIVALENT" if p_val >= 0.05 else "DIFFERENT"
            print(f"{ds:>8} {np.mean(ours_costs):>11.2f} {np.mean(thomas_costs):>13.2f} "
                  f"{p_val:>10.4f} {verdict:>12}")
