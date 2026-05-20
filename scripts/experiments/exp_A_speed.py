#!/usr/bin/env python3
"""Experiment A: Speed comparison (IV = codebase).
Both codebases at 6 workers, 5k dataset, 42 gens, 30 seeds.
Each run in a fresh subprocess -- zero shared state.

Skips any result file that already exists on disk so previous runs are preserved.

Estimated time: ~2.5 hours (60 runs x ~5 min, minus already-completed)

Usage:
    python3 exp_A_speed.py
"""
import json
import numpy as np
from scipy import stats
from _runner import run_thomas, run_ours, save_result, OUTPUT_DIR

SEEDS = [
    42, 123, 777, 2024, 9999,
    101, 202, 303, 404, 505,
    1001, 1002, 1003, 1004, 1005,
    2001, 2002, 2003, 2004, 2005,
    3001, 3002, 3003, 3004, 3005,
    4001, 4002, 4003, 4004, 4005,
]

if __name__ == '__main__':
    print("=" * 60)
    print("EXPERIMENT A: SPEED COMPARISON (IV = codebase)")
    print("Fixed: 6 workers, 5k dataset, 42 gens, pop=100")
    print(f"Seeds: {len(SEEDS)} total")
    print("=" * 60)

    for seed in SEEDS:
        fname = f"expA_thomas_5k_6w_seed{seed}.json"
        if (OUTPUT_DIR / fname).exists():
            print(f"  Skip (exists): {fname}")
        else:
            r = run_thomas("5k", workers=6)
            save_result(r, fname)

        fname = f"expA_ours_5k_6w_seed{seed}.json"
        if (OUTPUT_DIR / fname).exists():
            print(f"  Skip (exists): {fname}")
        else:
            r = run_ours("5k", workers=6, seed=seed)
            save_result(r, fname)

    # Collect results
    thomas_times, ours_times = [], []
    thomas_costs, ours_costs = [], []
    print(f"\n{'='*60}")
    print("EXPERIMENT A RESULTS")
    print(f"{'='*60}")
    print(f"{'Impl':<8} {'Seed':>6} {'Time (min)':>11} {'Best Cost':>11}")
    print("-" * 42)
    for seed in SEEDS:
        for impl in ["thomas", "ours"]:
            p = OUTPUT_DIR / f"expA_{impl}_5k_6w_seed{seed}.json"
            if p.exists():
                d = json.load(open(p))
                print(f"{impl:<8} {seed:>6} {d['elapsed_min']:>11.1f} {d['best_cost']:>11.4f}")
                if impl == "thomas":
                    thomas_times.append(d["elapsed_sec"])
                    thomas_costs.append(d["best_cost"])
                else:
                    ours_times.append(d["elapsed_sec"])
                    ours_costs.append(d["best_cost"])

    if len(thomas_times) >= 2 and len(ours_times) >= 2:
        t_avg, t_std = np.mean(thomas_times), np.std(thomas_times, ddof=1)
        o_avg, o_std = np.mean(ours_times), np.std(ours_times, ddof=1)
        t_stat, p_val = stats.ttest_ind(thomas_times, ours_times, equal_var=False)
        t_ci = stats.t.interval(0.95, len(thomas_times)-1, loc=np.mean(thomas_times), scale=stats.sem(thomas_times))
        o_ci = stats.t.interval(0.95, len(ours_times)-1, loc=np.mean(ours_times), scale=stats.sem(ours_times))

        print(f"\nThomas: {t_avg/60:.2f} +/- {t_std/60:.2f} min (n={len(thomas_times)})")
        print(f"  95% CI: [{t_ci[0]/60:.2f}, {t_ci[1]/60:.2f}] min")
        print(f"Ours:   {o_avg/60:.2f} +/- {o_std/60:.2f} min (n={len(ours_times)})")
        print(f"  95% CI: [{o_ci[0]/60:.2f}, {o_ci[1]/60:.2f}] min")
        print(f"Speedup: {t_avg/o_avg:.2f}x")
        print(f"Welch's t-test (time): t={t_stat:.3f}, p={p_val:.6f}")
        print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

        # Cost equivalence test
        t_cost, p_cost = stats.ttest_ind(thomas_costs, ours_costs, equal_var=False)
        print(f"\nCost comparison:")
        print(f"  Thomas mean cost: {np.mean(thomas_costs):.4f} +/- {np.std(thomas_costs, ddof=1):.4f}")
        print(f"  Ours mean cost:   {np.mean(ours_costs):.4f} +/- {np.std(ours_costs, ddof=1):.4f}")
        print(f"  Welch's t-test (cost): t={t_cost:.3f}, p={p_cost:.6f}")
        print(f"  {'DIFFERENT' if p_cost < 0.05 else 'EQUIVALENT (no significant difference)'} at alpha=0.05")
