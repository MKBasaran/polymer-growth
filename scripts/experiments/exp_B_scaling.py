#!/usr/bin/env python3
"""Experiment B: Worker scaling (IV = worker count).
Both codebases at [6, 8, 10, 13] workers, 5k dataset, 42 gens, 10 seeds.
Each run in a fresh subprocess -- zero shared state.

Skips any result file that already exists on disk.

Estimated time: ~5 hours (80 runs x ~4 min, minus already-completed)

Usage:
    python3 exp_B_scaling.py
"""
import json
import numpy as np
from scipy import stats
from _runner import run_thomas, run_ours, save_result, OUTPUT_DIR

WORKERS = [6, 8, 10, 13]
SEEDS = [42, 123, 777, 2024, 9999, 101, 202, 303, 404, 505]

if __name__ == '__main__':
    print("=" * 60)
    print("EXPERIMENT B: WORKER SCALING (IV = worker count)")
    print("Fixed: 5k dataset, 42 gens, pop=100")
    print(f"Workers: {WORKERS} | Seeds: {len(SEEDS)}")
    print("=" * 60)

    for w in WORKERS:
        for seed in SEEDS:
            fname = f"expB_ours_5k_{w}w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip (exists): {fname}")
            else:
                r = run_ours("5k", workers=w, seed=seed)
                save_result(r, fname)

            fname = f"expB_thomas_5k_{w}w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip (exists): {fname}")
            else:
                r = run_thomas("5k", workers=w)
                save_result(r, fname)

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT B RESULTS")
    print(f"{'='*60}")
    print(f"{'Workers':>8} {'Ours mean':>10} {'Ours CI':>16} "
          f"{'Thomas mean':>12} {'Thomas CI':>16} {'Ours/Thomas':>12}")
    print("-" * 78)

    ours_6w_times = []
    thomas_6w_times = []
    for w in WORKERS:
        o_times, t_times = [], []
        for seed in SEEDS:
            p = OUTPUT_DIR / f"expB_ours_5k_{w}w_seed{seed}.json"
            if p.exists(): o_times.append(json.load(open(p))["elapsed_sec"])
            p = OUTPUT_DIR / f"expB_thomas_5k_{w}w_seed{seed}.json"
            if p.exists(): t_times.append(json.load(open(p))["elapsed_sec"])

        if w == 6:
            ours_6w_times = o_times[:]
            thomas_6w_times = t_times[:]

        if o_times and t_times:
            o_avg = np.mean(o_times)
            t_avg = np.mean(t_times)
            o_ci = stats.t.interval(0.95, len(o_times)-1, loc=o_avg, scale=stats.sem(o_times))
            t_ci = stats.t.interval(0.95, len(t_times)-1, loc=t_avg, scale=stats.sem(t_times))
            ratio = o_avg / t_avg
            print(f"{w:>8} {o_avg/60:>9.2f}m [{o_ci[0]/60:.2f},{o_ci[1]/60:.2f}] "
                  f"{t_avg/60:>11.2f}m [{t_ci[0]/60:.2f},{t_ci[1]/60:.2f}] "
                  f"{ratio:>12.3f}")

    # Scaling table
    print(f"\nScaling vs 6w baseline:")
    print(f"{'Workers':>8} {'Ours speedup':>13} {'Thomas speedup':>15}")
    print("-" * 40)
    o_base = np.mean(ours_6w_times) if ours_6w_times else 1
    t_base = np.mean(thomas_6w_times) if thomas_6w_times else 1
    for w in WORKERS:
        o_times, t_times = [], []
        for seed in SEEDS:
            p = OUTPUT_DIR / f"expB_ours_5k_{w}w_seed{seed}.json"
            if p.exists(): o_times.append(json.load(open(p))["elapsed_sec"])
            p = OUTPUT_DIR / f"expB_thomas_5k_{w}w_seed{seed}.json"
            if p.exists(): t_times.append(json.load(open(p))["elapsed_sec"])
        if o_times and t_times:
            o_sp = o_base / np.mean(o_times)
            t_sp = t_base / np.mean(t_times)
            print(f"{w:>8} {o_sp:>13.2f}x {t_sp:>15.2f}x")
