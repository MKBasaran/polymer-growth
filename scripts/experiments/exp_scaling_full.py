#!/usr/bin/env python3
"""Experiment 3: Worker scaling at production config.

Tests parallelization speedup with the full Table VIII config
(pop=100, 42 gens on 5k) across worker counts [1, 2, 4, 6, 8, 13].

Previous speed_benchmark.json used pop=50, 10 gens -- too small
for thesis claims. This uses production workload.

Usage:
    nohup /usr/local/bin/python3 scripts/experiments/exp_scaling_full.py > scaling.log 2>&1 &
"""
import sys
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

WORKER_COUNTS = [1, 2, 4, 6, 8, 13]
SEEDS = [42, 123, 777]
DATASET = "5k"
POP_SIZE = 100


def main():
    from shared import run_fddc, THOMAS_TABLE_VIII
    ref = THOMAS_TABLE_VIII[DATASET]
    gens = ref["fddc_gens"]

    n_runs = len(WORKER_COUNTS) * len(SEEDS)
    print("=" * 60)
    print("EXPERIMENT 3: WORKER SCALING (PRODUCTION CONFIG)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {DATASET} | pop={POP_SIZE} | gens={gens}")
    print(f"Workers: {WORKER_COUNTS} | Seeds: {SEEDS}")
    print(f"Total runs: {n_runs}")
    print("=" * 60)

    # Estimate: 1-worker run might take ~60 min, rest faster
    print("Estimated runtime: 2-3 hours (1-worker run dominates)\n")

    for w in WORKER_COUNTS:
        for seed in SEEDS:
            out_file = OUTPUT_DIR / f"exp3_scaling_{w}w_seed{seed}.json"
            if out_file.exists():
                print(f"  Skip: {out_file.name}")
                continue

            print(f"  Running: workers={w} seed={seed}...")
            start = time.time()
            result = run_fddc(
                dataset_key=DATASET,
                gen_count=gens,
                pop_size=POP_SIZE,
                impl="new",
                seed=seed,
                workers=w,
            )
            result["n_workers_actual"] = w
            result["timestamp"] = datetime.now().isoformat()

            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_file.name} | {result['elapsed_min']:.1f} min | "
                  f"cost={result['best_cost']:.4f}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':>8} {'Mean Time (s)':>14} {'Speedup':>8} {'Efficiency':>11}")
    print("-" * 45)

    baseline_time = None
    for w in WORKER_COUNTS:
        times = []
        for seed in SEEDS:
            path = OUTPUT_DIR / f"exp3_scaling_{w}w_seed{seed}.json"
            if path.exists():
                with open(path) as f:
                    times.append(json.load(f)["elapsed_sec"])
        if not times:
            continue
        avg = np.mean(times)
        if w == 1:
            baseline_time = avg
        speedup = baseline_time / avg if baseline_time else 1.0
        efficiency = speedup / w * 100
        print(f"{w:>8} {avg:>14.1f} {speedup:>8.2f}x {efficiency:>10.1f}%")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()