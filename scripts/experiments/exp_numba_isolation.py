#!/usr/bin/env python3
"""Experiment 2: Numba JIT impact isolation.

Runs our FDDC pipeline with Numba enabled (default) vs disabled
(NUMBA_DISABLE_JIT=1). Same config, same seeds, same hardware.
Isolates Numba's contribution to overall performance.

When invoked with --disable-numba, sets the env var before importing
numba. The main function runs both modes via subprocess.

Usage:
    nohup /usr/local/bin/python3 scripts/experiments/exp_numba_isolation.py > numba_isolation.log 2>&1 &
"""
import os
import sys

# MUST be before any numba import
_NUMBA_DISABLED = '--disable-numba' in sys.argv
if _NUMBA_DISABLED:
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    sys.argv.remove('--disable-numba')

import time
import json
import subprocess
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

SEEDS = [42, 123, 777]
DATASETS = ["5k", "10k", "20k", "30k"]
POP_SIZE = 100


def run_single(dataset_key, seed, numba_enabled):
    """Run one FDDC optimization, return results dict."""
    from shared import run_fddc, THOMAS_TABLE_VIII
    ref = THOMAS_TABLE_VIII[dataset_key]
    label = "enabled" if numba_enabled else "disabled"
    print(f"  Running {dataset_key} seed={seed} numba={label}...")

    result = run_fddc(
        dataset_key=dataset_key,
        gen_count=ref["fddc_gens"],
        pop_size=POP_SIZE,
        impl="new",
        seed=seed,
    )
    result["numba"] = label
    result["timestamp"] = datetime.now().isoformat()
    return result


def run_worker_mode():
    """Called when script is invoked with --run-single."""
    dataset_key = sys.argv[sys.argv.index('--dataset') + 1]
    seed = int(sys.argv[sys.argv.index('--seed') + 1])
    numba_enabled = '--disable-numba' not in os.environ.get('_NUMBA_FLAG', '')

    label = "enabled" if not _NUMBA_DISABLED else "disabled"
    result = run_single(dataset_key, seed, not _NUMBA_DISABLED)

    out_file = OUTPUT_DIR / f"exp2_numba_{dataset_key}_{label}_seed{seed}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_file.name} | {result['elapsed_min']:.1f} min | "
          f"cost={result['best_cost']:.4f}")


def main():
    print("=" * 60)
    print("EXPERIMENT 2: NUMBA JIT IMPACT ISOLATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    n_runs = len(DATASETS) * len(SEEDS) * 2
    print(f"\nRuns: {n_runs} ({len(DATASETS)} datasets x {len(SEEDS)} seeds x 2 modes)")
    print(f"Estimated runtime: ~{n_runs * 5} min\n")

    script = str(Path(__file__).resolve())

    for ds in DATASETS:
        for seed in SEEDS:
            # --- With Numba (run in-process) ---
            out_enabled = OUTPUT_DIR / f"exp2_numba_{ds}_enabled_seed{seed}.json"
            if out_enabled.exists():
                print(f"  Skip: {out_enabled.name}")
            else:
                result = run_single(ds, seed, numba_enabled=True)
                with open(out_enabled, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved: {out_enabled.name} | "
                      f"{result['elapsed_min']:.1f} min | "
                      f"cost={result['best_cost']:.4f}")

            # --- Without Numba (subprocess to ensure env var set before import) ---
            out_disabled = OUTPUT_DIR / f"exp2_numba_{ds}_disabled_seed{seed}.json"
            if out_disabled.exists():
                print(f"  Skip: {out_disabled.name}")
            else:
                print(f"  Launching subprocess: {ds} seed={seed} numba=disabled")
                cmd = [
                    sys.executable, script,
                    '--disable-numba', '--run-single',
                    '--dataset', ds, '--seed', str(seed),
                ]
                proc = subprocess.run(cmd, capture_output=False)
                if proc.returncode != 0:
                    print(f"  ERROR: subprocess failed for {ds} seed={seed}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY: Numba Impact")
    print(f"{'='*60}")
    for ds in DATASETS:
        enabled_times, disabled_times = [], []
        for seed in SEEDS:
            for label, times_list in [("enabled", enabled_times),
                                       ("disabled", disabled_times)]:
                path = OUTPUT_DIR / f"exp2_numba_{ds}_{label}_seed{seed}.json"
                if path.exists():
                    with open(path) as f:
                        times_list.append(json.load(f)["elapsed_sec"])
        if enabled_times and disabled_times:
            e_avg = np.mean(enabled_times)
            d_avg = np.mean(disabled_times)
            diff_pct = (e_avg - d_avg) / d_avg * 100
            faster = "SLOWER" if diff_pct > 0 else "FASTER"
            print(f"  {ds}: Numba={e_avg:.0f}s | No-Numba={d_avg:.0f}s | "
                  f"Numba is {abs(diff_pct):.1f}% {faster}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    if '--run-single' in sys.argv:
        run_worker_mode()
    else:
        main()