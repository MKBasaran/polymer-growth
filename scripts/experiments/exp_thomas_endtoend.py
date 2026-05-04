#!/usr/bin/env python3
"""Experiment 1: True end-to-end speed comparison.

Runs Thomas's ACTUAL fddc.py pipeline (his Pool(6), his sequential
reproduce_pop1, his everything) vs our refactored pipeline on the
same M4 Pro hardware.

This is the only valid speed comparison -- previous experiments
plugged Thomas's sim into our FDDC, which is not end-to-end.

Usage:
    nohup /usr/local/bin/python3 scripts/experiments/exp_thomas_endtoend.py > thomas_e2e.log 2>&1 &
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)  # Must be before Pool creation

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')

import sys
import time
import json
import io
import warnings
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Thomas's Table VIII reference data
TABLE_VIII = {
    "5k":  {"file": "5k no BB.xlsx",  "gens": 42, "thomas_cost": 21.7358},
    "10k": {"file": "10k no BB.xlsx", "gens": 56, "thomas_cost": 108.5352},
    "20k": {"file": "20k no BB.xlsx", "gens": 44, "thomas_cost": 131.7344},
    "30k": {"file": "30k no BB.xlsx", "gens": 27, "thomas_cost": 340.2589},
}

BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])

POP_SIZE = 100
THOMAS_WORKERS = 6  # Thomas hardcodes Pool(6)


def run_thomas_endtoend(dataset_key):
    """Run Thomas's actual fddc.py end-to-end (headless)."""
    ref = TABLE_VIII[dataset_key]
    data_path = str(PROJECT_ROOT / "program code" / "Data" / ref["file"])
    gens = ref["gens"]

    import simulation as thomas_sim
    from distributionComparison import min_maxV2

    # Thomas's cost function -- fig=None lets parent create Agg figure
    norm = min_maxV2(data_path, thomas_sim.polymer, sigma=[1] * 6, transfac=1)

    from fddc import fddc as ThomasFDDC

    print(f"\n{'='*60}")
    print(f"THOMAS END-TO-END: {dataset_key} | pop={POP_SIZE} | gens={gens}")
    print(f"{'='*60}")

    # --- Init (includes Pool(6) + initial fitness for all 100 individuals) ---
    print("  Initializing Thomas FDDC (Pool(6) + initial pop eval)...")
    init_start = time.time()
    with redirect_stdout(io.StringIO()):
        fc = ThomasFDDC(
            bounds=BOUNDS,
            fitnessFunction=norm.costFunction,
            distribution_comparison=norm,
            populationSize=POP_SIZE,
            graph=False,
            ui_plot=False,
        )
    init_time = time.time() - init_start
    print(f"  Init complete: {init_time:.1f}s ({init_time/60:.1f} min)")

    # --- Run generations ---
    gen_start = time.time()
    cost_history = []
    gen_times = []
    for i in range(gens):
        g_start = time.time()
        with redirect_stdout(io.StringIO()):
            fc.run()
        g_time = time.time() - g_start
        gen_times.append(g_time)
        best_cost = float(fc.best_score) if not isinstance(fc.best_score, list) else float(fc.best_score[0])
        cost_history.append(best_cost)
        elapsed = time.time() - gen_start
        avg_gen = elapsed / (i + 1)
        remaining = (gens - i - 1) * avg_gen
        eta = datetime.now() + timedelta(seconds=remaining)
        print(f"  Gen {i+1:3d}/{gens} | Cost: {best_cost:.4f} | "
              f"{g_time:.0f}s/gen | ETA: {eta.strftime('%H:%M:%S')}")

    total_time = time.time() - init_start

    # Clean up Thomas's global pool
    import fddc as thomas_fddc_module
    if hasattr(thomas_fddc_module, 'p') and thomas_fddc_module.p is not None:
        thomas_fddc_module.p.terminate()
        thomas_fddc_module.p.join()

    return {
        "implementation": "thomas_original_endtoend",
        "dataset": dataset_key,
        "pop_size": POP_SIZE,
        "gens": gens,
        "workers": THOMAS_WORKERS,
        "best_cost": cost_history[-1] if cost_history else None,
        "cost_history": cost_history,
        "init_time_sec": init_time,
        "gen_time_sec": time.time() - gen_start,
        "total_time_sec": total_time,
        "total_time_min": total_time / 60,
        "total_time_hours": total_time / 3600,
        "avg_gen_time_sec": np.mean(gen_times),
        "thomas_reported_cost": ref["thomas_cost"],
        "timestamp": datetime.now().isoformat(),
    }


def run_ours_endtoend(dataset_key, seed=42, n_workers=6):
    """Run our refactored code end-to-end."""
    from shared import run_fddc
    ref = TABLE_VIII[dataset_key]
    print(f"\n{'='*60}")
    print(f"OUR CODE: {dataset_key} | pop={POP_SIZE} | gens={ref['gens']} | "
          f"workers={n_workers} | seed={seed}")
    print(f"{'='*60}")
    result = run_fddc(
        dataset_key=dataset_key,
        gen_count=ref["gens"],
        pop_size=POP_SIZE,
        impl="new",
        seed=seed,
        workers=n_workers,
    )
    result["implementation"] = f"ours_{n_workers}w"
    result["timestamp"] = datetime.now().isoformat()
    return result


def main():
    print("=" * 60)
    print("EXPERIMENT 1: TRUE END-TO-END SPEED COMPARISON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware: {os.cpu_count()} cores")
    print("=" * 60)
    print("\nEstimated runtime: UNKNOWN for Thomas (hours per dataset).")
    print("Our code: ~5 min per dataset at 6 workers.\n")

    # Thomas ran 24h per dataset on 2020 Intel with Pool(6).
    # On M4 Pro that's ~8h per dataset. Only run 5k for now.
    # Additional datasets are future work.
    datasets = ["5k"]

    for ds in datasets:
        out_thomas = OUTPUT_DIR / f"exp1_thomas_endtoend_{ds}.json"
        out_ours_6w = OUTPUT_DIR / f"exp1_ours_6w_{ds}_seed42.json"
        out_ours_full = OUTPUT_DIR / f"exp1_ours_fullw_{ds}_seed42.json"

        # --- Thomas end-to-end ---
        if out_thomas.exists():
            print(f"\nSkipping Thomas {ds} (exists: {out_thomas.name})")
        else:
            result = run_thomas_endtoend(ds)
            with open(out_thomas, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_thomas.name}")
            print(f"  Total: {result['total_time_min']:.1f} min | "
                  f"Best cost: {result['best_cost']}")

        # --- Our code at 6 workers (fair comparison) ---
        if out_ours_6w.exists():
            print(f"\nSkipping ours-6w {ds} (exists: {out_ours_6w.name})")
        else:
            result = run_ours_endtoend(ds, seed=42, n_workers=6)
            with open(out_ours_6w, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_ours_6w.name}")
            print(f"  Total: {result['elapsed_min']:.1f} min | "
                  f"Best cost: {result['best_cost']}")

        # --- Our code at max workers (real-world advantage) ---
        max_w = max(1, (os.cpu_count() or 4) - 1)
        if out_ours_full.exists():
            print(f"\nSkipping ours-{max_w}w {ds} (exists: {out_ours_full.name})")
        else:
            result = run_ours_endtoend(ds, seed=42, n_workers=max_w)
            with open(out_ours_full, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {out_ours_full.name}")
            print(f"  Total: {result['elapsed_min']:.1f} min | "
                  f"Best cost: {result['best_cost']}")

        # Print comparison for this dataset
        times = {}
        for label, path in [("thomas", out_thomas), ("ours_6w", out_ours_6w),
                            ("ours_full", out_ours_full)]:
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                key = "total_time_min" if "total_time_min" in d else "elapsed_min"
                times[label] = d[key]

        if "thomas" in times and "ours_6w" in times:
            speedup = times["thomas"] / times["ours_6w"]
            print(f"\n  >> {ds} SPEEDUP (6w fair): {speedup:.1f}x "
                  f"({times['thomas']:.1f} min vs {times['ours_6w']:.1f} min)")
        if "thomas" in times and "ours_full" in times:
            speedup = times["thomas"] / times["ours_full"]
            print(f"  >> {ds} SPEEDUP (full):   {speedup:.1f}x "
                  f"({times['thomas']:.1f} min vs {times['ours_full']:.1f} min)")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()