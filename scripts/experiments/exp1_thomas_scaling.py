#!/usr/bin/env python3
"""Test whether Thomas's code scales beyond Pool(6).

Patches Thomas's Pool size to test [6, 8, 13] workers.
Does NOT modify Thomas's source files -- monkey-patches at runtime.

If Thomas scales fine at 13 workers -> BLAS suppression claim is weak.
If Thomas slows down or crashes at 13 -> BLAS suppression claim is real.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp1_thomas_scaling.py
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

import matplotlib
matplotlib.use('Agg')

import sys
import os
import time
import json
import io
import warnings
from multiprocessing.pool import Pool
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_PATH = str(PROJECT_ROOT / "program code" / "Data" / "5k no BB.xlsx")

BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])

POP_SIZE = 100
GENS = 42
WORKER_COUNTS = [6, 8, 13]


def run_thomas_at_workers(n_workers):
    """Run Thomas's FDDC with a patched Pool size."""
    import simulation as thomas_sim
    from distributionComparison import min_maxV2
    import fddc as thomas_fddc_module

    norm = min_maxV2(DATA_PATH, thomas_sim.polymer, sigma=[1] * 6, transfac=1)

    # Monkey-patch: replace Pool(6) with Pool(n_workers)
    original_pool_init = Pool.__init__

    def patched_pool_init(self_pool, processes=None, *args, **kwargs):
        original_pool_init(self_pool, processes=n_workers, *args, **kwargs)

    Pool.__init__ = patched_pool_init

    print(f"\nThomas FDDC | pop={POP_SIZE} | gens={GENS} | Pool({n_workers})")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    t0 = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            fc = thomas_fddc_module.fddc(
                bounds=BOUNDS, fitnessFunction=norm.costFunction,
                distribution_comparison=norm, populationSize=POP_SIZE,
                graph=False, ui_plot=False)
        init_time = time.time() - t0
        print(f"Init: {init_time:.1f}s")

        cost_history = []
        for i in range(GENS):
            g_start = time.time()
            with redirect_stdout(io.StringIO()):
                fc.run()
            g_time = time.time() - g_start
            cost = float(fc.best_score) if not isinstance(fc.best_score, list) else float(fc.best_score[0])
            cost_history.append(cost)
            elapsed = time.time() - t0
            remaining = (GENS - i - 1) * elapsed / (i + 1)
            eta = datetime.now() + timedelta(seconds=remaining)
            print(f"  Gen {i+1:3d}/{GENS} | Cost: {cost:.4f} | "
                  f"{g_time:.0f}s/gen | ETA: {eta.strftime('%H:%M:%S')}")

        total = time.time() - t0
        status = "OK"
    except Exception as e:
        total = time.time() - t0
        cost_history = []
        status = f"FAILED: {e}"
        print(f"  FAILED: {e}")
    finally:
        # Restore original Pool init
        Pool.__init__ = original_pool_init
        # Kill Thomas's global pool
        if hasattr(thomas_fddc_module, 'p') and thomas_fddc_module.p is not None:
            thomas_fddc_module.p.terminate()
            thomas_fddc_module.p.join()

    result = {
        "implementation": f"thomas_pool{n_workers}",
        "dataset": "5k", "pop_size": POP_SIZE, "gens": GENS,
        "workers": n_workers,
        "best_cost": min(cost_history) if cost_history else None,
        "cost_history": cost_history,
        "total_time_sec": total, "total_time_min": total / 60,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"Total: {total/60:.1f} min | Status: {status}")
    if cost_history:
        print(f"Best: {min(cost_history):.4f}")
    return result


if __name__ == '__main__':
    print("=" * 60)
    print("THOMAS SCALING TEST: Does his code scale past Pool(6)?")
    print(f"Workers to test: {WORKER_COUNTS}")
    print("=" * 60)

    for w in WORKER_COUNTS:
        out_file = OUTPUT_DIR / f"exp1_thomas_pool{w}_5k.json"
        if out_file.exists():
            print(f"\nSkip: {out_file.name}")
            continue

        result = run_thomas_at_workers(w)
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {out_file.name}")

    # Summary
    print(f"\n{'='*60}")
    print("THOMAS SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':>8} {'Time (min)':>11} {'Best Cost':>11} {'Status':>10}")
    print("-" * 44)
    for w in WORKER_COUNTS:
        path = OUTPUT_DIR / f"exp1_thomas_pool{w}_5k.json"
        if path.exists():
            d = json.load(open(path))
            t = d['total_time_min']
            c = d['best_cost'] or 'N/A'
            s = d['status']
            cost_str = f"{c:.4f}" if isinstance(c, float) else c
            print(f"{w:>8} {t:>11.1f} {cost_str:>11} {s:>10}")
