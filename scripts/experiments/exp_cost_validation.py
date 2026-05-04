#!/usr/bin/env python3
"""Cost validation: run Thomas's code on 10k/20k/30k to see if costs match Table VIII.

If Thomas's code also gets high costs in 56/44/27 gens -> Table VIII came from 24hr budget.
If Thomas's code matches Table VIII -> our cost function differs.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp_cost_validation.py
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
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"

BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])

TABLE_VIII = {
    "5k":  {"file": "5k no BB.xlsx",  "gens": 42, "cost": 21.7358},
    "10k": {"file": "10k no BB.xlsx", "gens": 56, "cost": 108.5352},
    "20k": {"file": "20k no BB.xlsx", "gens": 44, "cost": 131.7344},
    "30k": {"file": "30k no BB.xlsx", "gens": 27, "cost": 340.2589},
}


def _noop(*a, **kw):
    pass


if __name__ == '__main__':
    import simulation as thomas_sim
    from distributionComparison import min_maxV2
    import fddc as thomas_fddc_module
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("COST VALIDATION: Thomas's code on all datasets")
    print("Pool(6), pop=100, Table VIII gen counts")
    print("=" * 60)

    for ds, ref in TABLE_VIII.items():
        fname = f"expD_thomas_{ds}_6w.json"
        if (OUTPUT_DIR / fname).exists():
            print(f"\nSkip: {fname}")
            d = json.load(open(OUTPUT_DIR / fname))
            print(f"  {ds}: {d['elapsed_min']:.1f} min | cost={d['best_cost']:.4f} | "
                  f"Thomas Table VIII={ref['cost']}")
            continue

        data_path = str(PROJECT_ROOT / "program code" / "Data" / ref["file"])
        gens = ref["gens"]

        norm = min_maxV2(data_path, thomas_sim.polymer, sigma=[1] * 6, transfac=1)
        plt.close(norm.fig)
        norm.fig = None
        norm.ax0 = None
        norm.ax1 = None
        norm.ax2 = None
        norm.plotDistributions = _noop

        print(f"\n{ds}: running Thomas FDDC, {gens} gens, Pool(6)...")

        t0 = time.time()
        with redirect_stdout(io.StringIO()):
            fc = thomas_fddc_module.fddc(
                bounds=BOUNDS, fitnessFunction=norm.costFunction,
                distribution_comparison=norm, populationSize=100,
                graph=False, ui_plot=False)

        cost_history = []
        for i in range(gens):
            with redirect_stdout(io.StringIO()):
                fc.run()
            cost = float(fc.best_score) if not isinstance(fc.best_score, list) else float(fc.best_score[0])
            cost_history.append(cost)
            if (i + 1) % 10 == 0 or i == gens - 1:
                print(f"  Gen {i+1}/{gens} | cost={cost:.4f}")

        total = time.time() - t0

        if hasattr(thomas_fddc_module, 'p') and thomas_fddc_module.p is not None:
            thomas_fddc_module.p.terminate()
            thomas_fddc_module.p.join()

        result = {
            "impl": "thomas", "dataset": ds, "workers": 6,
            "gens": gens, "pop": 100,
            "best_cost": min(cost_history), "cost_history": cost_history,
            "elapsed_sec": total, "elapsed_min": total / 60,
            "thomas_table_viii_cost": ref["cost"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(OUTPUT_DIR / fname, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"  DONE: {total/60:.1f} min | our_cost={min(cost_history):.4f} | "
              f"Thomas_Table_VIII={ref['cost']}")

    # Summary
    print(f"\n{'='*60}")
    print("COST VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':>8} {'Thomas now':>12} {'Table VIII':>12} {'Match?':>8}")
    print("-" * 44)
    for ds, ref in TABLE_VIII.items():
        path = OUTPUT_DIR / f"expD_thomas_{ds}_6w.json"
        if path.exists():
            d = json.load(open(path))
            ratio = d['best_cost'] / ref['cost']
            match = 'YES' if 0.7 < ratio < 1.3 else 'NO'
            print(f"{ds:>8} {d['best_cost']:>12.4f} {ref['cost']:>12.4f} {match:>8}")
