#!/usr/bin/env python3
"""Run ONLY Thomas's actual fddc.py end-to-end. Nothing else.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp1_run_thomas_only.py
"""
import multiprocessing as mp
mp.set_start_method('fork', force=True)

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

if __name__ == '__main__':
    import simulation as thomas_sim
    from distributionComparison import min_maxV2
    from fddc import fddc as ThomasFDDC

    norm = min_maxV2(DATA_PATH, thomas_sim.polymer, sigma=[1] * 6, transfac=1)

    print(f"Thomas FDDC | pop={POP_SIZE} | gens={GENS} | Pool(6)")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

    t0 = time.time()
    with redirect_stdout(io.StringIO()):
        fc = ThomasFDDC(bounds=BOUNDS, fitnessFunction=norm.costFunction,
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

    result = {
        "implementation": "thomas_original_endtoend",
        "dataset": "5k", "pop_size": POP_SIZE, "gens": GENS,
        "workers": 6, "best_cost": min(cost_history),
        "cost_history": cost_history,
        "init_time_sec": init_time,
        "total_time_sec": total, "total_time_min": total / 60,
        "avg_gen_time_sec": (total - init_time) / GENS,
        "timestamp": datetime.now().isoformat(),
    }

    out = OUTPUT_DIR / "exp1_thomas_endtoend_5k.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nTotal: {total/60:.1f} min | Best: {min(cost_history):.4f}")
    print(f"Saved: {out}")