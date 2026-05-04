#!/usr/bin/env python3
"""Final experiment suite: controlled single-IV experiments.

A) Speed comparison: Thomas vs Ours at 6 workers (IV = codebase)
B) Worker scaling: our code at [1,2,4,6,8,13] workers (IV = worker count)
C) Dataset scaling: our code on [5k,10k,20k,30k] (IV = dataset)

Each experiment varies ONE independent variable, controls everything else.
3 seeds per condition for statistical power.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp_final_suite.py --exp A
    /usr/local/bin/python3 scripts/experiments/exp_final_suite.py --exp B
    /usr/local/bin/python3 scripts/experiments/exp_final_suite.py --exp C
    /usr/local/bin/python3 scripts/experiments/exp_final_suite.py --exp all
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
import subprocess
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 777]


def _noop(*a, **kw):
    """Picklable no-op to replace matplotlib methods in workers."""
    pass
POP_SIZE = 100

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


def run_thomas(dataset, seed, workers=6):
    """Run Thomas's actual fddc.py end-to-end.

    Monkey-patches Pool size at runtime to test scaling.
    Thomas hardcoded Pool(6) -- we override without modifying his source.
    """
    import simulation as thomas_sim
    from distributionComparison import min_maxV2
    import fddc as thomas_fddc_module
    from multiprocessing.pool import Pool as _Pool

    ref = TABLE_VIII[dataset]
    data_path = str(PROJECT_ROOT / "program code" / "Data" / ref["file"])
    gens = ref["gens"]

    norm = min_maxV2(data_path, thomas_sim.polymer, sigma=[1] * 6, transfac=1)

    # Kill matplotlib state to prevent pickle errors in workers
    import matplotlib.pyplot as plt
    plt.close(norm.fig)
    norm.fig = None
    norm.ax0 = None
    norm.ax1 = None
    norm.ax2 = None
    norm.plotDistributions = _noop

    # Monkey-patch Pool to use requested worker count
    _orig_init = _Pool.__init__
    def _patched_init(self_pool, processes=None, *a, **kw):
        _orig_init(self_pool, processes=workers, *a, **kw)
    _Pool.__init__ = _patched_init

    t0 = time.time()
    try:
        with redirect_stdout(io.StringIO()):
            fc = thomas_fddc_module.fddc(
                bounds=BOUNDS, fitnessFunction=norm.costFunction,
                distribution_comparison=norm, populationSize=POP_SIZE,
                graph=False, ui_plot=False)

        cost_history = []
        for i in range(gens):
            with redirect_stdout(io.StringIO()):
                fc.run()
            cost = float(fc.best_score) if not isinstance(fc.best_score, list) else float(fc.best_score[0])
            cost_history.append(cost)

        total = time.time() - t0
    finally:
        _Pool.__init__ = _orig_init
        if hasattr(thomas_fddc_module, 'p') and thomas_fddc_module.p is not None:
            thomas_fddc_module.p.terminate()
            thomas_fddc_module.p.join()

    return {
        "impl": "thomas", "dataset": dataset, "seed": seed,
        "workers": workers, "gens": gens, "pop": POP_SIZE,
        "best_cost": min(cost_history), "cost_history": cost_history,
        "elapsed_sec": total, "elapsed_min": total / 60,
        "timestamp": datetime.now().isoformat(),
    }


def run_ours(dataset, seed, workers):
    """Run our optimized code."""
    from shared import run_fddc
    ref = TABLE_VIII[dataset]
    result = run_fddc(
        dataset_key=dataset, gen_count=ref["gens"], pop_size=POP_SIZE,
        impl="new", seed=seed, workers=workers, use_fast_sim=True,
    )
    result["impl"] = "ours"
    result["workers"] = workers
    result["timestamp"] = datetime.now().isoformat()
    return result


def save(result, filename):
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    return path


def run_exp_a():
    """A) Speed comparison: Thomas vs Ours at 6 workers, 5k dataset."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: SPEED COMPARISON (IV = codebase)")
    print("Fixed: 6 workers, 5k dataset, 42 gens, pop=100")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    for seed in SEEDS:
        # Thomas
        fname = f"expA_thomas_5k_6w_seed{seed}.json"
        if (OUTPUT_DIR / fname).exists():
            print(f"  Skip: {fname}")
        else:
            print(f"  Thomas seed={seed}...")
            r = run_thomas("5k", seed)
            save(r, fname)
            print(f"    {r['elapsed_min']:.1f} min | cost={r['best_cost']:.4f}")

        # Ours at 6 workers
        fname = f"expA_ours_5k_6w_seed{seed}.json"
        if (OUTPUT_DIR / fname).exists():
            print(f"  Skip: {fname}")
        else:
            print(f"  Ours 6w seed={seed}...")
            r = run_ours("5k", seed, 6)
            save(r, fname)
            print(f"    {r['elapsed_min']:.1f} min | cost={r['best_cost']:.4f}")

    # Summary
    print(f"\n{'Impl':<10} {'Seed':>6} {'Time (min)':>11} {'Best Cost':>11}")
    print("-" * 42)
    for seed in SEEDS:
        for impl in ["thomas", "ours"]:
            fname = f"expA_{impl}_5k_6w_seed{seed}.json"
            path = OUTPUT_DIR / fname
            if path.exists():
                d = json.load(open(path))
                print(f"{impl:<10} {seed:>6} {d['elapsed_min']:>11.1f} {d['best_cost']:>11.4f}")


def run_exp_b():
    """B) Worker scaling: BOTH codebases at varying worker counts, 5k dataset."""
    worker_counts = [1, 2, 4, 6, 8, 13]
    print("\n" + "=" * 60)
    print("EXPERIMENT B: WORKER SCALING (IV = worker count)")
    print("Fixed: 5k dataset, 42 gens, pop=100")
    print(f"Workers: {worker_counts} | Seeds: {SEEDS}")
    print("Both codebases tested at each worker count")
    print("=" * 60)

    for w in worker_counts:
        for seed in SEEDS:
            # Ours
            fname = f"expB_ours_5k_{w}w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip: {fname}")
            else:
                print(f"  Ours workers={w} seed={seed}...")
                r = run_ours("5k", seed, w)
                save(r, fname)
                print(f"    {r['elapsed_min']:.1f} min | cost={r['best_cost']:.4f}")

            # Thomas (patched to same worker count)
            fname = f"expB_thomas_5k_{w}w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip: {fname}")
            else:
                print(f"  Thomas workers={w} seed={seed}...")
                r = run_thomas("5k", seed, workers=w)
                save(r, fname)
                print(f"    {r['elapsed_min']:.1f} min | cost={r['best_cost']:.4f}")

    # Summary
    print(f"\n{'':>8} {'--- Ours ---':>30} {'--- Thomas ---':>30}")
    print(f"{'Workers':>8} {'Time (s)':>10} {'Speedup':>9} {'Eff':>7} "
          f"{'Time (s)':>10} {'Speedup':>9} {'Eff':>7}")
    print("-" * 68)
    ours_baseline = None
    thomas_baseline = None
    for w in worker_counts:
        ours_times, thomas_times = [], []
        for seed in SEEDS:
            p = OUTPUT_DIR / f"expB_ours_5k_{w}w_seed{seed}.json"
            if p.exists():
                ours_times.append(json.load(open(p))["elapsed_sec"])
            p = OUTPUT_DIR / f"expB_thomas_5k_{w}w_seed{seed}.json"
            if p.exists():
                thomas_times.append(json.load(open(p))["elapsed_sec"])

        o_avg = np.mean(ours_times) if ours_times else 0
        t_avg = np.mean(thomas_times) if thomas_times else 0
        if w == 1:
            ours_baseline = o_avg
            thomas_baseline = t_avg
        o_sp = ours_baseline / o_avg if o_avg else 0
        t_sp = thomas_baseline / t_avg if t_avg else 0
        o_eff = o_sp / w * 100
        t_eff = t_sp / w * 100
        print(f"{w:>8} {o_avg:>10.0f} {o_sp:>8.2f}x {o_eff:>6.0f}% "
              f"{t_avg:>10.0f} {t_sp:>8.2f}x {t_eff:>6.0f}%")


def run_exp_c():
    """C) Dataset scaling: our code on all datasets, 13 workers."""
    datasets = ["5k", "10k", "20k", "30k"]
    print("\n" + "=" * 60)
    print("EXPERIMENT C: DATASET SCALING (IV = dataset)")
    print("Fixed: our code, 13 workers, Table VIII gen counts, pop=100")
    print(f"Datasets: {datasets} | Seeds: {SEEDS}")
    print("=" * 60)

    for ds in datasets:
        for seed in SEEDS:
            fname = f"expC_ours_{ds}_13w_seed{seed}.json"
            if (OUTPUT_DIR / fname).exists():
                print(f"  Skip: {fname}")
                continue
            print(f"  {ds} seed={seed}...")
            r = run_ours(ds, seed, 13)
            save(r, fname)
            print(f"    {r['elapsed_min']:.1f} min | cost={r['best_cost']:.4f}")

    # Summary
    print(f"\n{'Dataset':>8} {'Gens':>5} {'Mean Time (min)':>16} {'Mean Cost':>11} {'Thomas Cost':>12}")
    print("-" * 56)
    for ds in datasets:
        times, costs = [], []
        for seed in SEEDS:
            path = OUTPUT_DIR / f"expC_ours_{ds}_13w_seed{seed}.json"
            if path.exists():
                d = json.load(open(path))
                times.append(d["elapsed_min"])
                costs.append(d["best_cost"])
        if times:
            ref = TABLE_VIII[ds]
            print(f"{ds:>8} {ref['gens']:>5} {np.mean(times):>16.1f} "
                  f"{np.mean(costs):>11.4f} {ref['cost']:>12.4f}")


if __name__ == '__main__':
    exp = sys.argv[sys.argv.index('--exp') + 1] if '--exp' in sys.argv else 'all'

    # JIT warmup
    print("JIT warmup...")
    from polymer_growth.core.simulation import _simulate_fast, SimulationParams
    _simulate_fast(SimulationParams(10, 100, 10000, 0.5, 0.001, 0.5, 0.5, 0.5, 0.5, True), 0)
    print("JIT ready.\n")

    if exp in ('A', 'a', 'all'):
        run_exp_a()
    if exp in ('B', 'b', 'all'):
        run_exp_b()
    if exp in ('C', 'c', 'all'):
        run_exp_c()

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
