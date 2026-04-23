#!/usr/bin/env python3
"""Fair comparison: our code at 6 workers (Thomas's worker count).
Pop=100, 42 gens, 5k no BB. ~15-20 min."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from polymer_growth.core.simulation import SimulationParams, simulate
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig

PROJECT = os.path.join(os.path.dirname(__file__), '..', '..')
_, exp = load_experimental_data(f'{PROJECT}/program code/Data/5k no BB.xlsx')
obj = MinMaxV2ObjectiveFunction(exp)

def wrapper(p, sigma=None):
    p = np.asarray(p).flatten().tolist()
    sp = SimulationParams(int(p[0]),int(p[1]),int(p[2]),p[3],p[4],p[5],p[6],p[7],p[8],bool(round(p[9])))
    return obj.compute_cost(simulate(sp, np.random.default_rng()), sigma=sigma)

bounds = np.array([[100,3000],[10000,120000],[1000000,5000000],
    [0.1,0.99],[0.0001,0.002],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,1.0],[0,1]])

if __name__ == '__main__':
    print("=" * 50)
    print("FAIR COMPARISON: Our code, Thomas's 6 workers")
    print("Config: pop=100, gen=42, 5k no BB, seed=42")
    print("Thomas result: cost=21.74, time=~24 hours")
    print("=" * 50)

    cfg = FDDCConfig(population_size=100, max_generations=42, n_workers=6,
                     memory_size=10, n_encounters=10, n_children=2,
                     mutation_rate=0.6, mutation_strength=0.001,
                     enable_fddc=True, sigma_points_per_index=4,
                     rank_selection_power=1.5)

    def progress(gen, cost):
        elapsed = time.time() - start
        rate = elapsed / gen if gen > 0 else 0
        eta = rate * (42 - gen)
        print(f"  Gen {gen:2d}/42 | Cost: {cost:.4f} | "
              f"{elapsed/60:.1f}min elapsed | ~{eta/60:.1f}min remaining")

    start = time.time()
    opt = FDDCOptimizer(bounds=bounds, objective_function=wrapper,
                        config=cfg, callback=progress)
    r = opt.optimize(seed=42)
    total = time.time() - start

    print(f"\n{'=' * 50}")
    print(f"RESULT")
    print(f"  Best cost:  {r.best_cost:.4f}")
    print(f"  Time:       {total/60:.1f} minutes")
    print(f"  Thomas:     cost=21.74, time=~24 hours")
    print(f"  Speedup:    {24*60 / (total/60):.0f}x faster (vs Thomas's hardware)")
    print(f"{'=' * 50}")

    out = f'{PROJECT}/validation_results/thomas_fair_compare.json'
    with open(out, 'w') as f:
        json.dump({"best_cost": float(r.best_cost), "elapsed_min": total/60,
                   "workers": 6, "pop": 100, "gens": 42}, f, indent=2)
    print(f"Saved: {out}")
