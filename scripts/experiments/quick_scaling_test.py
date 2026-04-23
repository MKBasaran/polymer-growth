#!/usr/bin/env python3
"""Quick scaling test: pop=20, 5 gens, 1 seed. ~5 min total."""
import sys, time, os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polymer_growth.core.simulation import SimulationParams, simulate
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig

_, exp = load_experimental_data(str(PROJECT_ROOT / "program code/Data/5k no BB.xlsx"))
obj = MinMaxV2ObjectiveFunction(exp)

def wrapper(p, sigma=None):
    p = np.asarray(p).flatten().tolist()
    sp = SimulationParams(int(p[0]),int(p[1]),int(p[2]),p[3],p[4],p[5],p[6],p[7],p[8],bool(round(p[9])))
    return obj.compute_cost(simulate(sp, np.random.default_rng()), sigma=sigma)

bounds = np.array([[100,3000],[10000,120000],[1000000,5000000],
    [0.1,0.99],[0.0001,0.002],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,1.0],[0,1]])

worker_counts = [1, 2, 4, 6, 8, os.cpu_count() - 1]

print(f"Quick scaling: pop=20, gens=5, seed=42")
print(f"{'Workers':>8} {'Time':>8} {'Speedup':>8}")
print("-" * 28)

baseline = None
for nw in worker_counts:
    cfg = FDDCConfig(population_size=20, max_generations=5, n_workers=nw,
                     memory_size=10, n_encounters=10, n_children=2,
                     mutation_rate=0.6, mutation_strength=0.001,
                     enable_fddc=True, sigma_points_per_index=4, rank_selection_power=1.5)
    t = time.time()
    opt = FDDCOptimizer(bounds=bounds, objective_function=wrapper, config=cfg)
    opt.optimize(seed=42)
    elapsed = time.time() - t
    if baseline is None:
        baseline = elapsed
    print(f"{nw:>8} {elapsed:>7.1f}s {baseline/elapsed:>7.2f}x")