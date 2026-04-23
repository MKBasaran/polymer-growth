#!/usr/bin/env python3
"""Test ProcessPool vs ThreadPool for simulation parallelism."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from polymer_growth.core.simulation import SimulationParams, simulate

PARAMS = SimulationParams(
    time_sim=1000, number_of_molecules=50000, monomer_pool=3000000,
    p_growth=0.72, p_death=0.000084, p_dead_react=0.73,
    l_exponent=0.41, d_exponent=0.75, l_naked=0.24, kill_spawns_new=True
)

def run_one_process(seed):
    """Module-level function for multiprocessing.Pool."""
    rng = np.random.default_rng(seed)
    dist = simulate(PARAMS, rng)
    return float(np.mean(dist.all_chains()))

if __name__ == '__main__':
    N = 20
    seeds = list(range(N))

    # Sequential
    t = time.time()
    for s in seeds:
        run_one_process(s)
    seq = time.time() - t
    print(f"Sequential:      {seq:.1f}s  1.00x")

    # ThreadPool
    for nw in [4, 8, 13]:
        t = time.time()
        with ThreadPoolExecutor(max_workers=nw) as ex:
            list(ex.map(run_one_process, seeds))
        elapsed = time.time() - t
        print(f"ThreadPool({nw:2d}):  {elapsed:.1f}s  {seq/elapsed:.2f}x")

    # multiprocessing.Pool (fork-based, no pickle for globals)
    for nw in [4, 8, 13]:
        t = time.time()
        with Pool(nw) as p:
            p.map(run_one_process, seeds)
        elapsed = time.time() - t
        print(f"ProcessPool({nw:2d}): {elapsed:.1f}s  {seq/elapsed:.2f}x")