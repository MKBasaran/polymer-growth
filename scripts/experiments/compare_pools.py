#!/usr/bin/env python3
"""Compare ThreadPool vs ProcessPool with BLAS threads disabled. ~3 min."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import numpy as np
from multiprocessing.pool import Pool
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
try: mp.set_start_method('fork', force=True)
except: pass

from polymer_growth.core.simulation import SimulationParams, simulate, _compute_vampiric_success_prob
_compute_vampiric_success_prob(np.array([1.0,2.0]), np.array([1.0,2.0]), 0.5,0.5,0.5,0.5)

PARAMS = SimulationParams(1000, 50000, 3000000, 0.72, 0.000084, 0.73, 0.41, 0.75, 0.24, True)

def run_one(seed):
    return float(np.mean(simulate(PARAMS, np.random.default_rng(seed)).all_chains()))

if __name__ == '__main__':
    N = 40
    seeds = list(range(N))

    t = time.time()
    for s in seeds: run_one(s)
    seq = time.time() - t

    print(f"BLAS=1, {N} sims, seq={seq:.1f}s")
    print(f"{'Method':<22} {'Time':>6} {'Speed':>7} {'Eff':>5}")
    print("-" * 42)

    for nw in [4, 8, 13]:
        t = time.time()
        with ThreadPoolExecutor(nw) as ex: list(ex.map(run_one, seeds))
        e = time.time() - t
        print(f"{'Thread('+str(nw)+')':<22} {e:>5.1f}s {seq/e:>6.2f}x {seq/e/nw*100:>4.0f}%")

    for nw in [4, 8, 13]:
        t = time.time()
        with Pool(nw) as p: p.map(run_one, seeds)
        e = time.time() - t
        print(f"{'Process('+str(nw)+')':<22} {e:>5.1f}s {seq/e:>6.2f}x {seq/e/nw*100:>4.0f}%")
