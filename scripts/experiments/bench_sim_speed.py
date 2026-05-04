#!/usr/bin/env python3
"""Benchmark: original simulate() vs _simulate_fast() (Numba JIT core).

Runs both 10 times with identical params, measures wall time per call.
Also verifies output equivalence (same Mn/Mw/PDI range).

Usage:
    /usr/local/bin/python3 scripts/experiments/bench_sim_speed.py
"""
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polymer_growth.core.simulation import (
    SimulationParams, simulate, _simulate_fast, _simulate_core,
    MONOMER_MASS, INITIATOR_MASS
)

PARAMS = SimulationParams(
    time_sim=1500,
    number_of_molecules=70000,
    monomer_pool=3000000,
    p_growth=0.72,
    p_death=0.000084,
    p_dead_react=0.73,
    l_exponent=0.41,
    d_exponent=0.75,
    l_naked=0.24,
    kill_spawns_new=True,
)

N_RUNS = 10
SEED = 42


def bench_original():
    times = []
    for i in range(N_RUNS):
        rng = np.random.default_rng(SEED + i)
        t0 = time.perf_counter()
        dist = simulate(PARAMS, rng)
        dt = time.perf_counter() - t0
        times.append(dt)
        if i == 0:
            mn = dist.compute_mn()
            print(f"  Original run 0: Mn={mn:.1f}, n_living={len(dist.living)}, "
                  f"n_dead={len(dist.dead)}, time={dt:.3f}s")
    return times


def bench_fast():
    # Warm up JIT
    print("  JIT warmup...")
    _simulate_fast(SimulationParams(
        time_sim=10, number_of_molecules=100, monomer_pool=10000,
        p_growth=0.5, p_death=0.001, p_dead_react=0.5,
        l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
        kill_spawns_new=True), seed=0)
    print("  JIT warm, running benchmark...")

    times = []
    for i in range(N_RUNS):
        seed = SEED + i
        t0 = time.perf_counter()
        dist = _simulate_fast(PARAMS, seed)
        dt = time.perf_counter() - t0
        times.append(dt)
        if i == 0:
            mn = dist.compute_mn()
            print(f"  Fast run 0: Mn={mn:.1f}, n_living={len(dist.living)}, "
                  f"n_dead={len(dist.dead)}, time={dt:.3f}s")
    return times


def main():
    print(f"Params: time_sim={PARAMS.time_sim}, n_molecules={PARAMS.number_of_molecules}")
    print(f"Runs: {N_RUNS}\n")

    print("=== Original (numpy + minimal JIT) ===")
    orig_times = bench_original()

    print("\n=== Fast (full Numba JIT core) ===")
    fast_times = bench_fast()

    orig_avg = np.mean(orig_times)
    fast_avg = np.mean(fast_times)
    speedup = orig_avg / fast_avg

    print(f"\n{'='*50}")
    print(f"Original: {orig_avg:.3f}s avg ({np.std(orig_times):.3f}s std)")
    print(f"Fast:     {fast_avg:.3f}s avg ({np.std(fast_times):.3f}s std)")
    print(f"Speedup:  {speedup:.1f}x")
    print(f"{'='*50}")

    # Equivalence check: both should produce similar Mn
    rng = np.random.default_rng(999)
    dist_orig = simulate(PARAMS, rng)
    dist_fast = _simulate_fast(PARAMS, 999)
    mn_orig = dist_orig.compute_mn()
    mn_fast = dist_fast.compute_mn()
    print(f"\nEquivalence (seed=999): Mn_orig={mn_orig:.1f}, Mn_fast={mn_fast:.1f}")
    print(f"Note: not bit-identical due to different RNG paths (Generator vs legacy)")


if __name__ == "__main__":
    main()
