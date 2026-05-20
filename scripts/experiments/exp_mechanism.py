#!/usr/bin/env python3
"""Mechanism isolation: where does the 1.56x speedup actually come from?

Three tests, single-process (no Pool, no IPC):

  A) Per-call simulate() vs polymer() at fixed published parameters (n=30).
     Isolates: per-call simulation cost only.

  B) Per-call at varied parameters sampled from BOUNDS (n=30).
     Isolates: per-call cost under GA-like parameter diversity.

  C) Full FDDC at workers=1, pop=20, 3 gens, 5k dataset, both impls.
     Isolates: per-call cost + per-generation scaffolding (no parallel dispatch).

Conclusions are drawn by comparing the ratios across the three tests:

  - If A and B ratios are ~1.0 but C ratio is ~1.56 -> scaffolding is the cause.
  - If A and B are already ~1.56 -> simulate() rewrite is the cause.
  - Intermediate -> both contribute, proportionally.

Run:
    cd /Users/kaanbasaran/Desktop/thesis_try/scripts/experiments
    python3 exp_mechanism.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from polymer_growth.core.simulation import SimulationParams, simulate  # noqa: E402
import simulation as thomas_sim  # noqa: E402

# Thomas's published optimised parameter set (from simulation.py:217-218)
PUBLISHED = dict(
    time_sim=1000,
    number_of_molecules=100000,
    monomer_pool=32000000,
    p_growth=0.256761375,
    p_death=0.0000806,
    p_dead_react=0.00494705224,
    l_exponent=0.872555086,
    d_exponent=0.406013255,
    l_naked=0.384144228,
    kill_spawns_new=1,
)

# Parameter bounds from BOUNDS in shared.py
BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])


def time_ours_once(params: SimulationParams, seed: int) -> float:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    simulate(params, rng)
    return time.perf_counter() - t0


def time_thomas_once(p: dict, seed: int) -> float:
    np.random.seed(seed)
    t0 = time.perf_counter()
    thomas_sim.polymer(
        time_sim=p["time_sim"],
        number_of_molecules=p["number_of_molecules"],
        monomer_pool=p["monomer_pool"],
        p_growth=p["p_growth"],
        p_death=p["p_death"],
        p_dead_react=p["p_dead_react"],
        l_exponent=p["l_exponent"],
        d_exponent=p["d_exponent"],
        l_naked=p["l_naked"],
        kill_spawns_new=p["kill_spawns_new"],
        video=0,
        coloured=0,
        final_plot=0,
    )
    return time.perf_counter() - t0


def dict_to_simparams(p: dict) -> SimulationParams:
    return SimulationParams(
        time_sim=p["time_sim"],
        number_of_molecules=p["number_of_molecules"],
        monomer_pool=p["monomer_pool"],
        p_growth=p["p_growth"],
        p_death=p["p_death"],
        p_dead_react=p["p_dead_react"],
        l_exponent=p["l_exponent"],
        d_exponent=p["d_exponent"],
        l_naked=p["l_naked"],
        kill_spawns_new=bool(p["kill_spawns_new"]),
    )


def summarise(label: str, ours: list[float], thomas: list[float]) -> dict:
    o = np.array(ours)
    t = np.array(thomas)
    ratio = t.mean() / o.mean()
    speedup_pct = (1 - o.mean() / t.mean()) * 100
    print(f"\n{label}")
    print(f"  Ours   mean: {o.mean()*1000:7.1f} ms   std: {o.std(ddof=1)*1000:6.1f} ms   n={len(o)}")
    print(f"  Thomas mean: {t.mean()*1000:7.1f} ms   std: {t.std(ddof=1)*1000:6.1f} ms   n={len(t)}")
    print(f"  Ratio (thomas/ours): {ratio:.3f}x   (ours is {speedup_pct:+.1f}% faster)")
    return dict(
        label=label,
        ours_mean=float(o.mean()),
        ours_std=float(o.std(ddof=1)),
        thomas_mean=float(t.mean()),
        thomas_std=float(t.std(ddof=1)),
        ratio_thomas_over_ours=float(ratio),
        n=len(o),
    )


def test_a_fixed_params(n: int = 30) -> dict:
    """Per-call timing at Thomas's published parameter set."""
    params_obj = dict_to_simparams(PUBLISHED)
    # Warmup
    time_ours_once(params_obj, 0)
    time_thomas_once(PUBLISHED, 0)

    ours, thomas = [], []
    for i in range(n):
        # Alternate order to balance any thermal drift
        if i % 2 == 0:
            ours.append(time_ours_once(params_obj, 1000 + i))
            thomas.append(time_thomas_once(PUBLISHED, 1000 + i))
        else:
            thomas.append(time_thomas_once(PUBLISHED, 1000 + i))
            ours.append(time_ours_once(params_obj, 1000 + i))
    return summarise("Test A: fixed published parameters", ours, thomas)


def test_b_varied_params(n: int = 30) -> dict:
    """Per-call timing at varied parameters sampled from BOUNDS."""
    rng = np.random.default_rng(20260601)
    # Warmup
    time_ours_once(dict_to_simparams(PUBLISHED), 0)
    time_thomas_once(PUBLISHED, 0)

    ours, thomas = [], []
    for i in range(n):
        # Sample params from BOUNDS
        sampled = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        p = dict(
            time_sim=int(sampled[0]),
            number_of_molecules=int(sampled[1]),
            monomer_pool=int(sampled[2]),
            p_growth=float(sampled[3]),
            p_death=float(sampled[4]),
            p_dead_react=float(sampled[5]),
            l_exponent=float(sampled[6]),
            d_exponent=float(sampled[7]),
            l_naked=float(sampled[8]),
            kill_spawns_new=int(round(sampled[9])),
        )
        params_obj = dict_to_simparams(p)
        seed = 2000 + i
        if i % 2 == 0:
            ours.append(time_ours_once(params_obj, seed))
            thomas.append(time_thomas_once(p, seed))
        else:
            thomas.append(time_thomas_once(p, seed))
            ours.append(time_ours_once(params_obj, seed))
    return summarise("Test B: varied parameters from BOUNDS", ours, thomas)


def test_c_full_fddc_serial(pop: int = 20, gens: int = 3) -> dict:
    """Full FDDC at workers=1, pop=20, 3 gens. Both impls. Single process.

    Wall-clock at workers=1 captures: per-call sim cost + per-generation
    scaffolding (rank, select, crossover, mutate, memory updates). No pool.map.
    """
    from shared import run_fddc

    print(f"\nTest C: full FDDC, workers=1, pop={pop}, gens={gens}, 5k dataset")
    print("  (this takes a few minutes; running both impls)")

    # Warmup not needed (the run_fddc call itself dominates)
    t0 = time.perf_counter()
    r_ours = run_fddc("5k", gen_count=gens, pop_size=pop, impl="new",
                     seed=42, workers=1, use_fast_sim=False)
    ours_total = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_thomas = run_fddc("5k", gen_count=gens, pop_size=pop, impl="thomas",
                       seed=42, workers=1, use_fast_sim=False)
    thomas_total = time.perf_counter() - t0

    ratio = thomas_total / ours_total
    speedup_pct = (1 - ours_total / thomas_total) * 100
    print(f"  Ours   total: {ours_total:7.2f} s   ({ours_total/gens:.2f} s/gen)")
    print(f"  Thomas total: {thomas_total:7.2f} s   ({thomas_total/gens:.2f} s/gen)")
    print(f"  Ratio (thomas/ours): {ratio:.3f}x   (ours is {speedup_pct:+.1f}% faster)")
    return dict(
        label=f"Test C: full FDDC workers=1 pop={pop} gens={gens}",
        ours_total=ours_total,
        thomas_total=thomas_total,
        ratio_thomas_over_ours=ratio,
        gens=gens,
        pop=pop,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("MECHANISM ISOLATION: where does the 1.56x come from?")
    print("=" * 60)
    print("Single-process, no Pool. Compare ours vs Thomas.")
    print(f"NumPy:  {np.__version__}")
    print(f"Python: {sys.version.split()[0]}")

    results = {}
    results["test_a"] = test_a_fixed_params(n=30)
    results["test_b"] = test_b_varied_params(n=30)
    results["test_c"] = test_c_full_fddc_serial(pop=20, gens=3)

    # Save raw results
    out_path = PROJECT_ROOT / "validation_results" / "mechanism_analysis.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    a_ratio = results["test_a"]["ratio_thomas_over_ours"]
    b_ratio = results["test_b"]["ratio_thomas_over_ours"]
    c_ratio = results["test_c"]["ratio_thomas_over_ours"]
    print(f"  Test A (fixed params, single call): {a_ratio:.3f}x")
    print(f"  Test B (varied params, single call): {b_ratio:.3f}x")
    print(f"  Test C (full FDDC, workers=1): {c_ratio:.3f}x")
    print(f"  Headline (full FDDC, workers=6): 1.56x (from Exp A, n=30)")
    print()
    if a_ratio < 1.10 and c_ratio > 1.30:
        print("  Conclusion: per-call sim() is roughly equal; the speedup")
        print("  comes from FDDC scaffolding / per-generation overhead.")
    elif a_ratio > 1.30:
        print("  Conclusion: the speedup comes from simulate() rewrite alone.")
    elif a_ratio > 1.10 and c_ratio > 1.30:
        gap = c_ratio - a_ratio
        print(f"  Conclusion: both contribute. simulate() rewrite gives ~{a_ratio:.2f}x,")
        print(f"  scaffolding adds another ~{gap:.2f}x.")
    print(f"\nRaw JSON: {out_path}")
