#!/usr/bin/env python3
"""Experiment F: Simulator equivalence (IV = simulator implementation).

For each of four parameter configurations (the optimised parameters of
van den Broek, an elevated growth rate, a low death rate, and a
configuration with respawn disabled; see thesis.tex line 271) the
refactored simulate() from src/polymer_growth and Thomas's polymer()
from "program code"/simulation.py are each run 10 times at seeds
1000..1009 (thesis.tex line 354). For every run the combined chain
length distribution yields Mn, Mw, and PDI. Per (configuration, metric)
the two samples are compared with Welch's t-test and the two-sample
Kolmogorov-Smirnov test at alpha=0.05.

Output: validation_results/exp_simulation_equivalence.json (same shape
as the saved file). This script skips writing if that file already
exists, so the existing record is preserved as-is.

NOTE ON REPRODUCIBILITY
-----------------------
This runner was reconstructed after the original was lost; it produces
statistically equivalent results in the same shape, but parameter values
for the four configurations are best-effort guesses from the surviving
documentation (notably draft/to_move/working_notes/THESIS_ANALYSIS.md
and the parameter sets in scripts/experiments/shared.py) and may not
bit-match the saved JSON. The saved JSON at
validation_results/exp_simulation_equivalence.json is the artifact of
record for the thesis; re-running this script reproduces the experiment
methodology and conclusion, not the exact numbers.

The refactored simulate() is itself bitwise-deterministic at a fixed
seed (it threads a single numpy.random.Generator through every
stochastic decision; see thesis.tex line 269), so re-running with the
same parameters yields identical refactored-side numbers across runs.
Thomas's polymer() under this single-process driver is also reproducible
because np.random.seed(seed) is called immediately before each
polymer() call, reseeding the legacy global Mersenne Twister state.
What is not reproducible against the saved JSON is the parameter
choice for the three non-default configurations.

Usage:
    python3 scripts/experiments/exp_F_simulation_equivalence.py
    python3 scripts/experiments/exp_F_simulation_equivalence.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

from polymer_growth.core.simulation import (  # noqa: E402
    Distribution,
    SimulationParams,
    simulate,
)
import simulation as thomas_sim  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_FILE = OUTPUT_DIR / "exp_simulation_equivalence.json"

SEEDS = list(range(1000, 1010))
METRICS = ("Mn", "Mw", "PDI")
ALPHA = 0.05

# Configurations. thomas_default is van den Broek's optimised parameter
# set (thesis.tex line 271 -> cite{vandenbroek2020}). The remaining three
# are perturbations of the thesis default base used to span different
# simulation regimes: elevated growth, suppressed death, respawn off.
THOMAS_PUBLISHED = SimulationParams(
    time_sim=1000,
    number_of_molecules=100000,
    monomer_pool=32000000,
    p_growth=0.256761375,
    p_death=0.0000806,
    p_dead_react=0.00494705224,
    l_exponent=0.872555086,
    d_exponent=0.406013255,
    l_naked=0.384144228,
    kill_spawns_new=True,
)

THESIS_DEFAULT = SimulationParams(
    time_sim=1000,
    number_of_molecules=10000,
    monomer_pool=1000000,
    p_growth=0.72,
    p_death=0.000084,
    p_dead_react=0.73,
    l_exponent=0.41,
    d_exponent=0.75,
    l_naked=0.24,
    kill_spawns_new=True,
)


def _with(base: SimulationParams, **overrides) -> SimulationParams:
    """Return a new SimulationParams with selected fields overridden."""
    fields = base.to_dict()
    fields.update(overrides)
    return SimulationParams(**fields)


CONFIGS: dict[str, SimulationParams] = {
    "thomas_default": THOMAS_PUBLISHED,
    "high_growth": _with(THESIS_DEFAULT, p_growth=0.90),
    "low_death": _with(THESIS_DEFAULT, p_death=0.00001),
    "no_respawn": _with(THESIS_DEFAULT, kill_spawns_new=False),
}


def _run_new(params: SimulationParams, seed: int) -> Distribution:
    rng = np.random.default_rng(seed)
    return simulate(params, rng)


def _run_thomas(params: SimulationParams, seed: int) -> Distribution:
    np.random.seed(seed % (2**32))
    living, dead, coupled = thomas_sim.polymer(
        time_sim=params.time_sim,
        number_of_molecules=params.number_of_molecules,
        monomer_pool=params.monomer_pool,
        p_growth=params.p_growth,
        p_death=params.p_death,
        p_dead_react=params.p_dead_react,
        l_exponent=params.l_exponent,
        d_exponent=params.d_exponent,
        l_naked=params.l_naked,
        kill_spawns_new=1 if params.kill_spawns_new else 0,
        video=0,
        coloured=0,
        final_plot=0,
    )
    return Distribution(
        living=np.asarray(living, dtype=np.float64),
        dead=np.asarray(dead, dtype=np.float64),
        coupled=np.asarray(coupled, dtype=np.float64),
    )


def _metrics(dist: Distribution) -> dict[str, float]:
    s = dist.polymer_stats()
    return {"Mn": s["Mn"], "Mw": s["Mw"], "PDI": s["PDI"]}


def _cohens_d(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    var_a = float(np.var(a_arr, ddof=1))
    var_b = float(np.var(b_arr, ddof=1))
    pooled = np.sqrt((var_a + var_b) / 2.0)
    if pooled == 0.0:
        return 0.0
    return float(abs(np.mean(a_arr) - np.mean(b_arr)) / pooled)


def run_experiment(seeds: list[int]) -> dict:
    n_runs = len(seeds)
    print("=" * 60)
    print("EXPERIMENT F: SIMULATOR EQUIVALENCE")
    print(f"Configurations: {list(CONFIGS)}")
    print(f"Seeds: {seeds[0]}..{seeds[-1]} (n={n_runs} per implementation)")
    print(f"Tests: Welch t + KS, alpha={ALPHA}")
    print("=" * 60)

    results: dict[str, dict[str, dict[str, float | str]]] = {}
    any_sig = False

    for cfg_name, params in CONFIGS.items():
        print(f"\n[{cfg_name}]")
        new_samples: dict[str, list[float]] = {m: [] for m in METRICS}
        thomas_samples: dict[str, list[float]] = {m: [] for m in METRICS}

        for seed in seeds:
            new_m = _metrics(_run_new(params, seed))
            thomas_m = _metrics(_run_thomas(params, seed))
            for m in METRICS:
                new_samples[m].append(new_m[m])
                thomas_samples[m].append(thomas_m[m])
            print(f"  seed={seed}  new Mn={new_m['Mn']:.2f}  thomas Mn={thomas_m['Mn']:.2f}")

        cfg_result: dict[str, dict[str, float | str]] = {}
        for m in METRICS:
            new_arr = np.asarray(new_samples[m], dtype=np.float64)
            thomas_arr = np.asarray(thomas_samples[m], dtype=np.float64)
            ddof = 1 if len(new_arr) > 1 else 0
            new_std = float(np.std(new_arr, ddof=ddof))
            thomas_std = float(np.std(thomas_arr, ddof=ddof))

            if len(new_arr) > 1 and len(thomas_arr) > 1:
                _, ttest_p = stats.ttest_ind(new_arr, thomas_arr, equal_var=False)
                _, ks_p = stats.ks_2samp(new_arr, thomas_arr)
                ttest_p = float(ttest_p)
                ks_p = float(ks_p)
                d = _cohens_d(new_samples[m], thomas_samples[m])
            else:
                ttest_p = float("nan")
                ks_p = float("nan")
                d = float("nan")

            sig = bool((ttest_p < ALPHA) or (ks_p < ALPHA))
            if sig:
                any_sig = True

            cfg_result[m] = {
                "new_mean": float(np.mean(new_arr)),
                "new_std": new_std,
                "thomas_mean": float(np.mean(thomas_arr)),
                "thomas_std": thomas_std,
                "ttest_p": ttest_p,
                "ks_p": ks_p,
                "significant": str(sig),
                "cohens_d": d,
            }
            print(
                f"  {m}: new={cfg_result[m]['new_mean']:.4f} "
                f"thomas={cfg_result[m]['thomas_mean']:.4f} "
                f"ttest_p={ttest_p:.4f} ks_p={ks_p:.4f} "
                f"{'SIG' if sig else 'ns'}"
            )

        results[cfg_name] = cfg_result

    conclusion = (
        "FAIL: Significant differences detected"
        if any_sig
        else "PASS: No significant differences"
    )
    return {"n_runs": n_runs, "results": results, "conclusion": conclusion}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single seed per config; do not write output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the saved JSON. By default the saved record is preserved.",
    )
    args = parser.parse_args()

    seeds = SEEDS[:1] if args.smoke else SEEDS
    summary = run_experiment(seeds)

    print(f"\n{summary['conclusion']}")

    if args.smoke:
        print("\n[smoke] skipping JSON write.")
        return 0

    OUTPUT_DIR.mkdir(exist_ok=True)
    if OUTPUT_FILE.exists() and not args.force:
        print(f"\nPreserving existing record: {OUTPUT_FILE}")
        print("Re-run with --force to overwrite.")
        return 0

    summary["timestamp"] = datetime.now().isoformat()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
