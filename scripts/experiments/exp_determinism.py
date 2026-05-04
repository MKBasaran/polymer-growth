#!/usr/bin/env python3
"""Experiment 4: Determinism verification.

Runs the same config 3 times with identical seed=42 and verifies
that cost histories are bit-identical. Proves the eval_seed fix
produces fully reproducible optimization.

Quick experiment -- pop=50, 10 gens on 5k (~3 min total).

Usage:
    /usr/local/bin/python3 scripts/experiments/exp_determinism.py
"""
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_RUNS = 3
SEED = 42
DATASET = "5k"
POP_SIZE = 50
GENS = 10


def main():
    from shared import run_fddc

    print("=" * 60)
    print("EXPERIMENT 4: DETERMINISM VERIFICATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {DATASET} | pop={POP_SIZE} | gens={GENS} | seed={SEED}")
    print(f"Runs: {N_RUNS} (must produce identical cost histories)")
    print("=" * 60)
    print("Estimated runtime: ~3 min\n")

    results = []
    for i in range(N_RUNS):
        out_file = OUTPUT_DIR / f"exp4_determinism_run{i+1}.json"
        if out_file.exists():
            print(f"  Loading existing: {out_file.name}")
            with open(out_file) as f:
                results.append(json.load(f))
            continue

        print(f"  Run {i+1}/{N_RUNS}...")
        result = run_fddc(
            dataset_key=DATASET,
            gen_count=GENS,
            pop_size=POP_SIZE,
            impl="new",
            seed=SEED,
        )
        result["run_number"] = i + 1
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file.name} | cost={result['best_cost']:.6f}")

    # --- Verify determinism ---
    print(f"\n{'='*60}")
    print("DETERMINISM CHECK")
    print(f"{'='*60}")

    if len(results) < 2:
        print("  Not enough runs to compare. Need at least 2.")
        return

    ref_history = results[0]["cost_history"]
    ref_params = results[0]["best_params"]
    all_identical = True

    for i in range(1, len(results)):
        h = results[i]["cost_history"]
        p = results[i]["best_params"]

        history_match = ref_history == h
        params_match = ref_params == p

        if history_match and params_match:
            print(f"  Run 1 vs Run {i+1}: IDENTICAL")
        else:
            all_identical = False
            if not history_match:
                diffs = [(j, ref_history[j], h[j])
                         for j in range(min(len(ref_history), len(h)))
                         if ref_history[j] != h[j]]
                print(f"  Run 1 vs Run {i+1}: COST HISTORY DIFFERS at {len(diffs)} generations")
                for gen, a, b in diffs[:5]:
                    print(f"    Gen {gen}: {a} vs {b}")
            if not params_match:
                print(f"  Run 1 vs Run {i+1}: BEST PARAMS DIFFER")
                for j, (a, b) in enumerate(zip(ref_params, p)):
                    if a != b:
                        print(f"    Param {j}: {a} vs {b}")

    # Final verdict
    summary = {
        "experiment": "determinism",
        "seed": SEED,
        "n_runs": N_RUNS,
        "identical": all_identical,
        "cost_histories": [r["cost_history"] for r in results],
        "best_costs": [r["best_cost"] for r in results],
        "best_params": [r["best_params"] for r in results],
        "timestamp": datetime.now().isoformat(),
    }
    summary_file = OUTPUT_DIR / "exp4_determinism_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    verdict = "PASS -- all runs identical" if all_identical else "FAIL -- runs differ"
    print(f"\n  VERDICT: {verdict}")
    print(f"  Saved: {summary_file.name}")
    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()