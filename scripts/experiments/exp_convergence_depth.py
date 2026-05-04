#!/usr/bin/env python3
"""Experiment 5: Convergence depth -- extended generations.

Runs FDDC for far more generations than Thomas could afford (200 vs 42).
Demonstrates the practical impact of our speedup: given more compute
budget, FDDC finds better solutions.

Usage:
    nohup /usr/local/bin/python3 scripts/experiments/exp_convergence_depth.py > convergence.log 2>&1 &
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

DATASET = "5k"
POP_SIZE = 100
SEED = 42
GEN_COUNTS = [42, 100, 200, 500, 1000]  # Thomas's budget -> 24x his budget


def main():
    from shared import run_fddc, THOMAS_TABLE_VIII
    ref = THOMAS_TABLE_VIII[DATASET]

    print("=" * 60)
    print("EXPERIMENT 5: CONVERGENCE DEPTH")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {DATASET} | pop={POP_SIZE} | seed={SEED}")
    print(f"Generation counts: {GEN_COUNTS}")
    print(f"Thomas's best cost at 42 gens: {ref['fddc_cost']}")
    print("=" * 60)
    print("Estimated runtime: ~30 min (200-gen run dominates)\n")

    results = []
    for gens in GEN_COUNTS:
        out_file = OUTPUT_DIR / f"exp5_convergence_{gens}gen.json"
        if out_file.exists():
            print(f"  Loading existing: {out_file.name}")
            with open(out_file) as f:
                results.append(json.load(f))
            continue

        print(f"  Running {gens} generations...")
        result = run_fddc(
            dataset_key=DATASET,
            gen_count=gens,
            pop_size=POP_SIZE,
            impl="new",
            seed=SEED,
        )
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_file.name} | {result['elapsed_min']:.1f} min | "
              f"cost={result['best_cost']:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Gens':>6} {'Best Cost':>12} {'Time (min)':>12} {'vs Thomas':>12}")
    print("-" * 45)
    for r in results:
        vs = r["best_cost"] / ref["fddc_cost"] * 100
        key = "elapsed_min" if "elapsed_min" in r else "total_time_min"
        print(f"{r['gens']:>6} {r['best_cost']:>12.4f} {r[key]:>12.1f} {vs:>11.1f}%")

    print(f"\nThomas reference: {ref['fddc_cost']:.4f} at 42 gens")
    if len(results) >= 2:
        improvement = (results[0]["best_cost"] - results[-1]["best_cost"])
        pct = improvement / results[0]["best_cost"] * 100
        print(f"Improvement from {GEN_COUNTS[0]} to {GEN_COUNTS[-1]} gens: "
              f"{pct:.1f}% cost reduction")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()