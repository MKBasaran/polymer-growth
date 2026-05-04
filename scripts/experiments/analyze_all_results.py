#!/usr/bin/env python3
"""Analyze all experiment results and produce thesis-ready summary.

Reads all exp*.json files from validation_results/ and generates
a consolidated summary table.

Usage:
    /usr/local/bin/python3 scripts/experiments/analyze_all_results.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "validation_results"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def analyze_exp1_speed():
    """Experiment 1: Thomas end-to-end speed comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: END-TO-END SPEED COMPARISON")
    print("=" * 70)

    datasets = ["5k", "10k", "20k", "30k"]
    rows = []
    for ds in datasets:
        thomas_f = OUTPUT_DIR / f"exp1_thomas_endtoend_{ds}.json"
        ours_6w_f = OUTPUT_DIR / f"exp1_ours_6w_{ds}_seed42.json"
        ours_full_f = OUTPUT_DIR / f"exp1_ours_fullw_{ds}_seed42.json"

        row = {"dataset": ds}
        for label, path, time_key in [
            ("thomas", thomas_f, "total_time_min"),
            ("ours_6w", ours_6w_f, "elapsed_min"),
            ("ours_full", ours_full_f, "elapsed_min"),
        ]:
            if path.exists():
                d = load_json(path)
                row[f"{label}_time"] = d.get(time_key, d.get("total_time_min"))
                row[f"{label}_cost"] = d.get("best_cost")
            else:
                row[f"{label}_time"] = None
                row[f"{label}_cost"] = None
        rows.append(row)

    print(f"\n{'Dataset':>8} {'Thomas (min)':>13} {'Ours 6w (min)':>14} "
          f"{'Speedup 6w':>11} {'Ours full (min)':>16} {'Speedup full':>13}")
    print("-" * 80)
    for r in rows:
        t = r.get("thomas_time")
        o6 = r.get("ours_6w_time")
        of = r.get("ours_full_time")
        s6 = f"{t/o6:.1f}x" if t and o6 else "N/A"
        sf = f"{t/of:.1f}x" if t and of else "N/A"
        print(f"{r['dataset']:>8} {t or 'N/A':>13} {o6 or 'N/A':>14} "
              f"{s6:>11} {of or 'N/A':>16} {sf:>13}")


def analyze_exp2_numba():
    """Experiment 2: Numba JIT impact."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: NUMBA JIT IMPACT")
    print("=" * 70)

    datasets = ["5k", "10k", "20k", "30k"]
    seeds = [42, 123, 777]

    print(f"\n{'Dataset':>8} {'Numba ON (s)':>13} {'Numba OFF (s)':>14} {'Difference':>11}")
    print("-" * 50)
    for ds in datasets:
        on_times, off_times = [], []
        for seed in seeds:
            for label, times_list in [("enabled", on_times), ("disabled", off_times)]:
                path = OUTPUT_DIR / f"exp2_numba_{ds}_{label}_seed{seed}.json"
                if path.exists():
                    times_list.append(load_json(path)["elapsed_sec"])
        if on_times and off_times:
            on_avg = np.mean(on_times)
            off_avg = np.mean(off_times)
            diff = (on_avg - off_avg) / off_avg * 100
            sign = "+" if diff > 0 else ""
            print(f"{ds:>8} {on_avg:>13.0f} {off_avg:>14.0f} {sign}{diff:>10.1f}%")
        else:
            print(f"{ds:>8} {'N/A':>13} {'N/A':>14} {'N/A':>11}")


def analyze_exp3_scaling():
    """Experiment 3: Worker scaling."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: WORKER SCALING (PRODUCTION CONFIG)")
    print("=" * 70)

    workers = [1, 2, 4, 6, 8, 13]
    seeds = [42, 123, 777]

    print(f"\n{'Workers':>8} {'Mean (s)':>10} {'Speedup':>9} {'Efficiency':>11}")
    print("-" * 42)
    baseline = None
    for w in workers:
        times = []
        for seed in seeds:
            path = OUTPUT_DIR / f"exp3_scaling_{w}w_seed{seed}.json"
            if path.exists():
                times.append(load_json(path)["elapsed_sec"])
        if not times:
            print(f"{w:>8} {'N/A':>10}")
            continue
        avg = np.mean(times)
        if w == 1:
            baseline = avg
        speedup = baseline / avg if baseline else 1.0
        efficiency = speedup / w * 100
        print(f"{w:>8} {avg:>10.0f} {speedup:>8.2f}x {efficiency:>10.1f}%")


def analyze_exp4_determinism():
    """Experiment 4: Determinism."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: DETERMINISM VERIFICATION")
    print("=" * 70)

    path = OUTPUT_DIR / "exp4_determinism_summary.json"
    if path.exists():
        d = load_json(path)
        verdict = "PASS" if d["identical"] else "FAIL"
        print(f"\n  Runs: {d['n_runs']} | Seed: {d['seed']}")
        print(f"  Cost histories identical: {d['identical']}")
        print(f"  Best costs: {d['best_costs']}")
        print(f"  Verdict: {verdict}")
    else:
        print("\n  No results found.")


def analyze_exp5_convergence():
    """Experiment 5: Convergence depth."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: CONVERGENCE DEPTH")
    print("=" * 70)

    gen_counts = [42, 100, 200]
    print(f"\n{'Gens':>6} {'Best Cost':>12} {'Time (min)':>12}")
    print("-" * 34)
    for g in gen_counts:
        path = OUTPUT_DIR / f"exp5_convergence_{g}gen.json"
        if path.exists():
            d = load_json(path)
            t = d.get("elapsed_min", d.get("total_time_min", 0))
            print(f"{g:>6} {d['best_cost']:>12.4f} {t:>12.1f}")
        else:
            print(f"{g:>6} {'N/A':>12} {'N/A':>12}")


def analyze_exp6_swe():
    """Experiment 6: SWE metrics."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: SWE METRICS")
    print("=" * 70)

    path = OUTPUT_DIR / "exp6_swe_report.json"
    if path.exists():
        d = load_json(path)
        t = d["thomas_code"]
        o = d["our_code"]
        ts = d["tests"]
        print(f"\n{'Metric':<25} {'Thomas':>10} {'Ours':>10}")
        print("-" * 48)
        print(f"{'Python files':<25} {t['file_count']:>10} {o['file_count']:>10}")
        print(f"{'Code lines':<25} {t['code_lines']:>10} {o['code_lines']:>10}")
        print(f"{'Avg lines/file':<25} {t['avg_lines_per_file']:>10} {o['avg_lines_per_file']:>10}")
        print(f"{'Test functions':<25} {'0':>10} {ts['test_functions']:>10}")
        print(f"{'Test files':<25} {'0':>10} {ts['test_files']:>10}")
    else:
        print("\n  No results found.")


def main():
    print("=" * 70)
    print("THESIS EXPERIMENT RESULTS - CONSOLIDATED ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    analyze_exp1_speed()
    analyze_exp2_numba()
    analyze_exp3_scaling()
    analyze_exp4_determinism()
    analyze_exp5_convergence()
    analyze_exp6_swe()

    # --- Save summary ---
    summary = {
        "generated": datetime.now().isoformat(),
        "experiments_analyzed": [
            "exp1_thomas_endtoend",
            "exp2_numba_isolation",
            "exp3_scaling_production",
            "exp4_determinism",
            "exp5_convergence_depth",
            "exp6_swe_report",
        ],
    }
    out = OUTPUT_DIR / "thesis_results_summary.json"
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSaved: {out.name}")
    print(f"Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()