#!/usr/bin/env python3
"""Master script: runs all thesis experiments in sequence.

Runs experiments in priority order, skipping any that already have
results. Safe to re-run -- idempotent.

Usage:
    nohup /usr/local/bin/python3 scripts/experiments/run_all_final.py > run_all.log 2>&1 &
"""
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_script(name, description, est_minutes):
    """Run an experiment script, print status."""
    script = SCRIPT_DIR / name
    if not script.exists():
        print(f"  MISSING: {name}")
        return False

    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  Script: {name}")
    print(f"  Estimated: ~{est_minutes} min")
    eta = datetime.now() + timedelta(minutes=est_minutes)
    print(f"  ETA: {eta.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(SCRIPT_DIR),
    )
    elapsed = (time.time() - start) / 60
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  >> {name}: {status} ({elapsed:.1f} min)")
    return result.returncode == 0


def main():
    start_time = time.time()
    print("=" * 60)
    print("THESIS EXPERIMENTS - MASTER RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    experiments = [
        # (script, description, est_minutes)
        ("exp_determinism.py",
         "Exp 4: Determinism verification (quick sanity check)", 3),
        ("generate_swe_report.py",
         "Exp 6: SWE metrics report (instant)", 1),
        ("exp_numba_isolation.py",
         "Exp 2: Numba JIT impact isolation", 120),
        ("exp_scaling_full.py",
         "Exp 3: Worker scaling at production config", 180),
        ("exp_convergence_depth.py",
         "Exp 5: Convergence depth (extended gens)", 30),
        ("exp_thomas_endtoend.py",
         "Exp 1: Thomas end-to-end speed (LONG)", 600),
    ]

    total_est = sum(e[2] for e in experiments)
    print(f"\nTotal estimated runtime: ~{total_est / 60:.0f} hours")
    print(f"Experiments: {len(experiments)}\n")

    results = {}
    for script, desc, est in experiments:
        ok = run_script(script, desc, est)
        results[script] = "PASS" if ok else "FAIL"

    # --- Final summary ---
    total_min = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for script, status in results.items():
        print(f"  {status:4s} | {script}")
    print(f"\nTotal wall time: {total_min:.0f} min ({total_min/60:.1f} hours)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run analysis if all passed
    if all(v == "PASS" for v in results.values()):
        print("\nAll experiments passed. Running analysis...")
        run_script("analyze_all_results.py", "Results analysis", 1)


if __name__ == "__main__":
    main()