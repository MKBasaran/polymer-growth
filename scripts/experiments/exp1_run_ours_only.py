#!/usr/bin/env python3
"""Run ONLY our refactored code end-to-end. Nothing else.

Usage:
    /usr/local/bin/python3 scripts/experiments/exp1_run_ours_only.py
    /usr/local/bin/python3 scripts/experiments/exp1_run_ours_only.py --workers 6
"""
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

from shared import run_fddc

workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else None
if workers is None:
    workers = max(1, (os.cpu_count() or 4) - 1)

POP_SIZE = 100
GENS = 42
SEED = 42

print(f"Our FDDC | pop={POP_SIZE} | gens={GENS} | workers={workers} | seed={SEED}")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

result = run_fddc(
    dataset_key="5k",
    gen_count=GENS,
    pop_size=POP_SIZE,
    impl="new",
    seed=SEED,
    workers=workers,
)
result["implementation"] = f"ours_{workers}w"
result["timestamp"] = datetime.now().isoformat()

tag = f"ours_{workers}w"
out = Path(PROJECT_ROOT / "validation_results" / f"exp1_{tag}_5k_seed42.json")
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nTotal: {result['elapsed_min']:.1f} min | Best: {result['best_cost']:.4f}")
print(f"Saved: {out}")