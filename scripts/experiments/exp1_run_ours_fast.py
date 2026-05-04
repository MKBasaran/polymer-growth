#!/usr/bin/env python3
"""Run our code with Numba JIT fast simulation."""
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

# Warm up Numba JIT before forking
print("JIT warmup...")
from polymer_growth.core.simulation import _simulate_fast, SimulationParams
_simulate_fast(SimulationParams(10, 100, 10000, 0.5, 0.001, 0.5, 0.5, 0.5, 0.5, True), 0)
print("JIT ready.")

from shared import run_fddc

workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else None
if workers is None:
    workers = max(1, (os.cpu_count() or 4) - 1)

POP_SIZE = 100
GENS = 42
SEED = 42

print(f"Our FDDC (FAST SIM) | pop={POP_SIZE} | gens={GENS} | workers={workers} | seed={SEED}")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")

result = run_fddc(
    dataset_key="5k",
    gen_count=GENS,
    pop_size=POP_SIZE,
    impl="new",
    seed=SEED,
    workers=workers,
    use_fast_sim=True,
)

out = Path(PROJECT_ROOT / "validation_results" / f"exp1_ours_fast_{workers}w_5k_seed42.json")
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nTotal: {result['elapsed_min']:.1f} min | Best: {result['best_cost']:.4f}")
print(f"Saved: {out}")
