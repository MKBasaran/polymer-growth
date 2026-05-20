"""Subprocess-isolated runner for Thomas and Ours experiments.
Each run executes in a fresh process — zero shared state."""
import subprocess
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)
PYTHON = sys.executable

TABLE_VIII = {
    "5k":  {"file": "5k no BB.xlsx",  "gens": 42, "cost": 21.7358},
    "10k": {"file": "10k no BB.xlsx", "gens": 56, "cost": 108.5352},
    "20k": {"file": "20k no BB.xlsx", "gens": 44, "cost": 131.7344},
    "30k": {"file": "30k no BB.xlsx", "gens": 27, "cost": 340.2589},
}

_THOMAS_TEMPLATE = '''
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import matplotlib; matplotlib.use('Agg')
import sys, time, io, json, warnings
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore', category=SyntaxWarning)

PROJECT_ROOT = Path("{project_root}")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

import simulation as thomas_sim
from distributionComparison import min_maxV2
import fddc as thomas_fddc_module
import matplotlib.pyplot as plt

BOUNDS = np.array([
    [100, 3000], [10000, 120000], [1000000, 5000000],
    [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
    [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
])

def _noop(*a, **kw): pass

data_path = str(PROJECT_ROOT / "program code/Data/{datafile}")
norm = min_maxV2(data_path, thomas_sim.polymer, sigma=[1]*6, transfac=1)
plt.close(norm.fig)
norm.fig = None; norm.ax0 = None; norm.ax1 = None; norm.ax2 = None
norm.plotDistributions = _noop

with redirect_stdout(io.StringIO()):
    fc = thomas_fddc_module.fddc(bounds=BOUNDS, fitnessFunction=norm.costFunction,
                                 distribution_comparison=norm, populationSize=100,
                                 graph=False, ui_plot=False)

thomas_fddc_module.p.terminate()
thomas_fddc_module.p.join()
thomas_fddc_module.p = mp.Pool({workers})

t0 = time.time()
cost_history = []
for i in range({gens}):
    gs = time.time()
    with redirect_stdout(io.StringIO()):
        fc.run()
    cost = float(fc.best_score) if not isinstance(fc.best_score, list) else float(fc.best_score[0])
    cost_history.append(cost)
    gt = time.time() - gs
    print(f"  Gen {{i+1:4d}}/{gens} | Cost: {{cost:.4f}} | {{gt:.1f}}s/gen", flush=True)

elapsed = time.time() - t0
thomas_fddc_module.p.terminate()
thomas_fddc_module.p.join()

result = json.dumps({{
    "impl": "thomas", "dataset": "{dataset}", "workers": {workers},
    "gens": {gens}, "pop": 100, "seed": None,
    "best_cost": min(cost_history), "cost_history": cost_history,
    "elapsed_sec": elapsed, "elapsed_min": elapsed / 60,
}})
print(f"RESULT|{{result}}")
'''

_OURS_TEMPLATE = '''
import sys
from pathlib import Path
PROJECT_ROOT = Path("{project_root}")
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "experiments"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "program code"))

import json
from shared import run_fddc
r = run_fddc("{dataset}", gen_count={gens}, pop_size=100,
             impl="new", seed={seed}, workers={workers})
result = json.dumps({{
    "impl": "ours", "dataset": "{dataset}", "workers": {workers},
    "gens": {gens}, "pop": 100, "seed": {seed},
    "best_cost": r["best_cost"], "cost_history": r["cost_history"],
    "elapsed_sec": r["elapsed_sec"], "elapsed_min": r["elapsed_min"],
}})
print(f"RESULT|{{result}}")
'''


def _run_subprocess(code, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}", flush=True)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [PYTHON, script_path],
            cwd=str(PROJECT_ROOT),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=7200,
        )
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT|"):
                result = json.loads(line[7:])
            else:
                print(line)

        if result is None:
            print(f"ERROR: no RESULT line. Return code: {proc.returncode}")
            if proc.returncode != 0:
                print(proc.stdout[-2000:])
        return result
    finally:
        Path(script_path).unlink(missing_ok=True)


def run_thomas(dataset, workers, gens=None):
    if gens is None:
        gens = TABLE_VIII[dataset]["gens"]
    ref = TABLE_VIII[dataset]
    code = _THOMAS_TEMPLATE.format(
        project_root=PROJECT_ROOT, datafile=ref["file"],
        dataset=dataset, workers=workers, gens=gens,
    )
    label = f"Thomas {workers}w | {dataset} | {gens} gens"
    return _run_subprocess(code, label)


def run_ours(dataset, workers, seed=42, gens=None):
    if gens is None:
        gens = TABLE_VIII[dataset]["gens"]
    code = _OURS_TEMPLATE.format(
        project_root=PROJECT_ROOT, dataset=dataset,
        workers=workers, seed=seed, gens=gens,
    )
    label = f"Ours {workers}w | {dataset} | {gens} gens | seed={seed}"
    return _run_subprocess(code, label)


def save_result(result, filename):
    if result is None:
        return
    result["timestamp"] = datetime.now().isoformat()
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {path.name}")
    print(f"  {result['elapsed_min']:.1f} min | cost={result['best_cost']:.4f}")
