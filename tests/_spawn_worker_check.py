"""Standalone spawn-mode worker check (run as a subprocess by the test suite).

This emulates the Windows multiprocessing start method ('spawn') on any OS and
runs a real parallel FDDC optimization. Before the worker-callable fix this
crashed with "'NoneType' object is not callable" because the pool's worker
globals were never populated in spawned children. Exit code 0 + the SUCCESS
marker means the parallel path works under spawn.

Kept out of normal collection (filename does not match test_*.py); invoked by
tests/test_spawn_workers.py via subprocess so pytest's own import machinery does
not interfere with spawn re-imports.
"""

import multiprocessing as mp
import sys

import numpy as np


def _run() -> None:
    # Import first (fddc forces 'fork' at import on POSIX), THEN override to
    # 'spawn' so the pool created inside optimize() uses the Windows method.
    from polymer_growth.core.parameters import ParameterBounds
    from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
    from polymer_growth.gui.worker_callables import (
        WorkerObjective, WorkerSimulate, WorkerCost,
    )

    mp.set_start_method("spawn", force=True)
    if mp.get_start_method() != "spawn":
        print("FAIL: could not set spawn start method")
        sys.exit(2)

    exp_values = np.linspace(1.0, 10.0, 40)
    bounds = ParameterBounds().as_array()
    seed = 42

    base_obj = WorkerObjective(exp_values, seed)
    base_sim = WorkerSimulate(seed)
    base_cost = WorkerCost(exp_values)

    cfg = FDDCConfig(population_size=6, max_generations=2, memory_size=6,
                     n_workers=2, sigma_length=int(np.count_nonzero(exp_values)))
    opt = FDDCOptimizer(
        bounds=bounds, objective_function=base_obj, config=cfg,
        simulate_fn=base_sim, cost_fn=base_cost,
        worker_objective=base_obj, worker_simulate=base_sim, worker_cost=base_cost,
    )

    result = opt.optimize(seed=seed)
    assert len(result.cost_history) == 2, result.cost_history
    print("SUCCESS: spawn parallel run completed cleanly")


if __name__ == "__main__":
    mp.freeze_support()
    if "--child-guard" not in sys.argv:
        try:
            _run()
        except Exception as exc:  # noqa: BLE001 - surface any worker crash
            print(f"FAIL: {type(exc).__name__}: {exc}")
            sys.exit(1)
