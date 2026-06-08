"""Worker threads for simulation and optimization tasks.

These QThread subclasses run compute-intensive work without blocking the GUI.
"""

import threading

from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from polymer_growth.core import simulate, SimulationParams
from polymer_growth.objective import load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from polymer_growth.gui.worker_callables import (
    WorkerObjective, WorkerSimulate, WorkerCost,
)


class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking the UI."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, params: SimulationParams, seed: int,
                 track_kinetics: bool = False):
        super().__init__()
        self.params = params
        self.seed = seed
        self.track_kinetics = track_kinetics

    def run(self):
        try:
            rng = np.random.default_rng(self.seed)
            result = simulate(self.params, rng,
                              track_kinetics=self.track_kinetics)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationWorker(QThread):
    """Worker thread for running FDDC optimization without blocking the UI."""

    progress = Signal(int, float)
    console_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, experimental_data_path: str, config: FDDCConfig,
                 bounds: np.ndarray, seed: int,
                 seed_vector: Optional[np.ndarray] = None,
                 seed_noise_scale: float = 0.05):
        super().__init__()
        self.experimental_data_path = experimental_data_path
        self.config = config
        # Seed FDDC's initial pop1 from caller-supplied parameter values when
        # provided; default None preserves the uniform-random init.
        self.config.seed_vector = seed_vector
        self.config.seed_noise_scale = seed_noise_scale
        self.bounds = bounds
        self.seed = seed
        self._cancel_lock = threading.Lock()
        self._is_cancelled = False

    def cancel(self):
        with self._cancel_lock:
            self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        with self._cancel_lock:
            return self._is_cancelled

    def run(self):
        try:
            exp_lengths, exp_values = load_experimental_data(
                self.experimental_data_path)

            if self.config.sigma_length is None:
                self.config.sigma_length = int(np.count_nonzero(exp_values))

            # Picklable callables for the worker pool (spawn-safe on Windows).
            base_objective = WorkerObjective(exp_values, self.seed)
            base_simulate = WorkerSimulate(self.seed)
            base_cost = WorkerCost(exp_values)

            # In-process callables additionally honour cancellation; the
            # optimizer raises out of these when the user cancels mid-run.
            def objective_wrapper(params_array, sigma=None, eval_seed=None):
                if self.is_cancelled:
                    raise InterruptedError("Optimization cancelled")
                return base_objective(params_array, sigma=sigma,
                                      eval_seed=eval_seed)

            def simulate_fn(params_array, eval_seed):
                if self.is_cancelled:
                    raise InterruptedError("Optimization cancelled")
                return base_simulate(params_array, eval_seed)

            cost_fn = base_cost

            def progress_callback(gen, cost):
                if not self.is_cancelled:
                    self.progress.emit(gen, cost)

            def console_callback(message):
                if not self.is_cancelled:
                    self.console_message.emit(message)

            optimizer = FDDCOptimizer(
                bounds=self.bounds,
                objective_function=objective_wrapper,
                config=self.config,
                callback=progress_callback,
                console_callback=console_callback,
                simulate_fn=simulate_fn,
                cost_fn=cost_fn,
                worker_objective=base_objective,
                worker_simulate=base_simulate,
                worker_cost=base_cost,
            )

            result = optimizer.optimize(seed=self.seed)

            if not self.is_cancelled:
                self.finished.emit(result)

        except InterruptedError:
            self.error.emit("Optimization cancelled by user")
        except Exception as e:
            self.error.emit(str(e))
