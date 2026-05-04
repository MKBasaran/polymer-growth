"""Worker threads for simulation and optimization tasks.

These QThread subclasses run compute-intensive work without blocking the GUI.
"""

import threading
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from polymer_growth.core import simulate, SimulationParams
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig


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
                 bounds: np.ndarray, seed: int):
        super().__init__()
        self.experimental_data_path = experimental_data_path
        self.config = config
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
            objective = MinMaxV2ObjectiveFunction(exp_values)

            if self.config.sigma_length is None:
                self.config.sigma_length = int(np.count_nonzero(exp_values))

            def objective_wrapper(params_array, sigma=None, eval_seed=None):
                if self.is_cancelled:
                    raise InterruptedError("Optimization cancelled")
                params_array = np.asarray(params_array).flatten()
                params_list = params_array.tolist()
                params = SimulationParams(
                    time_sim=int(params_list[0]),
                    number_of_molecules=int(params_list[1]),
                    monomer_pool=int(params_list[2]),
                    p_growth=params_list[3],
                    p_death=params_list[4],
                    p_dead_react=params_list[5],
                    l_exponent=params_list[6],
                    d_exponent=params_list[7],
                    l_naked=params_list[8],
                    kill_spawns_new=bool(round(params_list[9]))
                )
                rng = np.random.default_rng(
                    eval_seed if eval_seed is not None else self.seed)
                dist = simulate(params, rng)
                return objective.compute_cost(dist, sigma=sigma)

            def _make_params(params_array):
                params_list = np.asarray(params_array).flatten().tolist()
                return SimulationParams(
                    time_sim=int(params_list[0]),
                    number_of_molecules=int(params_list[1]),
                    monomer_pool=int(params_list[2]),
                    p_growth=params_list[3],
                    p_death=params_list[4],
                    p_dead_react=params_list[5],
                    l_exponent=params_list[6],
                    d_exponent=params_list[7],
                    l_naked=params_list[8],
                    kill_spawns_new=bool(round(params_list[9]))
                )

            def simulate_fn(params_array, eval_seed):
                if self.is_cancelled:
                    raise InterruptedError("Optimization cancelled")
                rng = np.random.default_rng(
                    eval_seed if eval_seed is not None else self.seed)
                return simulate(_make_params(params_array), rng)

            def cost_fn(dist, sigma=None):
                return objective.compute_cost(dist, sigma=sigma)

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
            )

            result = optimizer.optimize(seed=self.seed)

            if not self.is_cancelled:
                self.finished.emit(result)

        except InterruptedError:
            self.error.emit("Optimization cancelled by user")
        except Exception as e:
            self.error.emit(str(e))
