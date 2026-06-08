"""Top-level, picklable worker callables for FDDC parallel evaluation.

The GUI worker threads used to hand the optimizer *closures* (functions defined
inside a method, capturing ``self``). Closures cannot be pickled, so on any
platform that uses the ``spawn`` multiprocessing start method -- i.e. Windows --
the optimizer's process pool could not reconstruct them in its workers. The
module-level worker globals stayed ``None`` and the first parallel task raised
``TypeError: 'NoneType' object is not callable``. (On macOS/Linux the optimizer
forces ``fork``, which inherits the closures, so the bug never showed there.)

These small classes hold only picklable state -- the experimental target array
and a default seed -- and reconstruct the simulation/cost on call. They are sent
to each worker via ``Pool(initializer=...)`` and therefore work identically
under both ``fork`` and ``spawn``.
"""

import numpy as np

from polymer_growth.core import simulate, SimulationParams
from polymer_growth.objective import MinMaxV2ObjectiveFunction


def params_from_array(params_array) -> SimulationParams:
    """Build SimulationParams from a 10-element parameter vector.

    Order matches ParameterBounds.as_array(); integer dimensions are cast and
    ``kill_spawns_new`` uses the round-to-bool convention used everywhere else.
    """
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
        kill_spawns_new=bool(round(params_list[9])),
    )


class WorkerObjective:
    """Picklable ``(params, sigma, eval_seed) -> cost`` callable."""

    def __init__(self, exp_values: np.ndarray, default_seed: int):
        self._objective = MinMaxV2ObjectiveFunction(exp_values)
        self._default_seed = default_seed

    def __call__(self, params_array, sigma=None, eval_seed=None) -> float:
        rng = np.random.default_rng(
            eval_seed if eval_seed is not None else self._default_seed)
        dist = simulate(params_from_array(params_array), rng)
        return self._objective.compute_cost(dist, sigma=sigma)


class WorkerSimulate:
    """Picklable ``(params, eval_seed) -> Distribution`` callable."""

    def __init__(self, default_seed: int):
        self._default_seed = default_seed

    def __call__(self, params_array, eval_seed):
        rng = np.random.default_rng(
            eval_seed if eval_seed is not None else self._default_seed)
        return simulate(params_from_array(params_array), rng)


class WorkerCost:
    """Picklable ``(distribution, sigma) -> cost`` callable."""

    def __init__(self, exp_values: np.ndarray):
        self._objective = MinMaxV2ObjectiveFunction(exp_values)

    def __call__(self, dist, sigma=None) -> float:
        return self._objective.compute_cost(dist, sigma=sigma)
