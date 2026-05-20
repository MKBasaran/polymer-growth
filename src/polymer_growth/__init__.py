"""
Polymer Growth Simulation and Parameter Optimization
=====================================================

A modern Python package for stochastic agent-based polymer growth simulation
and parameter inference using genetic algorithms.

Core modules:
- core: Simulation engine
- objective: Cost/fitness functions
- optimizers: FDDC and other GA variants
- cli: Command-line interface
- gui: Graphical user interface

Example:
    >>> from polymer_growth.core import simulate, SimulationParams
    >>> import numpy as np
    >>> params = SimulationParams(
    ...     time_sim=1000,
    ...     number_of_molecules=100000,
    ...     monomer_pool=31600000,
    ...     p_growth=0.72,
    ...     p_death=0.000084,
    ...     p_dead_react=0.73,
    ...     l_exponent=0.41,
    ...     d_exponent=0.75,
    ...     l_naked=0.24,
    ...     kill_spawns_new=True
    ... )
    >>> rng = np.random.default_rng(42)
    >>> distribution = simulate(params, rng)
"""

__version__ = "0.1.0"

from polymer_growth.core.simulation import simulate, SimulationParams, Distribution
from polymer_growth.core.parameters import ParameterBounds

__all__ = ["simulate", "SimulationParams", "Distribution", "ParameterBounds", "__version__"]