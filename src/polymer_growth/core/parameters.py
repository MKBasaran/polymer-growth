"""Parameter bounds and validation utilities."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class ParameterBounds:
    """
    Parameter bounds for optimization.

    Each parameter is a tuple (lower, upper).
    """

    time_sim: Tuple[float, float] = (100, 3000)
    number_of_molecules: Tuple[float, float] = (10000, 120000)
    monomer_pool: Tuple[float, float] = (1_000_000, 5_000_000)
    p_growth: Tuple[float, float] = (0.1, 0.99)
    p_death: Tuple[float, float] = (0.0001, 0.002)
    p_dead_react: Tuple[float, float] = (0.1, 0.9)
    l_exponent: Tuple[float, float] = (0.1, 0.9)
    d_exponent: Tuple[float, float] = (0.1, 0.9)
    l_naked: Tuple[float, float] = (0.1, 1.0)
    kill_spawns_new: Tuple[float, float] = (0, 1)  # Binary

    def as_array(self) -> np.ndarray:
        """Convert to numpy array (10×2)."""
        return np.array([
            self.time_sim,
            self.number_of_molecules,
            self.monomer_pool,
            self.p_growth,
            self.p_death,
            self.p_dead_react,
            self.l_exponent,
            self.d_exponent,
            self.l_naked,
            self.kill_spawns_new,
        ])

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        """Convert to dictionary."""
        return {
            'time_sim': self.time_sim,
            'number_of_molecules': self.number_of_molecules,
            'monomer_pool': self.monomer_pool,
            'p_growth': self.p_growth,
            'p_death': self.p_death,
            'p_dead_react': self.p_dead_react,
            'l_exponent': self.l_exponent,
            'd_exponent': self.d_exponent,
            'l_naked': self.l_naked,
            'kill_spawns_new': self.kill_spawns_new,
        }

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Clip values to bounds."""
        bounds_array = self.as_array()
        return np.clip(values, bounds_array[:, 0], bounds_array[:, 1])

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        """Sample uniformly from bounds."""
        bounds_array = self.as_array()
        return rng.uniform(bounds_array[:, 0], bounds_array[:, 1])


def validate_parameters(params_array: np.ndarray, bounds: ParameterBounds = None) -> bool:
    """
    Check if parameters are within bounds.

    Args:
        params_array: Array of 10 parameter values
        bounds: Parameter bounds (default: ParameterBounds())

    Returns:
        True if all parameters are in bounds
    """
    if bounds is None:
        bounds = ParameterBounds()

    bounds_array = bounds.as_array()
    return np.all((params_array >= bounds_array[:, 0]) & (params_array <= bounds_array[:, 1]))