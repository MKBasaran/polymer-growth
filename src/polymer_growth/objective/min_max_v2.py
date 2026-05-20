"""
MinMaxV2 objective function for polymer distribution fitting.

This is the cost function used in the thesis (Basaran, 2020).

Algorithm:
    1. Min-max normalize both experimental and simulated distributions
    2. Find peak locations in both
    3. Shift simulated distribution to align peaks
    4. Compute weighted cost across partitions
    5. Add penalty for extra simulated values outside experimental range
    6. Apply exponential penalty for peak misalignment
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from polymer_growth.core import Distribution
from polymer_growth.objective.loaders import preprocess_simulation_histogram


@dataclass
class MinMaxV2Config:
    """
    Configuration for MinMaxV2 objective function.

    Attributes:
        sigma: Partition weights (default: [1,1,1,1,1,1])
               Higher values increase weight for that partition
        transfac: Translation penalty factor (default: 1.0)
                  Controls exponential penalty for peak misalignment
        max_cost: Maximum allowed cost (default: 100000)
                  Prevents inf values, caps extreme costs
    """
    sigma: list[float] = None
    transfac: float = 1.0
    max_cost: float = 100000.0

    def __post_init__(self):
        if self.sigma is None:
            self.sigma = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class MinMaxV2ObjectiveFunction:
    """
    MinMaxV2 objective function for comparing simulated and experimental distributions.

    This function:
    - Normalizes distributions by their max values
    - Aligns peaks via translation
    - Computes weighted partition-based cost
    - Applies penalties for misalignment

    Example:
        >>> from polymer_growth.objective import load_experimental_data
        >>> exp_lengths, exp_values = load_experimental_data("data.xlsx")
        >>> obj = MinMaxV2ObjectiveFunction(exp_values)
        >>> cost = obj.compute_cost(distribution)
    """

    def __init__(
        self,
        experimental_values: np.ndarray,
        config: Optional[MinMaxV2Config] = None
    ):
        """
        Initialize objective function.

        Args:
            experimental_values: Experimental distribution values (1D array)
            config: Configuration (uses defaults if None)
        """
        if not isinstance(experimental_values, np.ndarray):
            raise TypeError("experimental_values must be a numpy array")
        if experimental_values.ndim != 1:
            raise ValueError("experimental_values must be 1-dimensional")
        if len(experimental_values) == 0:
            raise ValueError("experimental_values must not be empty")

        self.exp_values = experimental_values
        self.config = config if config is not None else MinMaxV2Config()

        # Precompute normalized experimental distribution
        exp_max = self.exp_values.max()
        if exp_max == 0:
            raise ValueError("Experimental distribution is all zeros")
        self.exp_norm = self.exp_values / exp_max

        # Cache experimental peak location
        self.exp_peak_idx = np.argmax(self.exp_norm)

    def compute_cost(
        self,
        distribution: Distribution,
        sigma: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute cost for a simulated distribution.

        Args:
            distribution: Simulated polymer distribution
            sigma: Optional sigma weights for partitions (array-like)
                   If None, uses self.config.sigma

        Returns:
            Cost value (lower is better)
        """
        # Convert distribution to histogram
        sim_hist = preprocess_simulation_histogram(
            distribution.living,
            distribution.dead,
            distribution.coupled,
            len(self.exp_values)
        )

        # Normalize simulated distribution
        sim_max = sim_hist.max()
        if sim_max == 0:
            return self.config.max_cost  # No chains -> max penalty

        sim_norm = sim_hist / sim_max

        # Align peaks
        trans_sim_norm, shift_distance, peak_percentage = self._align_peaks(sim_norm)

        # Use provided sigma or fall back to config
        active_sigma = sigma if sigma is not None else np.array(self.config.sigma)

        # Compute cost with active sigma
        cost = self._compute_partition_cost(trans_sim_norm, active_sigma)

        # Apply peak alignment penalty
        cost *= np.exp(peak_percentage / self.config.transfac)

        # Cap at max cost
        if np.isinf(cost) or cost > self.config.max_cost:
            cost = self.config.max_cost

        return float(cost)

    def _align_peaks(
        self,
        sim_norm: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        Align simulated distribution peak to experimental peak.

        Args:
            sim_norm: Normalized simulated distribution

        Returns:
            Tuple of (shifted_sim, shift_distance, peak_percentage_error)
        """
        # Find simulated peak
        sim_peak_idx = np.argmax(sim_norm)

        # Compute shift distance (positive = shift left, negative = shift right)
        shift = sim_peak_idx - self.exp_peak_idx

        # Shift the distribution
        if shift >= 0:
            # Shift left: remove first `shift` elements, pad right
            trans_sim = np.concatenate([sim_norm[shift:], np.zeros(shift)])
        else:
            # Shift right: pad left, remove last `abs(shift)` elements
            trans_sim = np.concatenate([np.zeros(abs(shift)), sim_norm[:len(sim_norm) + shift]])

        # Compute peak location error (relative)
        if self.exp_peak_idx == 0:
            peak_percentage = abs(shift)  # Avoid division by zero
        else:
            peak_percentage = abs(shift / self.exp_peak_idx)

        return trans_sim, shift, peak_percentage

    def _compute_partition_cost(
        self,
        trans_sim_norm: np.ndarray,
        sigma: np.ndarray
    ) -> float:
        """
        Compute weighted partition-based cost.

        Cost components:
        1. Partition costs: Divide experimental non-zero region into N partitions,
           compute normalized error for each, weight by sigma
        2. Remainder cost: Handle any indices not covered by partitions
        3. Extra cost: Penalize simulated values outside experimental non-zero region

        Args:
            trans_sim_norm: Shifted, normalized simulated distribution
            sigma: Partition weights (array of weights)

        Returns:
            Total cost
        """
        # Find non-zero indices in experimental data
        exp_nonzero_idx = np.where(self.exp_norm > 0)[0]

        if len(exp_nonzero_idx) == 0:
            return 0.0  # Edge case: no experimental data

        # Find non-zero indices in simulated data
        sim_nonzero_idx = np.where(trans_sim_norm > 0)[0]

        # Extra indices: simulated has values where experimental doesn't
        extra_idx = [i for i in sim_nonzero_idx if i not in exp_nonzero_idx]

        # Partition size based on sigma length
        n_partitions = len(sigma)
        partition_size = len(exp_nonzero_idx) // n_partitions

        total_cost = 0.0

        # Compute cost for each partition
        for i in range(n_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size
            partition_idx = exp_nonzero_idx[start:end]

            # Relative error sum for this partition
            error_sum = 0.0
            for idx in partition_idx:
                exp_val = self.exp_norm[idx]
                sim_val = trans_sim_norm[idx]
                max_val = max(exp_val, sim_val)

                if max_val > 0:
                    error = abs(exp_val - sim_val) / max_val
                    error_sum += error

            # Weight by sigma
            partition_cost = error_sum * sigma[i]
            total_cost += partition_cost

        # Handle missed values (remainder after partitioning)
        missed_count = len(exp_nonzero_idx) - partition_size * n_partitions
        if missed_count > 0:
            remainder_idx = exp_nonzero_idx[-missed_count:]
            for idx in remainder_idx:
                exp_val = self.exp_norm[idx]
                sim_val = trans_sim_norm[idx]
                max_val = max(exp_val, sim_val)

                if max_val > 0:
                    error = abs(exp_val - sim_val) / max_val
                    total_cost += error

        # Extra cost for simulated values outside experimental range
        extra_cost = 0.0
        for idx in extra_idx:
            extra_cost += abs(self.exp_norm[idx] - trans_sim_norm[idx])
        total_cost += extra_cost

        return total_cost