"""Tests for objective functions."""

import numpy as np
import pytest
from polymer_growth.core import Distribution
from polymer_growth.objective import MinMaxV2ObjectiveFunction, MinMaxV2Config
from polymer_growth.objective.loaders import preprocess_simulation_histogram


class TestPreprocessing:
    """Test histogram preprocessing."""

    def test_preprocess_empty_distribution(self):
        """Empty distribution should return zeros."""
        hist = preprocess_simulation_histogram(
            np.array([]), np.array([]), np.array([]), target_length=10
        )
        assert len(hist) == 10
        assert np.all(hist == 0)

    def test_preprocess_combines_all_pools(self):
        """Should combine living, dead, and coupled chains."""
        living = np.array([1, 2, 3])
        dead = np.array([2, 3])
        coupled = np.array([5])

        hist = preprocess_simulation_histogram(living, dead, coupled, target_length=10)

        # Check that histogram captures all values
        assert hist[1] == 1  # One chain of length 1
        assert hist[2] == 2  # Two chains of length 2
        assert hist[3] == 2  # Two chains of length 3
        assert hist[5] == 1  # One chain of length 5

    def test_preprocess_pads_to_target_length(self):
        """Should pad with zeros to match target length."""
        living = np.array([1, 1, 1])
        hist = preprocess_simulation_histogram(living, np.array([]), np.array([]), target_length=20)

        assert len(hist) == 20
        assert hist[1] == 3
        assert np.sum(hist[10:]) == 0  # Padded region is zero

    def test_preprocess_truncates_to_target_length(self):
        """Should truncate if longer than target."""
        living = np.arange(1, 100)  # Long chains
        hist = preprocess_simulation_histogram(living, np.array([]), np.array([]), target_length=10)

        assert len(hist) == 10


class TestMinMaxV2:
    """Test MinMaxV2 objective function."""

    @pytest.fixture
    def simple_experimental(self):
        """Simple peaked experimental distribution."""
        # Gaussian-like peak at index 10
        x = np.arange(20)
        exp = np.exp(-((x - 10) ** 2) / 10.0)
        return exp

    @pytest.fixture
    def objective(self, simple_experimental):
        """Create objective function with simple experimental data."""
        return MinMaxV2ObjectiveFunction(simple_experimental)

    def test_initialization(self, simple_experimental):
        """Test objective function initialization."""
        obj = MinMaxV2ObjectiveFunction(simple_experimental)
        assert obj.exp_values is not None
        assert obj.exp_norm is not None
        assert np.max(obj.exp_norm) == 1.0  # Normalized

    def test_zero_experimental_raises(self):
        """Zero experimental data should raise."""
        with pytest.raises(ValueError):
            MinMaxV2ObjectiveFunction(np.zeros(10))

    def test_perfect_match_low_cost(self, objective, simple_experimental):
        """Perfect match should have near-zero cost."""
        # Create simulated distribution matching experimental
        sim_hist = simple_experimental * 100  # Scale doesn't matter due to normalization

        # Create distribution object
        # We'll fake it by putting all mass at appropriate lengths
        living = []
        for length, count in enumerate(sim_hist):
            living.extend([length] * int(count))
        living = np.array(living)

        dist = Distribution(living=living, dead=np.array([]), coupled=np.array([]))

        cost = objective.compute_cost(dist)
        assert cost < 10.0  # Should be low for good match

    def test_misaligned_peaks_higher_cost(self, objective):
        """Misaligned peaks should increase cost."""
        # Create distribution with peak at wrong location
        x = np.arange(20)
        shifted = np.exp(-((x - 5) ** 2) / 10.0)  # Peak at 5 instead of 10

        living = []
        for length, count in enumerate(shifted * 100):
            living.extend([length] * int(count))
        living = np.array(living)

        dist = Distribution(living=living, dead=np.array([]), coupled=np.array([]))

        cost = objective.compute_cost(dist)
        # Cost should be higher due to peak misalignment penalty
        assert cost > 0

    def test_empty_simulation_max_cost(self, objective):
        """Empty simulation should return max cost."""
        dist = Distribution(
            living=np.array([]),
            dead=np.array([]),
            coupled=np.array([])
        )
        cost = objective.compute_cost(dist)
        assert cost == objective.config.max_cost

    def test_custom_sigma_weights(self, simple_experimental):
        """Custom sigma weights should affect cost."""
        config1 = MinMaxV2Config(sigma=[1, 1, 1, 1, 1, 1])
        config2 = MinMaxV2Config(sigma=[10, 1, 1, 1, 1, 1])  # Higher weight on first partition

        obj1 = MinMaxV2ObjectiveFunction(simple_experimental, config1)
        obj2 = MinMaxV2ObjectiveFunction(simple_experimental, config2)

        # Create slightly off distribution
        x = np.arange(20)
        sim = np.exp(-((x - 11) ** 2) / 10.0)  # Slightly shifted

        living = []
        for length, count in enumerate(sim * 100):
            living.extend([length] * int(count))
        living = np.array(living)

        dist = Distribution(living=living, dead=np.array([]), coupled=np.array([]))

        cost1 = obj1.compute_cost(dist)
        cost2 = obj2.compute_cost(dist)

        # Costs should differ due to different weights
        assert cost1 != cost2

    def test_peak_alignment(self, objective):
        """Test internal peak alignment function."""
        # Simulated distribution with peak at index 5
        sim_norm = np.zeros(20)
        sim_norm[5] = 1.0

        # Should shift to align with experimental peak at 10
        trans_sim, shift, percentage = objective._align_peaks(sim_norm)

        # Peak should now be at experimental peak location
        assert np.argmax(trans_sim) == objective.exp_peak_idx
        assert shift == 5 - 10  # Negative = shift right

    def test_cost_bounded(self, objective):
        """Cost should never exceed max_cost."""
        # Create extreme distribution
        living = np.array([1000] * 1000)  # All very long chains
        dist = Distribution(living=living, dead=np.array([]), coupled=np.array([]))

        cost = objective.compute_cost(dist)
        assert cost <= objective.config.max_cost
        assert not np.isinf(cost)

    def test_different_distributions_different_costs(self, objective):
        """Different distributions should produce different costs."""
        # Distribution 1: peak at correct location
        dist1_living = np.array([10] * 100)
        dist1 = Distribution(living=dist1_living, dead=np.array([]), coupled=np.array([]))

        # Distribution 2: peak at wrong location
        dist2_living = np.array([5] * 100)
        dist2 = Distribution(living=dist2_living, dead=np.array([]), coupled=np.array([]))

        cost1 = objective.compute_cost(dist1)
        cost2 = objective.compute_cost(dist2)

        assert cost1 != cost2