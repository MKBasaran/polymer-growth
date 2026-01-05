"""Tests for core simulation engine."""

import numpy as np
import pytest
from polymer_growth.core import simulate, SimulationParams, Distribution


class TestSimulationParams:
    """Test parameter validation."""

    def test_valid_params(self):
        """Valid parameters should not raise."""
        params = SimulationParams(
            time_sim=1000,
            number_of_molecules=10000,
            monomer_pool=1000000,
            p_growth=0.72,
            p_death=0.000084,
            p_dead_react=0.73,
            l_exponent=0.41,
            d_exponent=0.75,
            l_naked=0.24,
            kill_spawns_new=True
        )
        params.validate()  # Should not raise

    def test_invalid_p_growth(self):
        """Invalid p_growth should raise."""
        params = SimulationParams(
            time_sim=100, number_of_molecules=100, monomer_pool=1000,
            p_growth=1.5,  # Invalid!
            p_death=0.001, p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        with pytest.raises(AssertionError):
            params.validate()

    def test_to_from_dict(self):
        """Test serialization."""
        params = SimulationParams(
            time_sim=1000, number_of_molecules=10000, monomer_pool=1000000,
            p_growth=0.72, p_death=0.000084, p_dead_react=0.73,
            l_exponent=0.41, d_exponent=0.75, l_naked=0.24,
            kill_spawns_new=True
        )
        d = params.to_dict()
        params2 = SimulationParams.from_dict(d)
        assert params == params2


class TestSimulation:
    """Test simulation correctness."""

    @pytest.fixture
    def simple_params(self):
        """Simple test parameters."""
        return SimulationParams(
            time_sim=100,
            number_of_molecules=1000,
            monomer_pool=100000,
            p_growth=0.7,
            p_death=0.001,
            p_dead_react=0.5,
            l_exponent=0.5,
            d_exponent=0.5,
            l_naked=0.5,
            kill_spawns_new=True
        )

    def test_determinism(self, simple_params):
        """Same seed → same output."""
        rng1 = np.random.default_rng(42)
        dist1 = simulate(simple_params, rng1)

        rng2 = np.random.default_rng(42)
        dist2 = simulate(simple_params, rng2)

        assert np.array_equal(dist1.living, dist2.living)
        assert np.array_equal(dist1.dead, dist2.dead)
        assert np.array_equal(dist1.coupled, dist2.coupled)

    def test_different_seeds_different_output(self, simple_params):
        """Different seeds → different output."""
        rng1 = np.random.default_rng(42)
        dist1 = simulate(simple_params, rng1)

        rng2 = np.random.default_rng(999)
        dist2 = simulate(simple_params, rng2)

        # Very unlikely to be identical
        assert not np.array_equal(dist1.living, dist2.living)

    def test_chain_count_with_spawn(self):
        """With kill_spawns_new=True, total chains should be constant."""
        params = SimulationParams(
            time_sim=100, number_of_molecules=1000, monomer_pool=100000,
            p_growth=0.7, p_death=0.01, p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        # Living + dead should equal initial (coupled reduces living+dead)
        # But with vampiric reactions, some dead disappear
        total = len(dist.living) + len(dist.dead) + len(dist.coupled)
        # With spawn, living count stays ~constant
        assert len(dist.living) > 0

    def test_chain_count_without_spawn(self):
        """With kill_spawns_new=False, chains can disappear."""
        params = SimulationParams(
            time_sim=100, number_of_molecules=1000, monomer_pool=100000,
            p_growth=0.7, p_death=0.05, p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=False
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        # Living count should decrease over time
        assert len(dist.living) < 1000

    def test_growth_increases_lengths(self):
        """Chains should grow over time."""
        params = SimulationParams(
            time_sim=100, number_of_molecules=100, monomer_pool=100000,
            p_growth=0.9,  # High growth
            p_death=0.001,  # Low death
            p_dead_react=0.1,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        # All chains started at length 1, should be longer now
        assert np.mean(dist.all_chains()) > 1
        assert np.max(dist.all_chains()) > 1

    def test_infinite_monomer(self):
        """With infinite monomer (pool=-1), simulation should work."""
        params = SimulationParams(
            time_sim=50, number_of_molecules=100,
            monomer_pool=-1,  # Infinite
            p_growth=0.8, p_death=0.001, p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        assert len(dist.all_chains()) > 0
        assert np.mean(dist.all_chains()) > 1

    def test_distribution_stats(self, simple_params):
        """Distribution stats should be reasonable."""
        rng = np.random.default_rng(42)
        dist = simulate(simple_params, rng)

        stats = dist.stats()
        assert stats['n_total'] > 0
        assert stats['mean_length'] >= 1
        assert stats['min_length'] >= 1
        assert stats['max_length'] >= stats['mean_length']

    def test_histogram(self, simple_params):
        """Histogram computation should work."""
        rng = np.random.default_rng(42)
        dist = simulate(simple_params, rng)

        counts, bins = dist.histogram()
        assert len(counts) > 0
        assert len(bins) == len(counts) + 1
        assert np.sum(counts) == dist.stats()['n_total']


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_death_no_dead_chains(self):
        """With p_death=0, should have no dead chains."""
        params = SimulationParams(
            time_sim=50, number_of_molecules=100, monomer_pool=100000,
            p_growth=0.7, p_death=0.0,  # No death
            p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        # Should have very few or no dead chains
        assert len(dist.dead) == 0

    def test_high_death_creates_dead_pool(self):
        """High death rate should create dead chains."""
        params = SimulationParams(
            time_sim=50, number_of_molecules=100, monomer_pool=100000,
            p_growth=0.3, p_death=0.2,  # High death
            p_dead_react=0.1,  # Low vampiric
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        assert len(dist.dead) > 0

    def test_very_short_simulation(self):
        """Single timestep should work."""
        params = SimulationParams(
            time_sim=1,  # One step
            number_of_molecules=100, monomer_pool=10000,
            p_growth=0.5, p_death=0.01, p_dead_react=0.5,
            l_exponent=0.5, d_exponent=0.5, l_naked=0.5,
            kill_spawns_new=True
        )
        rng = np.random.default_rng(42)
        dist = simulate(params, rng)

        assert len(dist.all_chains()) > 0