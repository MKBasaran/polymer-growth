"""Tests for FDDC seeded initial-population behaviour.

These exercise ``FDDCOptimizer._initialize_populations`` directly, both in the
default (uniform-random) mode and in the new seeded mode where pop1 is the
Simulation tab's parameter vector plus per-dimension Gaussian noise whose std
scales with each parameter's bound width.

Parameter order / bounds come straight from ParameterBounds.as_array():
    0 time_sim            (100, 3000)        integer
    1 number_of_molecules (10000, 120000)    integer
    2 monomer_pool        (1e6, 5e6)         integer
    3 p_growth            (0.1, 0.99)
    4 p_death             (0.0001, 0.002)
    5 p_dead_react        (0.1, 0.9)
    6 l_exponent          (0.1, 0.9)
    7 d_exponent          (0.1, 0.9)
    8 l_naked             (0.1, 1.0)
    9 kill_spawns_new     (0, 1)             boolean switch (never noised)
"""

import numpy as np
import pytest

from polymer_growth.core.parameters import ParameterBounds
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig

BOUNDS = ParameterBounds().as_array()
LOWER = BOUNDS[:, 0]
UPPER = BOUNDS[:, 1]
WIDTH = UPPER - LOWER
N_PARAMS = BOUNDS.shape[0]

INTEGER_DIMS = (0, 1, 2)
BOOL_DIM = 9
CONTINUOUS_DIMS = tuple(i for i in range(N_PARAMS)
                        if i not in INTEGER_DIMS and i != BOOL_DIM)


def _noop_objective(params, sigma=None, eval_seed=None):
    """Cheap objective so _initialize_populations' warm-up call is free."""
    return 0.0


def _init_pop1(config: FDDCConfig, seed: int) -> np.ndarray:
    """Run _initialize_populations in isolation and return pop1 as an array."""
    opt = FDDCOptimizer(
        bounds=BOUNDS,
        objective_function=_noop_objective,
        config=config,
    )
    opt.rng = np.random.default_rng(seed)
    opt._initialize_populations()
    return np.array(opt.pop1)


# --------------------------------------------------------------------------- #
# Regression: seed_vector=None must not change behaviour
# --------------------------------------------------------------------------- #

def test_regression_seed_vector_none_matches_uniform_sequence():
    """With seed_vector=None, pop1 must be exactly the uniform-random draws the
    pre-change code produced for the same RNG seed."""
    seed = 7
    pop_size = 50
    config = FDDCConfig(population_size=pop_size, seed_vector=None)
    pop1 = _init_pop1(config, seed)

    # Re-derive the exact draw sequence the original loop performed: one
    # rng.uniform(lower, upper, size=n_params) per individual, before any other
    # RNG consumption.
    rng = np.random.default_rng(seed)
    expected = np.array([
        rng.uniform(LOWER, UPPER, size=N_PARAMS) for _ in range(pop_size)
    ])

    np.testing.assert_array_equal(pop1, expected)


def test_regression_seed_vector_none_is_deterministic():
    config_a = FDDCConfig(population_size=30, seed_vector=None)
    config_b = FDDCConfig(population_size=30, seed_vector=None)
    np.testing.assert_array_equal(_init_pop1(config_a, 42), _init_pop1(config_b, 42))


# --------------------------------------------------------------------------- #
# Zero-noise determinism
# --------------------------------------------------------------------------- #

def test_zero_noise_individuals_equal_seed():
    v = LOWER + 0.3337 * WIDTH
    v[BOOL_DIM] = 1.0  # valid boolean seed value
    config = FDDCConfig(population_size=50, seed_vector=v.copy(),
                        seed_noise_scale=0.0)
    pop1 = _init_pop1(config, 42)

    for ind in pop1:
        for i in CONTINUOUS_DIMS:
            assert ind[i] == v[i]
        for i in INTEGER_DIMS:
            assert ind[i] == round(v[i])
        assert ind[BOOL_DIM] == v[BOOL_DIM]


# --------------------------------------------------------------------------- #
# Per-parameter noise scaling (the heart of the change)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [42, 123, 2026])
def test_per_parameter_noise_scaling(seed):
    """Seed at the centre of every dimension; noise std must equal
    0.05 * (upper - lower) per dimension, and the mean must sit on the seed."""
    v = (LOWER + UPPER) / 2.0
    scale = 0.05
    pop_size = 2000
    sigma = scale * WIDTH

    config = FDDCConfig(population_size=pop_size, seed_vector=v.copy(),
                        seed_noise_scale=scale)
    pop1 = _init_pop1(config, seed)

    # Boolean dim is constant by design.
    assert np.all(pop1[:, BOOL_DIM] == v[BOOL_DIM])

    for i in range(N_PARAMS):
        if i == BOOL_DIM:
            continue
        col = pop1[:, i]
        emp_mean = col.mean()
        emp_std = col.std()
        assert abs(emp_mean - v[i]) < 0.15 * sigma[i], (
            f"dim {i}: mean {emp_mean} off seed {v[i]} (sigma {sigma[i]})")
        assert 0.80 * sigma[i] < emp_std < 1.20 * sigma[i], (
            f"dim {i}: std {emp_std} outside +/-20% of sigma {sigma[i]}")


# --------------------------------------------------------------------------- #
# Bounds respect
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("at_bound", ["lower", "upper"])
def test_bounds_respected_at_extremes(at_bound):
    v = LOWER.copy() if at_bound == "lower" else UPPER.copy()
    config = FDDCConfig(population_size=2000, seed_vector=v.copy(),
                        seed_noise_scale=0.10)
    pop1 = _init_pop1(config, 42)

    assert np.all(pop1 >= LOWER - 0.0), "some individual fell below lower bound"
    assert np.all(pop1 <= UPPER + 0.0), "some individual rose above upper bound"
    # Tighter explicit per-dimension check.
    for i in range(N_PARAMS):
        assert pop1[:, i].min() >= LOWER[i]
        assert pop1[:, i].max() <= UPPER[i]


# --------------------------------------------------------------------------- #
# Integer typing
# --------------------------------------------------------------------------- #

def test_integer_dimensions_are_integer_valued():
    v = (LOWER + UPPER) / 2.0
    config = FDDCConfig(population_size=200, seed_vector=v.copy(),
                        seed_noise_scale=0.05)
    pop1 = _init_pop1(config, 42)
    for i in INTEGER_DIMS:
        col = pop1[:, i]
        assert np.all(col == np.round(col)), f"dim {i} has non-integer values"


# --------------------------------------------------------------------------- #
# Boolean immutability
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bool_value", [0.0, 1.0])
def test_boolean_dimension_is_immutable(bool_value):
    v = (LOWER + UPPER) / 2.0
    v[BOOL_DIM] = bool_value
    config = FDDCConfig(population_size=500, seed_vector=v.copy(),
                        seed_noise_scale=0.20)
    pop1 = _init_pop1(config, 42)
    col = pop1[:, BOOL_DIM]
    assert np.all((col == 0.0) | (col == 1.0))
    assert np.all(col == bool_value)


# --------------------------------------------------------------------------- #
# Determinism under same seed
# --------------------------------------------------------------------------- #

def test_seeded_init_deterministic_same_seed():
    v = (LOWER + UPPER) / 2.0
    v[BOOL_DIM] = 1.0
    cfg_a = FDDCConfig(population_size=60, seed_vector=v.copy(),
                       seed_noise_scale=0.07)
    cfg_b = FDDCConfig(population_size=60, seed_vector=v.copy(),
                       seed_noise_scale=0.07)
    np.testing.assert_array_equal(_init_pop1(cfg_a, 99), _init_pop1(cfg_b, 99))


def test_seeded_init_differs_across_seeds():
    v = (LOWER + UPPER) / 2.0
    v[BOOL_DIM] = 1.0
    cfg = FDDCConfig(population_size=60, seed_vector=v.copy(),
                     seed_noise_scale=0.07)
    a = _init_pop1(cfg, 1)
    b = _init_pop1(cfg, 2)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Out-of-bounds seed
# --------------------------------------------------------------------------- #

def test_out_of_bounds_seed_is_clipped():
    v = (LOWER + UPPER) / 2.0
    v[3] = UPPER[3] + 5.0  # p_growth pushed above its upper bound
    config = FDDCConfig(population_size=1000, seed_vector=v.copy(),
                        seed_noise_scale=0.05)
    pop1 = _init_pop1(config, 42)
    for i in range(N_PARAMS):
        assert pop1[:, i].min() >= LOWER[i]
        assert pop1[:, i].max() <= UPPER[i]


def test_wrong_length_seed_vector_raises():
    config = FDDCConfig(population_size=20,
                        seed_vector=np.zeros(N_PARAMS - 1))
    opt = FDDCOptimizer(bounds=BOUNDS, objective_function=_noop_objective,
                        config=config)
    opt.rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        opt._initialize_populations()
