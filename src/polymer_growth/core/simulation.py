"""
Agent-based stochastic polymer growth simulation.

Implements the model from thesis (Basaran, 2020):
- Living chains can grow (add monomer) or die (become dead)
- Dead chains can react vampirically with living chains
- Monomer pool depletion affects growth/death rates

Key equations:
    monomer_ratio = current_pool / initial_pool
    P(growth) = p_growth × monomer_ratio
    P(death) = p_death × monomer_ratio
    P(vampiric_success) = pdr / (l_living^f(le,ln) × l_dead^f(de,ln))
"""

from dataclasses import dataclass, asdict
from typing import Tuple
import numpy as np
from numba import jit


@dataclass
class SimulationParams:
    """
    Parameters for polymer growth simulation.

    See thesis.txt Table I for parameter descriptions.

    Attributes:
        time_sim: Number of simulation timesteps
        number_of_molecules: Initial number of polymer chains
        monomer_pool: Initial monomer pool size (set to -1 for infinite)
        p_growth: Base probability of growth per timestep
        p_death: Base probability of death (chain termination) per timestep
        p_dead_react: Base probability for vampiric reaction
        l_exponent: Living chain length exponent for coupling
        d_exponent: Dead chain length exponent for coupling
        l_naked: Accessible surface ratio (controls exponent scaling)
        kill_spawns_new: Whether death events spawn new chains (bool)
    """

    time_sim: int
    number_of_molecules: int
    monomer_pool: float  # Can be -1 for infinite
    p_growth: float
    p_death: float
    p_dead_react: float
    l_exponent: float
    d_exponent: float
    l_naked: float
    kill_spawns_new: bool

    def validate(self) -> None:
        """Validate parameter ranges."""
        assert self.time_sim > 0, "time_sim must be positive"
        assert self.number_of_molecules > 0, "number_of_molecules must be positive"
        assert 0 <= self.p_growth < 1, "p_growth must be in [0, 1)"
        assert 0 <= self.p_death < 1, "p_death must be in [0, 1)"
        assert 0 <= self.p_dead_react <= 1, "p_dead_react must be in [0, 1]"
        assert 0 <= self.l_exponent <= 1, "l_exponent must be in [0, 1]"
        assert 0 <= self.d_exponent <= 1, "d_exponent must be in [0, 1]"
        assert 0 < self.l_naked <= 1, "l_naked must be in (0, 1]"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimulationParams":
        """Create from dictionary."""
        return cls(**d)


@dataclass
class Distribution:
    """
    Polymer chain length distribution.

    Attributes:
        living: Array of living chain lengths
        dead: Array of dead chain lengths
        coupled: Array of coupled (vampiric) chain lengths
    """

    living: np.ndarray
    dead: np.ndarray
    coupled: np.ndarray

    def all_chains(self) -> np.ndarray:
        """Return all chains (living + dead + coupled)."""
        return np.concatenate([self.living, self.dead, self.coupled])

    def histogram(self, bins: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram of all chain lengths.

        Args:
            bins: Number of bins (default: max_length + 1 for exact counts)

        Returns:
            (counts, bin_edges)
        """
        all_chains = self.all_chains()
        if bins is None:
            bins = int(all_chains.max()) + 1
        return np.histogram(all_chains, bins=bins)

    def stats(self) -> dict:
        """Compute distribution statistics."""
        all_chains = self.all_chains()
        return {
            'n_living': len(self.living),
            'n_dead': len(self.dead),
            'n_coupled': len(self.coupled),
            'n_total': len(all_chains),
            'mean_length': float(np.mean(all_chains)),
            'median_length': float(np.median(all_chains)),
            'max_length': int(np.max(all_chains)),
            'min_length': int(np.min(all_chains)),
        }


@jit(nopython=True)
def _compute_vampiric_success_prob(
    l_living: np.ndarray,
    l_dead: np.ndarray,
    p_dead_react: float,
    l_exponent: float,
    d_exponent: float,
    l_naked: float
) -> np.ndarray:
    """
    Compute vampiric coupling success probability (Numba JIT).

    Formula from thesis:
        p_success = pdr / (l_living^f × l_dead^f)
    where f = min(length × exponent/naked, exponent)

    Args:
        l_living: Living chain lengths being attacked
        l_dead: Dead (vampiric) chain lengths
        p_dead_react: Base reaction probability
        l_exponent: Living exponent
        d_exponent: Dead exponent
        l_naked: Naked ratio

    Returns:
        Array of success probabilities
    """
    # Living exponent: min(l × le/ln, le)
    living_exp = np.minimum(l_living * (l_exponent / l_naked), l_exponent)

    # Dead exponent: min(l × de/ln, de)
    dead_exp = np.minimum(l_dead * (d_exponent / l_naked), d_exponent)

    # p_success = pdr / (l_living^living_exp × l_dead^dead_exp)
    denominator = np.power(l_living, living_exp) * np.power(l_dead, dead_exp)

    return p_dead_react / denominator


def simulate(params: SimulationParams, rng: np.random.Generator) -> Distribution:
    """
    Run agent-based polymer growth simulation.

    Each timestep:
        1. Living chains may grow (add monomer)
        2. Living chains may die (become dead)
        3. Dead chains may attack living chains (vampiric coupling)

    Monomer pool depletion affects growth/death probabilities via monomer_ratio.

    Args:
        params: Simulation parameters (validated)
        rng: NumPy random generator for reproducibility

    Returns:
        Distribution of living, dead, and coupled chain lengths

    Example:
        >>> params = SimulationParams(
        ...     time_sim=1000,
        ...     number_of_molecules=10000,
        ...     monomer_pool=1000000,
        ...     p_growth=0.72,
        ...     p_death=0.000084,
        ...     p_dead_react=0.73,
        ...     l_exponent=0.41,
        ...     d_exponent=0.75,
        ...     l_naked=0.24,
        ...     kill_spawns_new=True
        ... )
        >>> rng = np.random.default_rng(42)
        >>> dist = simulate(params, rng)
        >>> print(dist.stats())
    """
    # Validate parameters
    params.validate()

    # Initialize pools
    living = np.ones(params.number_of_molecules, dtype=np.float64)
    dead = np.array([], dtype=np.float64)
    coupled = np.array([], dtype=np.float64)

    # Track monomer pool
    initial_monomer_pool = params.monomer_pool
    current_monomer_pool = params.monomer_pool

    # Main simulation loop
    for t in range(params.time_sim):
        if len(living) == 0:
            break  # No living chains left

        # Compute monomer ratio
        if current_monomer_pool < 0:
            monomer_ratio = 1.0  # Infinite monomer
        else:
            monomer_ratio = current_monomer_pool / initial_monomer_pool

        # Random numbers for fate decisions
        r = rng.random(len(living))

        # GROWTH: r < p_growth × monomer_ratio
        growth_threshold = params.p_growth * monomer_ratio
        will_grow = r < growth_threshold
        living[will_grow] += 1

        # Update monomer pool for growth
        if current_monomer_pool >= 0:
            monomers_used = np.sum(will_grow)
            current_monomer_pool = max(0, current_monomer_pool - monomers_used)

        # DEATH: p_growth × monomer_ratio <= r < (p_growth + p_death) × monomer_ratio
        death_threshold = (params.p_growth + params.p_death) * monomer_ratio
        will_die = (r >= growth_threshold) & (r < death_threshold)

        new_dead = living[will_die].copy()

        if params.kill_spawns_new:
            # Death spawns new chain → reset to length 1
            living[will_die] = 1
            if current_monomer_pool >= 0:
                spawns = np.sum(will_die)
                current_monomer_pool = max(0, current_monomer_pool - spawns)
        else:
            # Death removes chain → delete from living
            living = living[~will_die]

        # Add newly dead chains to dead pool
        if len(new_dead) > 0:
            dead = np.concatenate([dead, new_dead])

        # VAMPIRIC REACTIONS
        if len(dead) > 0 and len(living) > 0:
            # Each dead chain picks a random target (living or dead)
            n_targets = len(dead) + len(living)
            target_idx = rng.integers(0, n_targets, size=len(dead))

            # Filter: only dead chains that picked living targets
            attacks_living = target_idx >= len(dead)
            attacking_dead_idx = np.where(attacks_living)[0]

            if len(attacking_dead_idx) > 0:
                # Which living chains are attacked
                living_target_idx = target_idx[attacking_dead_idx] - len(dead)

                # Compute success probabilities
                l_living = living[living_target_idx]
                l_dead = dead[attacking_dead_idx]
                p_success = _compute_vampiric_success_prob(
                    l_living, l_dead, params.p_dead_react,
                    params.l_exponent, params.d_exponent, params.l_naked
                )

                # Roll for success
                r_success = rng.random(len(p_success))
                successful = r_success < p_success

                # Successful attacks: living + dead → coupled
                if np.any(successful):
                    successful_living_idx = living_target_idx[successful]
                    successful_dead_idx = attacking_dead_idx[successful]

                    # Update living chains (add dead length)
                    living[successful_living_idx] += dead[successful_dead_idx]

                    # Remove successful dead chains
                    still_dead_mask = np.ones(len(dead), dtype=bool)
                    still_dead_mask[successful_dead_idx] = False
                    dead = dead[still_dead_mask]

    return Distribution(living=living, dead=dead, coupled=coupled)