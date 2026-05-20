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
from typing import Tuple, Optional, Union
import numpy as np


# PEtOx chemistry constants
MONOMER_MASS = 99.13  # g/mol (2-ethyl-2-oxazoline)
INITIATOR_MASS = 180.0  # g/mol (methyl tosylate initiator)


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

    def length_to_mw(self, lengths: np.ndarray) -> np.ndarray:
        """
        Convert chain lengths (degree of polymerization) to molecular weight.

        MW = (DP × monomer_mass) + initiator_mass

        Args:
            lengths: Array of chain lengths (DP)

        Returns:
            Array of molecular weights (g/mol)
        """
        return lengths * MONOMER_MASS + INITIATOR_MASS

    def compute_mn(self, chains: np.ndarray = None) -> float:
        """
        Compute number-average molecular weight (Mn).

        Mn = Σ(Ni × Mi) / Σ(Ni)

        For polymer chains: Mn = mean(MW) = mean(DP) × monomer_mass + initiator_mass

        Args:
            chains: Chain lengths to use (default: all_chains)

        Returns:
            Mn in g/mol
        """
        if chains is None:
            chains = self.all_chains()
        if len(chains) == 0:
            return 0.0

        mw = self.length_to_mw(chains)
        return float(np.mean(mw))

    def compute_mw(self, chains: np.ndarray = None) -> float:
        """
        Compute weight-average molecular weight (Mw).

        Mw = Σ(Ni × Mi²) / Σ(Ni × Mi)

        This weights heavier chains more than lighter ones.

        Args:
            chains: Chain lengths to use (default: all_chains)

        Returns:
            Mw in g/mol
        """
        if chains is None:
            chains = self.all_chains()
        if len(chains) == 0:
            return 0.0

        mw = self.length_to_mw(chains)
        # Mw = Σ(Mi²) / Σ(Mi) when Ni=1 for each chain
        return float(np.sum(mw ** 2) / np.sum(mw))

    def compute_pdi(self, chains: np.ndarray = None) -> float:
        """
        Compute polydispersity index (PDI or Đ).

        PDI = Mw / Mn

        PDI = 1.0 means perfectly monodisperse (all chains same length)
        PDI > 1.0 indicates distribution breadth

        Args:
            chains: Chain lengths to use (default: all_chains)

        Returns:
            PDI (dimensionless, >= 1.0)
        """
        mn = self.compute_mn(chains)
        if mn == 0:
            return 0.0
        mw = self.compute_mw(chains)
        return mw / mn

    def polymer_stats(self) -> dict:
        """
        Compute polymer characterization metrics.

        Returns dict with:
            - Mn: Number-average molecular weight (g/mol)
            - Mw: Weight-average molecular weight (g/mol)
            - PDI: Polydispersity index (Mw/Mn)
            - DP_n: Number-average degree of polymerization
            - DP_w: Weight-average degree of polymerization
        """
        chains = self.all_chains()
        if len(chains) == 0:
            return {'Mn': 0.0, 'Mw': 0.0, 'PDI': 0.0, 'DP_n': 0.0, 'DP_w': 0.0}

        mn = self.compute_mn(chains)
        mw = self.compute_mw(chains)
        pdi = mw / mn if mn > 0 else 0.0

        # Degree of polymerization (chain length averages)
        dp_n = float(np.mean(chains))  # Number-average DP
        dp_w = float(np.sum(chains ** 2) / np.sum(chains))  # Weight-average DP

        return {
            'Mn': mn,
            'Mw': mw,
            'PDI': pdi,
            'DP_n': dp_n,
            'DP_w': dp_w,
        }


@dataclass
class KineticsData:
    """
    Per-timestep kinetics data for polymer characterization.

    Stores Mn, Mw, PDI, and chain counts at each simulation timestep.
    Useful for chemists analyzing reaction kinetics and polymerization dynamics.

    Attributes:
        timesteps: Array of timestep indices
        mn: Number-average molecular weight at each timestep (g/mol)
        mw: Weight-average molecular weight at each timestep (g/mol)
        pdi: Polydispersity index at each timestep
        n_living: Number of living chains at each timestep
        n_dead: Number of dead chains at each timestep
        monomer_conversion: Fraction of monomers consumed at each timestep
    """
    timesteps: np.ndarray
    mn: np.ndarray
    mw: np.ndarray
    pdi: np.ndarray
    n_living: np.ndarray
    n_dead: np.ndarray
    monomer_conversion: np.ndarray

    def to_dataframe(self):
        """
        Convert to pandas DataFrame for easy export.

        Returns:
            pandas.DataFrame with kinetics data
        """
        import pandas as pd
        return pd.DataFrame({
            'timestep': self.timesteps,
            'Mn': self.mn,
            'Mw': self.mw,
            'PDI': self.pdi,
            'n_living': self.n_living,
            'n_dead': self.n_dead,
            'conversion': self.monomer_conversion,
        })

    def to_excel(self, path: str):
        """
        Export kinetics data to Excel file.

        Args:
            path: Output file path (.xlsx)
        """
        df = self.to_dataframe()
        df.to_excel(path, index=False, sheet_name='Kinetics')

    def to_csv(self, path: str):
        """
        Export kinetics data to CSV file.

        Args:
            path: Output file path (.csv)
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False)


@dataclass
class SimulationResult:
    """
    Complete result from simulation including optional kinetics.

    Attributes:
        distribution: Final chain length distribution
        kinetics: Per-timestep kinetics data (None if not tracked)
    """
    distribution: Distribution
    kinetics: Optional[KineticsData] = None


def _compute_kinetics_snapshot(
    living: np.ndarray,
    dead: np.ndarray,
    coupled: np.ndarray,
    current_monomer: float,
    initial_monomer: float
) -> Tuple[float, float, float, int, int, float]:
    """
    Compute kinetics metrics at a single timestep.

    Returns:
        (Mn, Mw, PDI, n_living, n_dead, conversion)
    """
    all_chains = np.concatenate([living, dead, coupled])
    n_living = len(living)
    n_dead = len(dead) + len(coupled)

    if len(all_chains) == 0:
        return (0.0, 0.0, 0.0, n_living, n_dead, 0.0)

    # Compute MW for all chains
    mw_array = all_chains * MONOMER_MASS + INITIATOR_MASS

    # Mn = mean(MW)
    mn = float(np.mean(mw_array))

    # Mw = sum(MW^2) / sum(MW)
    mw = float(np.sum(mw_array ** 2) / np.sum(mw_array))

    # PDI = Mw / Mn
    pdi = mw / mn if mn > 0 else 0.0

    # Conversion
    if initial_monomer > 0:
        conversion = 1.0 - (current_monomer / initial_monomer)
    else:
        conversion = 0.0  # Infinite monomer

    return (mn, mw, pdi, n_living, n_dead, conversion)


def simulate(
    params: SimulationParams,
    rng: np.random.Generator,
    track_kinetics: bool = False,
    kinetics_interval: int = 1
) -> Union["Distribution", "SimulationResult"]:
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
        track_kinetics: If True, track Mn/Mw/PDI at each timestep and return
                       SimulationResult instead of Distribution
        kinetics_interval: Record kinetics every N timesteps (default: 1)
                          Use higher values to reduce memory for long simulations

    Returns:
        If track_kinetics=False: Distribution of chain lengths (backward compatible)
        If track_kinetics=True: SimulationResult with distribution + kinetics data

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

        # With kinetics tracking:
        >>> result = simulate(params, rng, track_kinetics=True)
        >>> print(result.kinetics.to_dataframe())
        >>> result.kinetics.to_excel('kinetics.xlsx')
    """
    # Validate parameters
    params.validate()

    # Cache params as locals to avoid repeated attribute lookups (1000+ iterations)
    time_sim = params.time_sim
    p_growth = params.p_growth
    p_death = params.p_death
    p_dead_react = params.p_dead_react
    l_exponent = params.l_exponent
    d_exponent = params.d_exponent
    l_naked = params.l_naked
    kill_spawns_new = params.kill_spawns_new

    # Initialize pools
    living = np.ones(params.number_of_molecules, dtype=np.float64)
    dead = np.array([], dtype=np.float64)
    coupled = np.array([], dtype=np.float64)

    # Track monomer pool
    initial_monomer_pool = params.monomer_pool
    current_monomer_pool = params.monomer_pool

    # Pre-compute exponent ratios (used every timestep in vampiric section)
    le_ratio = l_exponent / l_naked
    de_ratio = d_exponent / l_naked

    # Initialize kinetics tracking if enabled
    if track_kinetics:
        kinetics_timesteps = []
        kinetics_mn = []
        kinetics_mw = []
        kinetics_pdi = []
        kinetics_n_living = []
        kinetics_n_dead = []
        kinetics_conversion = []

    # Main simulation loop
    for t in range(time_sim):
        if len(living) == 0:
            break  # No living chains left

        # Compute monomer ratio
        if current_monomer_pool < 0:
            monomer_ratio = 1.0  # Infinite monomer
        else:
            monomer_ratio = current_monomer_pool / initial_monomer_pool

        # Random numbers for fate decisions
        r = rng.random(len(living))

        # GROWTH: r < p_growth x monomer_ratio
        growth_threshold = p_growth * monomer_ratio
        will_grow = r < growth_threshold
        living[will_grow] += 1

        # Update monomer pool for growth
        if current_monomer_pool >= 0:
            monomers_used = np.sum(will_grow)
            current_monomer_pool = max(0, current_monomer_pool - monomers_used)

        # DEATH: p_growth x monomer_ratio <= r < (p_growth + p_death) x monomer_ratio
        death_threshold = (p_growth + p_death) * monomer_ratio
        will_die = (r >= growth_threshold) & (r < death_threshold)

        # Fancy indexing already returns a copy -- no .copy() needed
        new_dead = living[will_die]

        if kill_spawns_new:
            # Death spawns new chain -> reset to length 1
            living[will_die] = 1
            if current_monomer_pool >= 0:
                spawns = np.sum(will_die)
                current_monomer_pool = max(0, current_monomer_pool - spawns)
        else:
            # Death removes chain -> delete from living
            living = living[~will_die]

        # VAMPIRIC REACTIONS (before adding new_dead, matching Thomas)
        n_dead = len(dead)
        n_living = len(living)
        if n_dead > 0 and n_living > 0:
            # Each dead chain picks a random target (living or dead)
            n_targets = n_dead + n_living
            # Use ceil(random*n - 1) pattern matching Thomas's RNG usage
            which_chain_attacked = np.ceil(
                rng.random(n_dead) * n_targets - 1).astype(int)

            # Filter: only dead chains that picked living targets
            dead_vampiric = np.where(which_chain_attacked > (n_dead - 1))[0]

            if len(dead_vampiric) > 0:
                which_living_attacked = which_chain_attacked[dead_vampiric] - n_dead

                # Compute success probabilities inline (matching Thomas's np.float_power)
                l_liv = living[which_living_attacked]
                l_ded = dead[dead_vampiric]

                # Roll for success
                r_success = rng.random(len(which_living_attacked))

                p_success = p_dead_react / (
                    np.float_power(l_liv, np.minimum(l_liv * le_ratio, l_exponent))
                    * np.float_power(l_ded, np.minimum(l_ded * de_ratio, d_exponent))
                )

                # Apply successful couplings directly (no np.any guard needed)
                success_mask = np.where(r_success < p_success)
                living[which_living_attacked[success_mask]] += dead[dead_vampiric[success_mask]]

                still_dead = np.ones(n_dead)
                still_dead[dead_vampiric[success_mask]] = 0
                dead = dead[still_dead == 1]

        # Add newly dead chains AFTER vampiric reactions (matches Thomas)
        if len(new_dead) > 0:
            dead = np.hstack((dead, new_dead))

        # Track kinetics at interval
        if track_kinetics and (t % kinetics_interval == 0):
            mn, mw, pdi, n_liv, n_ded, conv = _compute_kinetics_snapshot(
                living, dead, coupled, current_monomer_pool, initial_monomer_pool
            )
            kinetics_timesteps.append(t)
            kinetics_mn.append(mn)
            kinetics_mw.append(mw)
            kinetics_pdi.append(pdi)
            kinetics_n_living.append(n_liv)
            kinetics_n_dead.append(n_ded)
            kinetics_conversion.append(conv)

    # Build final distribution
    distribution = Distribution(living=living, dead=dead, coupled=coupled)

    if track_kinetics:
        # Final snapshot
        if params.time_sim % kinetics_interval != 0:
            mn, mw, pdi, n_liv, n_ded, conv = _compute_kinetics_snapshot(
                living, dead, coupled, current_monomer_pool, initial_monomer_pool
            )
            kinetics_timesteps.append(params.time_sim)
            kinetics_mn.append(mn)
            kinetics_mw.append(mw)
            kinetics_pdi.append(pdi)
            kinetics_n_living.append(n_liv)
            kinetics_n_dead.append(n_ded)
            kinetics_conversion.append(conv)

        kinetics = KineticsData(
            timesteps=np.array(kinetics_timesteps),
            mn=np.array(kinetics_mn),
            mw=np.array(kinetics_mw),
            pdi=np.array(kinetics_pdi),
            n_living=np.array(kinetics_n_living),
            n_dead=np.array(kinetics_n_dead),
            monomer_conversion=np.array(kinetics_conversion)
        )
        return SimulationResult(distribution=distribution, kinetics=kinetics)

    return distribution