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

from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, List, Dict, Union
import numpy as np
from numba import jit


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
    """
    living_exp = np.minimum(l_living * (l_exponent / l_naked), l_exponent)
    dead_exp = np.minimum(l_dead * (d_exponent / l_naked), d_exponent)
    denominator = np.power(l_living, living_exp) * np.power(l_dead, dead_exp)
    return p_dead_react / denominator


@jit(nopython=True)
def _simulate_core(
    time_sim: int,
    n_molecules: int,
    monomer_pool: float,
    p_growth: float,
    p_death: float,
    p_dead_react: float,
    l_exponent: float,
    d_exponent: float,
    l_naked: float,
    kill_spawns_new: bool,
    seed: int
):
    """Full simulation loop compiled to machine code via Numba.

    Pre-allocates all arrays at max possible size. Uses index counters
    instead of dynamic array resizing. Returns raw buffers + counts.

    Returns:
        (living_buf, n_living, dead_buf, n_dead, final_monomer_pool)
    """
    np.random.seed(seed)

    # Pre-allocate at max possible sizes
    # Living: starts at n_molecules. With kill_spawns_new=True, stays constant.
    # Without, can only shrink. Max is n_molecules.
    max_living = n_molecules
    living = np.ones(max_living, dtype=np.float64)
    n_living = n_molecules

    # Dead: each timestep can add at most n_living dead chains.
    # Over all timesteps, bounded by n_molecules (each chain can die once).
    # With vampiric coupling removing dead chains, practical max is n_molecules.
    max_dead = n_molecules * 2  # Safety margin for edge cases
    dead = np.empty(max_dead, dtype=np.float64)
    n_dead = 0

    initial_monomer_pool = monomer_pool
    current_monomer_pool = monomer_pool

    for t in range(time_sim):
        if n_living == 0:
            break

        # Monomer ratio
        if current_monomer_pool < 0.0:
            monomer_ratio = 1.0
        else:
            monomer_ratio = current_monomer_pool / initial_monomer_pool

        # Random fate for each living chain
        growth_threshold = p_growth * monomer_ratio
        death_threshold = (p_growth + p_death) * monomer_ratio

        # Count new dead for this timestep
        new_dead_count = 0

        if kill_spawns_new:
            # GROWTH + DEATH with respawn: living array stays same size
            for i in range(n_living):
                r = np.random.random()
                if r < growth_threshold:
                    # Growth
                    living[i] += 1.0
                    if current_monomer_pool >= 0.0:
                        current_monomer_pool -= 1.0
                        if current_monomer_pool < 0.0:
                            current_monomer_pool = 0.0
                elif r < death_threshold:
                    # Death: add to dead, respawn as length 1
                    if n_dead < max_dead:
                        dead[n_dead] = living[i]
                        n_dead += 1
                    living[i] = 1.0
                    if current_monomer_pool >= 0.0:
                        current_monomer_pool -= 1.0
                        if current_monomer_pool < 0.0:
                            current_monomer_pool = 0.0
        else:
            # GROWTH + DEATH without respawn: living shrinks
            write_idx = 0
            for i in range(n_living):
                r = np.random.random()
                if r < growth_threshold:
                    # Growth
                    living[write_idx] = living[i] + 1.0
                    write_idx += 1
                    if current_monomer_pool >= 0.0:
                        current_monomer_pool -= 1.0
                        if current_monomer_pool < 0.0:
                            current_monomer_pool = 0.0
                elif r < death_threshold:
                    # Death: add to dead, don't keep in living
                    if n_dead < max_dead:
                        dead[n_dead] = living[i]
                        n_dead += 1
                else:
                    # Survives unchanged
                    living[write_idx] = living[i]
                    write_idx += 1
            n_living = write_idx

        # VAMPIRIC REACTIONS
        if n_dead > 0 and n_living > 0:
            n_total = n_dead + n_living

            # Each dead chain picks a random target
            # Process attacks: dead chains that picked a living target
            dead_survived = np.ones(n_dead, dtype=np.int8)

            for d_idx in range(n_dead):
                target = int(np.random.random() * n_total)
                if target >= n_dead:
                    # Attacks a living chain
                    living_idx = target - n_dead
                    if living_idx < n_living:
                        # Compute success probability
                        l_liv = living[living_idx]
                        l_ded = dead[d_idx]
                        f_living = min(l_liv * (l_exponent / l_naked), l_exponent)
                        f_dead = min(l_ded * (d_exponent / l_naked), d_exponent)
                        denom = (l_liv ** f_living) * (l_ded ** f_dead)
                        p_success = p_dead_react / denom if denom > 0.0 else 0.0

                        r_success = np.random.random()
                        if r_success < p_success:
                            # Successful coupling: living absorbs dead
                            living[living_idx] += dead[d_idx]
                            dead_survived[d_idx] = 0

            # Compact dead array: remove successful couplings
            write_idx = 0
            for d_idx in range(n_dead):
                if dead_survived[d_idx] == 1:
                    dead[write_idx] = dead[d_idx]
                    write_idx += 1
            n_dead = write_idx

    return living, n_living, dead, n_dead, current_monomer_pool


def _simulate_fast(params: 'SimulationParams', seed: int) -> 'Distribution':
    """Fast simulation using Numba JIT core. Drop-in replacement for simulate()."""
    params.validate()

    living_buf, n_living, dead_buf, n_dead, _ = _simulate_core(
        time_sim=params.time_sim,
        n_molecules=params.number_of_molecules,
        monomer_pool=params.monomer_pool,
        p_growth=params.p_growth,
        p_death=params.p_death,
        p_dead_react=params.p_dead_react,
        l_exponent=params.l_exponent,
        d_exponent=params.d_exponent,
        l_naked=params.l_naked,
        kill_spawns_new=params.kill_spawns_new,
        seed=seed,
    )

    return Distribution(
        living=living_buf[:n_living].copy(),
        dead=dead_buf[:n_dead].copy(),
        coupled=np.array([], dtype=np.float64),
    )


def _simulate_fast_hist(params_tuple, seed: int, target_length: int) -> np.ndarray:
    """Simulate and return histogram directly. No Distribution object.

    Returns a numpy array of ~target_length ints -- tiny for IPC (~4KB vs 560KB).
    Used by megabatch workers to minimize pickle overhead.
    """
    (time_sim, n_mol, monomer_pool, p_growth, p_death,
     p_dead_react, l_exp, d_exp, l_naked, kill_spawns) = params_tuple

    living_buf, n_living, dead_buf, n_dead, _ = _simulate_core(
        time_sim=int(time_sim),
        n_molecules=int(n_mol),
        monomer_pool=float(monomer_pool),
        p_growth=float(p_growth),
        p_death=float(p_death),
        p_dead_react=float(p_dead_react),
        l_exponent=float(l_exp),
        d_exponent=float(d_exp),
        l_naked=float(l_naked),
        kill_spawns_new=bool(round(kill_spawns)),
        seed=int(seed) % (2**31),
    )

    living = living_buf[:n_living]
    dead = dead_buf[:n_dead]

    # Build histogram in-place -- no Distribution object
    all_chains = np.concatenate([living, dead]) if n_dead > 0 else living
    if len(all_chains) == 0:
        return np.zeros(target_length, dtype=np.float64)

    max_len = int(all_chains.max())
    hist, _ = np.histogram(all_chains, bins=np.arange(max_len + 2))
    hist = hist.astype(np.float64)

    if len(hist) < target_length:
        hist = np.concatenate([hist, np.zeros(target_length - len(hist))])
    elif len(hist) > target_length:
        hist = hist[:target_length]

    return hist


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

    # Initialize pools
    living = np.ones(params.number_of_molecules, dtype=np.float64)
    dead = np.array([], dtype=np.float64)
    coupled = np.array([], dtype=np.float64)

    # Track monomer pool
    initial_monomer_pool = params.monomer_pool
    current_monomer_pool = params.monomer_pool

    # Initialize kinetics tracking if enabled
    if track_kinetics:
        n_records = (params.time_sim // kinetics_interval) + 1
        kinetics_timesteps = []
        kinetics_mn = []
        kinetics_mw = []
        kinetics_pdi = []
        kinetics_n_living = []
        kinetics_n_dead = []
        kinetics_conversion = []

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