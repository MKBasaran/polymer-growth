"""Load and preprocess experimental polymer distribution data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_experimental_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load experimental polymer distribution from Excel file.

    Expected format:
        Column 0: Molar mass (g/mol)
        Column 1: Linear differential molar mass (distribution values)

    Conversion:
        Chain length = (molar_mass - 180) / 99.13

    Args:
        file_path: Path to Excel file (.xls or .xlsx)

    Returns:
        Tuple of (chain_lengths, distribution_values)
        - chain_lengths: Array of polymer chain lengths (integers)
        - distribution_values: Corresponding distribution values

    Example:
        >>> lengths, values = load_experimental_data("data/experimental.xlsx")
        >>> print(f"Max chain length: {lengths.max()}")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Experimental data file not found: {file_path}")

    # Read Excel file
    df = pd.read_excel(file_path)

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns, got {df.shape[1]}")

    # Extract columns
    molar_mass = df[df.columns[0]].values
    dist_values = df[df.columns[1]].values

    # Convert molar mass to chain lengths
    # Formula from legacy code: cl = (mm - 180) / 99.13
    chain_lengths = ((molar_mass - 180) / 99.13).astype(int)

    # Create dictionary mapping chain length -> value
    max_length = chain_lengths.max()
    cl_to_val = {cl: 0.0 for cl in range(max_length + 1)}

    # Fill values
    for cl, val in zip(chain_lengths, dist_values):
        cl_to_val[cl] = val

    # Forward-fill zeros (use previous value if current is zero)
    for cl in range(1, max_length + 1):
        if cl_to_val[cl] == 0:
            cl_to_val[cl] = cl_to_val[cl - 1]

    # Convert to arrays
    result_lengths = np.arange(max_length + 1)
    result_values = np.array([cl_to_val[cl] for cl in result_lengths])

    return result_lengths, result_values


def preprocess_simulation_histogram(
    living: np.ndarray,
    dead: np.ndarray,
    coupled: np.ndarray,
    target_length: int
) -> np.ndarray:
    """
    Convert simulation output to histogram matching experimental data length.

    Args:
        living: Living chain lengths
        dead: Dead chain lengths
        coupled: Coupled chain lengths
        target_length: Length to match (from experimental data)

    Returns:
        Histogram of all chains, padded/truncated to target_length
    """
    # Combine all chains
    all_chains = np.concatenate([living, dead, coupled])

    if len(all_chains) == 0:
        return np.zeros(target_length)

    # Compute histogram
    max_length = int(all_chains.max())
    hist, _ = np.histogram(all_chains, bins=np.arange(max_length + 2))

    # Pad or truncate to match target length
    if len(hist) < target_length:
        # Pad with zeros
        hist = np.concatenate([hist, np.zeros(target_length - len(hist))])
    elif len(hist) > target_length:
        # Truncate
        hist = hist[:target_length]

    return hist