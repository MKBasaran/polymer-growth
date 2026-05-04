# Polymer Growth Simulation & Optimization

A Python tool for simulating agent-based polymer chain growth and fitting simulation parameters to experimental data using genetic algorithms.

**Based on:** Thomas van den Broek (2020) - "Genetic Algorithms To Better Understand Polymer Growth"

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MKBasaran/polymer-growth.git
cd polymer-growth

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies (including GUI)
pip install -e ".[gui,dev]"
```

### Run the GUI (Recommended)

```bash
python -m polymer_growth.cli.main gui
```

This opens a graphical interface with two tabs:
1. **Simulation Tab** - Run single simulations with custom parameters
2. **Optimization Tab** - Fit parameters to experimental data using FDDC

### Run via CLI

```bash
# Run a simulation
python -m polymer_growth.cli.main simulate --time 1000 --molecules 10000 --seed 42

# Fit to experimental data
python -m polymer_growth.cli.main fit data/experimental.xlsx --generations 20 --seed 42
```

## Features

### Simulation

- Agent-based stochastic polymer growth model
- 10 configurable parameters (see Table I in thesis)
- Three reaction types: Growth, Death (termination), Vampiric coupling
- Chemistry metrics: Mn, Mw, PDI (polydispersity index)
- Per-timestep kinetics tracking and export

### Optimization (FDDC)

- **FDDC: Fitness-Diversity Driven Co-evolution** - Best performing algorithm from thesis
- Two co-evolving populations: Solutions (parameters) vs Problems (sigma weights)
- Memory-based fitness to handle stochastic simulation noise
- Novelty-based ranking for population diversity
- **Parallel evaluation** using multiple CPU cores (configurable workers)
- Automatic run output organization in `runs/` directory

### Comparison to Other Algorithms

| Algorithm | Avg Generations | Performance |
|-----------|-----------------|-------------|
| Basic GA | ~77 | Baseline |
| Island Model | ~41 | 1.9x faster |
| **FDDC (used)** | ~20 | **3.9x faster** |

## Project Structure

```
polymer-growth/
├── src/polymer_growth/
│   ├── core/                  # Simulation engine
│   │   ├── simulation.py      # Main simulation with Mn/Mw/PDI
│   │   └── run_manager.py     # Output organization
│   ├── optimizers/
│   │   └── fddc.py            # FDDC optimizer with parallel eval
│   ├── objective/             # Cost functions (MinMaxV2)
│   ├── cli/                   # Command-line interface
│   └── gui/                   # PySide6 graphical interface
├── tests/                     # Test suite (29 tests)
├── standalone_prototypes/     # Algorithm prototypes
└── program code/              # Legacy thesis code + experimental data
```

## Simulation Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `time_sim` | Simulation timesteps | 100 - 10000 |
| `number_of_molecules` | Initial polymer chains | 1000 - 100000 |
| `monomer_pool` | Initial monomers (-1 = infinite) | 10000 - 100000000 |
| `p_growth` | Growth probability per timestep | 0.1 - 0.99 |
| `p_death` | Death (termination) probability | 0.00001 - 0.01 |
| `p_dead_react` | Vampiric reaction probability | 0.1 - 0.99 |
| `l_exponent` | Living chain length exponent | 0.1 - 0.99 |
| `d_exponent` | Dead chain length exponent | 0.1 - 0.99 |
| `l_naked` | Accessible surface ratio | 0.1 - 0.99 |
| `kill_spawns_new` | Death spawns new chain? | True/False |

## Chemistry Metrics

The simulation computes standard polymer characterization metrics:

- **Mn** (Number-average MW): Mean molecular weight
- **Mw** (Weight-average MW): Weight-weighted mean (heavier chains count more)
- **PDI** (Polydispersity Index): Mw/Mn (1.0 = monodisperse, >1 = broad distribution)

Chemistry constants:
- Monomer: 2-ethyl-2-oxazoline (99.13 g/mol)
- Initiator: Methyl tosylate (180.0 g/mol)

## Requirements

- Python >= 3.9
- NumPy, Pandas, Matplotlib, SciPy
- Numba (JIT compilation for performance)
- PySide6 (for GUI)
- Click (for CLI)

## Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_simulation.py -v
```

All 29 tests passing.

## Output Organization

When you run optimizations, results are automatically saved to timestamped directories:

```
runs/
├── optimization_dataset5K_20260305_143022/
│   ├── config.json          # FDDC configuration used
│   ├── optimization_results.json  # Best parameters, cost history
│   └── cost_history.csv     # Per-generation cost
└── simulation_test_20260305_143055/
    ├── params.json          # Simulation parameters
    ├── distribution.json    # Final chain distribution
    └── kinetics.csv         # Per-timestep Mn/Mw/PDI (if tracked)
```

## Example Usage

### Python API

```python
import numpy as np
from polymer_growth.core import simulate, SimulationParams

# Create parameters
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

# Run simulation
rng = np.random.default_rng(42)
dist = simulate(params, rng)

# Get results
print(dist.stats())           # Chain counts, mean/max length
print(dist.polymer_stats())   # Mn, Mw, PDI

# With kinetics tracking
result = simulate(params, rng, track_kinetics=True)
result.kinetics.to_csv('kinetics.csv')
result.kinetics.to_excel('kinetics.xlsx')
```

### Optimization API

```python
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from polymer_growth.objective import MinMaxV2ObjectiveFunction

# Load experimental data and create objective
objective = MinMaxV2ObjectiveFunction(experimental_distribution)

# Configure optimizer
config = FDDCConfig(
    population_size=50,
    max_generations=20,
    n_workers=6  # Parallel evaluation
)

# Run optimization
optimizer = FDDCOptimizer(bounds, objective_wrapper, config=config)
result = optimizer.optimize(seed=42)

print(f"Best cost: {result.best_cost}")
print(f"Best params: {result.best_params}")
```

## CLI vs GUI Feature Comparison

| Feature | CLI | GUI |
|---------|-----|-----|
| Run simulation | Yes | Yes |
| All 10 parameters | Fixed defaults | Adjustable |
| FDDC optimization | Yes | Yes |
| Progress display | Text | Progress bar + plot |
| Parallel evaluation | Yes (auto) | Yes (6 workers) |
| Run output saving | Manual | Automatic |
| Kinetics tracking | Supported | Not exposed yet |
| Mn/Mw/PDI display | In stats | Not shown yet |

## License

MIT License

## References

- van den Broek, T. (2020). Genetic Algorithms To Better Understand Polymer Growth. Thesis.
- Monnery, B.D., et al. (2018). Defined high molar mass poly(2-oxazoline)s. Angewandte Chemie.

## Contact

Kaan Basaran - Thesis Project 2026

*Last updated: 2026-03-05*
