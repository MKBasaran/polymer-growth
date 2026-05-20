# Polymer Growth Simulator and FDDC Optimiser

This repository contains a polymer growth simulator and a Fitness-Diversity Driven Co-evolution (FDDC) optimiser. The package implements an agent-based stochastic simulator for poly(2-ethyl-2-oxazoline) chain growth and a genetic algorithm that infers kinetic parameters from gel permeation chromatography data. The `program code/` directory contains Thomas van den Broek's earlier 2020 implementation, preserved unmodified.

## Requirements

- Python 3.10 is the validated version. Later 3.10.x patch releases are expected to work; 3.11 and 3.12 are accepted by the package but were not used for the validation suite.
- NumPy 1.24–1.x, SciPy 1.10+, pandas 1.5+, matplotlib 3.7+. Full pins are in `pyproject.toml`.
- PySide6 6.4+ for the GUI (optional extra).
- macOS or Linux for the `scripts/experiments/` runners, which use POSIX `fork` start method to share state with worker subprocesses. The GUI and the core simulator run on Windows; only the experiment dispatch scripts are POSIX-only.

## Installation

```bash
git clone <repository-url>
cd polymer-growth
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,dev]"
```

## Quick start

Run the test suite (29 tests, completes in under thirty seconds):
```bash
pytest
```

Launch the GUI:
```bash
polymer-sim gui
```

Run a single simulation from the command line:
```bash
polymer-sim simulate --time 1000 --molecules 10000 --seed 42
```

Run one of the experiment scripts:
```bash
python scripts/experiments/exp_determinism.py
```
Results land in `validation_results/` as JSON files, one per run. The experiment scripts are:

- `exp_A_speed.py` — wall-clock speed comparison between codebases
- `exp_B_scaling.py` — parallel scaling across worker counts
- `exp_C_datasets.py` and `exp_D_validation.py` — cross-dataset cost comparison
- `exp_determinism.py` — bitwise-identical cost histories at a fixed seed
- `exp_mechanism.py` — per-call speedup isolation

`exp_A_speed.py` and `exp_B_scaling.py` require several hours each; the others complete in under an hour on the reference hardware.

## Repository layout

```
src/polymer_growth/    Refactored package (core simulator, FDDC optimiser, CLI, GUI)
tests/                 Pytest suite (29 tests)
scripts/experiments/   Experiment runners
validation_results/    JSON outputs from the experiment runs
program code/          Thomas van den Broek's earlier 2020 implementation, unmodified.
docs/                  User manual and GUI screenshots
```

The reference hardware for all measurements is an Apple M4 Pro (14 CPU cores, 48 GB unified memory, macOS Sequoia, Python 3.10).

## Citation

The experiments referenced by the scripts in `scripts/experiments/` are
documented in:

    Basaran, K. (2026). Modernising Legacy Scientific Simulation Code:
    A Case Study in Polymer Growth Modelling. BSc thesis, Department of
    Advanced Computing Sciences, Maastricht University.

The earlier implementation under `program code/` is from:

    van den Broek, T. (2020). Genetic Algorithms to Better Understand
    Polymer Growth. Bachelor's thesis, Maastricht University, Department
    of Advanced Computing Sciences (formerly Department of Data Science
    and Knowledge Engineering).

## License

MIT. See `LICENSE`.
