# Polymer Growth Parameter Inference

**Scientific Software for Stochastic Polymer Growth Simulation & Parameter Optimization**

> This repository contains polymer growth modeling code transitioning from thesis research code to production-quality open-source software.

---

## 🚨 Current Status: LEGACY CODE → REFACTORING IN PROGRESS

**What's Here:**
- ✅ Working polymer growth simulation (agent-based, stochastic)
- ✅ Multiple genetic algorithm optimizers (baseline GA, island model, co-evolution, FDDC)
- ✅ Multiple cost functions for distribution comparison
- ✅ Tkinter GUI for experiments
- ✅ Thesis-validated algorithms (FDDC: 20 generation average to convergence)

**What's Missing:**
- ❌ No reproducibility controls (seed management)
- ❌ No tests
- ❌ No command-line interface
- ❌ Poor performance (no JIT, inefficient loops)
- ❌ Hardcoded paths and magic numbers everywhere
- ❌ No documentation

**🎯 Goal:** Transform into pip-installable package with CLI, docs, tests, and 10x better performance.

---

## 📋 Documentation

**Start here:**
1. **[NEXT_STEPS.md](./NEXT_STEPS.md)** - What to do right now
2. **[AUDIT_REPORT.md](./AUDIT_REPORT.md)** - Comprehensive codebase analysis (10,000+ words)
3. **[CODEBASE_MAP.md](./CODEBASE_MAP.md)** - Quick reference guide to all files
4. **[thesis.txt](./thesis.txt)** - Original thesis (source of truth for algorithms)

---

## 🚀 Quick Start (Legacy Code)

### Option 1: Run GUI
```bash
cd "program code"
python user_interface.py
```

Then:
1. Select algorithm (recommend: "fitness diversity driven coevolution")
2. Choose cost function (recommend: "Min max V2")
3. Select data file: `Data/5k no BB.xlsx`
4. Set parameter bounds
5. Click "Run"

### Option 2: Run Reproducibility Test
```bash
python legacy_reproduce.py
```

This runs the FDDC optimizer on synthetic data with controlled seed. Results saved to `results/YYYYMMDD_HHMMSS/`.

---

## 🔬 Scientific Background

### The Problem
Polymers (repeating chain molecules) are used in medicine for controlled drug release. The drug release rate depends on polymer length. To synthesize polymers with the right length distribution, we need to know the probabilities of different chemical reactions during synthesis.

### The Model
We simulate polymer growth as a stochastic agent-based process:
- Each chain is an agent with a length
- At each timestep, chains can:
  1. **Grow** (add one monomer unit) with probability `p_growth × monomer_ratio`
  2. **Die** (stop growing) with probability `p_death × monomer_ratio`
  3. **React vampirically** (dead chain couples with living chain) with probability depending on both chain lengths

The simulation is governed by 10 parameters (see [thesis.txt](./thesis.txt) Table I for full descriptions).

### The Challenge
Given an experimental polymer length distribution, find the 10 parameter values that best reproduce it. This is a difficult optimization problem:
- Stochastic simulation (noise in fitness function)
- Multi-modal landscape (many local optima)
- 10-dimensional search space

### The Solution (from Thesis)
Three genetic algorithm variants were tested:
1. **Baseline GA** (various selection methods): 77 generations average
2. **Island Model** (4 islands with migration): 41 generations average
3. **Fitness-Diversity Driven Co-evolution (FDDC)**: **20 generations average** ✅ WINNER

FDDC uses two populations:
- Population 1: Candidate parameter solutions
- Population 2: Sigma weights (emphasize different parts of distribution)

The populations co-evolve, with novelty search preventing premature convergence.

---

## 📂 Repository Structure

```
thesis_try/
│
├── AUDIT_REPORT.md           ⭐ Detailed analysis of legacy code
├── NEXT_STEPS.md             ⭐ Immediate action plan
├── CODEBASE_MAP.md           ⭐ File navigation guide
├── thesis.txt                   Original thesis text
├── legacy_reproduce.py          Reproducibility test script
│
├── program code/             📁 LEGACY CODE (as-is from thesis)
│   ├── simulation.py            Core stochastic simulation (227 lines)
│   ├── distributionComparison.py Cost functions (508 lines)
│   ├── fddc.py                  FDDC optimizer (515 lines) ⭐ BEST
│   ├── GA_base.py               Baseline GA (649 lines)
│   ├── island_GA.py             Island model (318 lines)
│   ├── co_evolution.py          Co-evolution (300 lines)
│   ├── user_interface.py        Tkinter GUI (798 lines)
│   ├── helper.py                Parameter validation
│   ├── data_generator.py        Synthetic data creator
│   ├── Data/                    Experimental polymer distributions
│   └── fakeData/                Synthetic validation data
│
├── polymer_growth/           📁 NEW PACKAGE (to be created)
│   ├── core/                    Refactored simulation engine
│   ├── objective/               Cost functions
│   ├── optimizers/              GA variants
│   ├── io/                      Data loading/saving
│   └── cli/                     Command-line interface
│
├── tests/                    📁 TEST SUITE (to be created)
├── examples/                 📁 EXAMPLES (to be created)
├── docs/                     📁 DOCUMENTATION (to be created)
└── configs/                  📁 EXPERIMENT CONFIGS (to be created)
```

---

## 🧪 How It Works Today (Legacy)

### Example: Fit parameters to experimental data

```python
import numpy as np
from program_code.simulation import polymer
from program_code.distributionComparison import min_maxV2
from program_code.fddc import fddc

# Define parameter bounds
bounds = np.array([
    [100, 3000],           # time_sim
    [10000, 120000],       # number_of_molecules
    [1000000, 5000000],    # monomer_pool
    [0.1, 0.99],           # p_growth
    [0.0001, 0.002],       # p_death
    [0.1, 0.9],            # p_dead_react
    [0.1, 0.9],            # l_exponent
    [0.1, 0.9],            # d_exponent
    [0.1, 1.0],            # l_naked
    [0, 1]                 # kill_spawns_new
])

# Create cost function
cost_func = min_maxV2("program code/Data/5k no BB.xlsx", polymer)

# Create FDDC optimizer
optimizer = fddc(
    bounds=bounds,
    fitnessFunction=cost_func.costFunction,
    distribution_comparison=cost_func,
    populationSize=50
)

# Run optimization
for generation in range(200):
    optimizer.run()
    print(f"Gen {generation}: Best cost = {optimizer.best_score}")
    if optimizer.best_score < 10.0:
        break

print(f"Best parameters: {optimizer.best}")
```

**Issues with this workflow:**
- No seed control (not reproducible)
- Manual loop (no clean stopping condition)
- Results not saved automatically
- No progress tracking
- Hardcoded paths

---

## 🎯 Future Usage (After Refactor)

### Command-line interface
```bash
# Simulate with fixed parameters
polymer-sim simulate --config params.yml --seed 42 --out results/

# Fit to experimental data
polymer-sim fit \
  --target data/5k_no_bb.csv \
  --optimizer fddc \
  --config configs/bounds_default.yml \
  --seed 42 \
  --out results/fit_5k/

# Reproduce thesis result
polymer-sim reproduce thesis-fddc-5k --out results/
```

### Python API
```python
from polymer_growth.core import simulate
from polymer_growth.objective import MinMaxV2Objective
from polymer_growth.optimizers import FDDC
from polymer_growth.io import load_target_distribution

# Load experimental data
target = load_target_distribution("data/5k_no_bb.csv")

# Create objective function
objective = MinMaxV2Objective(target_distribution=target)

# Create optimizer
optimizer = FDDC(
    objective=objective,
    bounds=bounds_dict,
    population_size=50,
    seed=42
)

# Run optimization
result = optimizer.optimize(max_generations=200, threshold=10.0)

# Save results
result.save("results/fit_5k/")
result.plot_trajectory("results/fit_5k/trajectory.png")
```

---

## 🔧 Development Roadmap

### Phase 0: Legacy Reproducibility (Week 1) ✅ IN PROGRESS
- [x] Audit complete codebase
- [x] Create reproducibility script
- [ ] Document all workflows
- [ ] Pin dependencies

### Phase 1: Core Simulation (Week 2-3)
- [ ] Extract pure simulation function
- [ ] Add comprehensive tests
- [ ] Add Numba JIT (target: 2-5x speedup)
- [ ] RNG seeding control

### Phase 2: Objective Functions (Week 4)
- [ ] Standardize cost function interface
- [ ] Support CSV/NumPy (not just Excel)
- [ ] Separate plotting from computation
- [ ] Tests for normalization correctness

### Phase 3: Optimizers (Week 5-6)
- [ ] Refactor FDDC (priority)
- [ ] Refactor GA_base, island_GA
- [ ] Remove all global state
- [ ] Parallel execution with configurable workers

### Phase 4: CLI & Config (Week 7)
- [ ] Click-based CLI
- [ ] YAML configuration files
- [ ] Structured logging
- [ ] Subcommands: simulate, fit, reproduce, benchmark

### Phase 5: Performance (Week 8)
- [ ] Numba JIT critical loops (5x speedup)
- [ ] Parallel population eval (2x speedup)
- [ ] Caching & vectorization (2x speedup)
- [ ] **Target: 10x total speedup**

### Phase 6: Documentation & Release (Week 9-10)
- [ ] Theory documentation (from thesis)
- [ ] API reference (auto-generated)
- [ ] Tutorials & examples
- [ ] GitHub Actions CI
- [ ] PyPI release: `pip install polymer-growth`
- [ ] Zenodo DOI for citation
- [ ] CITATION.cff file

---

## 📊 Algorithm Performance (from Thesis)

| Algorithm                | Mean Generations | Std Dev | Notes                    |
|--------------------------|------------------|---------|--------------------------|
| Baseline GA (Roulette)   | 77.3             | 32.5    | Population 100           |
| Island Model (4 islands) | 41.3             | 17.6    | Circular migration, 7 gen|
| **FDDC**                 | **20.1**         | **10.6**| **Best performer** ⭐    |

*Experiments on synthetic data with known optimum, convergence threshold: cost < 10*

---

## 🧮 Model Parameters

From [thesis.txt](./thesis.txt) Table I:

| Parameter               | Symbol | Range          | Description                                |
|-------------------------|--------|----------------|--------------------------------------------|
| simulation_time         | -      | [100, 3000]    | Number of timesteps                        |
| number_of_molecules     | -      | [1e4, 1.2e5]   | Initial chain count                        |
| monomer_pool            | -      | [1e6, 5e6]     | Available monomer units                    |
| probability_of_growth   | pg     | [0.1, 0.99]    | Growth probability per timestep            |
| probability_of_death    | pd     | [1e-4, 2e-3]   | Death (chain termination) probability      |
| probability_dead_react  | pdr    | [0.1, 0.9]     | Vampiric coupling base probability         |
| living_exponent         | le     | [0.1, 0.9]     | Living chain length exponent (coupling)    |
| death_exponent          | de     | [0.1, 0.9]     | Dead chain length exponent (coupling)      |
| living_naked            | ln     | [0.1, 1.0]     | Accessible surface ratio                   |
| death_spawns_new_monomer| -      | {0, 1}         | Whether death events spawn new chains      |

### Key Equations (from thesis)

**Monomer ratio:**
```
monomer_ratio = current_monomer_pool / initial_monomer_pool
```

**Growth condition:**
```
if random() < pg × monomer_ratio:
    chain_length += 1
```

**Death condition:**
```
if pg × monomer_ratio ≤ random() < (pg + pd) × monomer_ratio:
    chain becomes dead
```

**Vampiric coupling success probability:**
```
p_success = pdr / (
    l_living^min(l_living × le/ln, le) ×
    l_dead^min(l_dead × de/ln, de)
)
```

---

## 📦 Dependencies

### Current (Legacy)
```
numpy>=1.21.0
pandas>=1.3.0        # Excel loading
matplotlib>=3.4.0
scipy>=1.7.0
scikit-learn>=0.24.0
```

### Future (Refactored)
```
# Core
numpy>=1.21.0
numba>=0.54.0        # JIT compilation

# I/O & Config
pandas>=1.3.0
pyyaml>=5.4.0

# CLI
click>=8.0.0
structlog>=21.0.0

# Visualization
matplotlib>=3.4.0

# Development
pytest>=6.2.0
ruff>=0.1.0
mypy>=0.910
```

---

## 📚 References

1. **Original Simulation:** Van Appeven et al. (2018) - Understanding Polymer Growth
2. **Selection Methods:** Saini (2017) - Review of Selection Methods in Genetic Algorithms
3. **Island Model:** Herrera & Lozano (2000) - Gradual Distributed Genetic Algorithms
4. **Co-evolution:** Paredis (1995) - Coevolutionary Computation
5. **FDDC:** Franz et al. (2017) - On the Combination of Coevolution and Novelty Search
6. **Experimental Data:** Monnery et al. (2018) - Defined High Molar Mass Poly(2-oxazoline)s

Full bibliography in [thesis.txt](./thesis.txt).

---

## 🤝 Contributing

**Current Status:** Refactoring in progress. Not yet accepting external contributions.

Once refactored (Phase 6):
- Contribution guidelines in CONTRIBUTING.md
- Code style: Black + Ruff
- Type hints: mypy strict mode
- Tests: pytest with 90%+ coverage
- Documentation: MkDocs

---

## 📄 License

To be determined (likely MIT or Apache 2.0 for open source release).

---

## 📧 Contact

**Author:** Kaan Basaran (Thesis Project)
**Institution:** [Your University]
**Supervisor:** [Supervisor Name]

**Issues:** (Will create GitHub Issues after Phase 6)

---

## 🙏 Acknowledgments

- Dr. Bryn D. Monnery for providing experimental data and domain expertise
- Rachel Cavill for supervision and guidance
- Original simulation developers (Van Appeven et al., 2018)

---

## ⚠️ Current Limitations

**Be aware when using legacy code:**
1. **Not reproducible** - no seed control
2. **Not tested** - no unit tests
3. **Not fast** - Python loops, no JIT
4. **Not documented** - beyond this README
5. **Not pip-installable** - manual setup required
6. **Hardcoded paths** - must run from specific directories

**These will all be fixed in the refactored version.**

---

*Last updated: 2026-01-05*
*Status: Phase 0 in progress (audit complete)*