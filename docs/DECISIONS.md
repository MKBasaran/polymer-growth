# Architecture & Design Decisions

**Purpose:** Record key technical decisions for thesis defense and future maintenance.

**Format:** ADR (Architecture Decision Record) style

---

## ADR-001: GUI Framework Choice (2026-01-05)

**Status:** DECIDED

**Context:**
- Legacy GUI uses Tkinter (standard library)
- Previous student failed to deliver responsive GUI
- Must be usable by non-programmers (chemists)
- Must not freeze during long computations
- Time constraint: 14 days total

**Options Considered:**
1. **Keep Tkinter + add threading**
   - Pros: No new dependencies, familiar
   - Cons: Threading with Tkinter is tricky, matplotlib integration fragile

2. **Rewrite in PyQt5/PySide**
   - Pros: More robust, better threading, professional look
   - Cons: Large rewrite, new dependency, learning curve

3. **Rewrite in web framework (Flask + React)**
   - Pros: Modern, responsive, could add remote access
   - Cons: Massive scope increase, overkill for desktop use

**Decision:** **Keep Tkinter + add proper threading/multiprocessing**

**Rationale:**
- Minimize rewrite scope (14-day constraint)
- Tkinter is "good enough" if threading is done correctly
- Use `concurrent.futures` or `multiprocessing` for computation
- Communicate via queue to update GUI safely
- Proven pattern in scientific Python GUIs

**Consequences:**
- Must carefully manage thread safety
- Matplotlib updates must happen on main thread
- Progress updates via queue mechanism
- Testing must include manual GUI responsiveness checks

**Implementation Plan:**
1. Extract computation to separate function
2. Run in background thread/process
3. Update GUI via `after()` callbacks checking queue
4. Add progress bar and cancel button
5. Disable controls during run, re-enable on completion

**Success Criteria:**
- GUI remains responsive during 60-second FDDC run
- User can interact with other windows
- Progress updates every 1-2 seconds
- Clean error handling if computation fails

---

## ADR-002: Package Structure (2026-01-05)

**Status:** DECIDED

**Context:**
- Legacy code is flat directory with 25 files
- Need pip-installable package
- Need separation: core logic vs CLI vs GUI
- Must support both programmatic API and user interfaces

**Options Considered:**
1. **Flat package** (`polymer_growth/*.py`)
   - Pros: Simple
   - Cons: No organization, namespace pollution

2. **Nested subpackages** (`polymer_growth/{core,objective,optimizers,cli,gui}/`)
   - Pros: Clear separation, scalable
   - Cons: More boilerplate

3. **Src layout** (`src/polymer_growth/...`)
   - Pros: Best practice, prevents accidental imports
   - Cons: Slightly more complex setup

**Decision:** **Src layout with nested subpackages**

**Structure:**
```
polymer_growth/
├── pyproject.toml
├── src/
│   └── polymer_growth/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── simulation.py
│       │   ├── parameters.py
│       │   └── rng.py
│       ├── objective/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── min_max_v2.py
│       │   └── loaders.py
│       ├── optimizers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── fddc.py
│       ├── io/
│       │   └── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py
│       └── gui/
│           ├── __init__.py
│           └── app.py
├── tests/
└── examples/
```

**Rationale:**
- Src layout prevents import confusion during development
- Subpackages allow independent testing
- CLI and GUI are siblings, both use core/objective/optimizers
- Clear public API via `__init__.py` exports

**Consequences:**
- Must use `pip install -e .` for development
- Imports: `from polymer_growth.core import simulate`
- Tests import from installed package, not local files

---

## ADR-003: RNG/Seed Management (2026-01-05)

**Status:** DECIDED

**Context:**
- Current code uses global `np.random` and `random` modules
- No seed control → non-reproducible
- Multiprocessing complicates seeding (child processes inherit RNG state)
- Must support multiple runs with different seeds

**Options Considered:**
1. **Global seed setting** (`np.random.seed()` at start)
   - Pros: Simple
   - Cons: Doesn't work with multiprocessing, fragile

2. **Pass `numpy.random.Generator` everywhere**
   - Pros: Proper isolation, modern numpy best practice
   - Cons: Requires changing all function signatures

3. **Seed per worker process**
   - Pros: Works with multiprocessing
   - Cons: Tricky to get right

**Decision:** **Hybrid approach**
- Core simulation takes `rng: np.random.Generator` parameter
- Top-level functions take `seed: int` and create Generator
- For multiprocessing: use `SeedSequence` to spawn independent generators

**Implementation:**
```python
# Core simulation
def simulate(params: SimulationParams, rng: np.random.Generator) -> Distribution:
    ...

# User-facing wrapper
def simulate_with_seed(params: SimulationParams, seed: int) -> Distribution:
    rng = np.random.default_rng(seed)
    return simulate(params, rng)

# For parallel evaluation
def evaluate_population_parallel(population, seed):
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(len(population))
    # Each worker gets independent generator
    ...
```

**Consequences:**
- Breaking change: all simulation calls need `rng` parameter
- Must document seed handling in REPRODUCIBILITY_PROTOCOL.md
- Tests must verify determinism

**Success Criteria:**
- Same seed → bit-exact same output (verified with hash)
- Parallel execution with seed → reproducible
- Different seeds → different outputs

---

## ADR-004: CLI vs GUI Code Sharing (2026-01-05)

**Status:** DECIDED

**Context:**
- Both CLI and GUI need to run simulation + optimization
- Must avoid code duplication
- GUI needs progress updates, CLI can use simple print
- Both should produce identical results

**Decision:** **Backend-first architecture**

**Implementation:**
```python
# Core computation (no UI)
class FDDCOptimizer:
    def optimize(self, callback=None):
        for generation in range(max_gen):
            # ... computation ...
            if callback:
                callback(generation, self.best_cost)
        return result

# CLI usage
def cli_fit():
    optimizer = FDDCOptimizer(...)
    result = optimizer.optimize(callback=lambda gen, cost: print(f"Gen {gen}: {cost}"))

# GUI usage
def gui_fit():
    optimizer = FDDCOptimizer(...)
    queue = Queue()

    def callback(gen, cost):
        queue.put({'gen': gen, 'cost': cost})

    # Run in thread
    thread = Thread(target=lambda: optimizer.optimize(callback=callback))
    thread.start()

    # Update GUI from main thread
    def check_queue():
        while not queue.empty():
            update = queue.get()
            update_progress_bar(update)
        if thread.is_alive():
            root.after(100, check_queue)
```

**Consequences:**
- Optimizer must be callback-aware
- CLI and GUI share 100% of computation code
- Testing computation doesn't require GUI
- Progress updates are optional (callback can be None)

---

## ADR-005: Testing Strategy (2026-01-05)

**Status:** DECIDED

**Context:**
- No tests currently exist
- 14-day timeline
- Must verify correctness and reproducibility
- GUI testing is time-consuming

**Decision:** **Tiered testing pyramid**

**Tiers:**
1. **Unit tests** (fast, many)
   - Simulation determinism
   - Parameter validation
   - Distribution operations
   - ~50 tests, <5s total runtime

2. **Integration tests** (medium, fewer)
   - Full FDDC run on toy problem (5 generations)
   - CLI commands end-to-end
   - ~10 tests, ~30s total runtime

3. **Manual GUI tests** (slow, checklist)
   - Responsiveness during computation
   - Error handling
   - Visual correctness
   - ~10 test cases, manual execution

**Coverage Goals:**
- Core simulation: 90%+
- Optimizers: 80%+
- CLI: 70%+
- GUI: Manual testing only (time constraint)

**Tools:**
- pytest
- pytest-cov
- Manual test checklist in RELEASE_CHECKLIST.md

**Non-goals for v1.0:**
- Property-based testing (nice to have)
- Performance regression tests (manual benchmarking sufficient)
- GUI automation (Selenium/pyautogui overkill)

---

## ADR-006: Performance Optimization Strategy (2026-01-05)

**Status:** DECIDED

**Context:**
- Simulation is likely bottleneck (Python loops)
- Must show measurable improvement for thesis
- 14-day timeline limits deep optimization
- Need before/after metrics

**Decision:** **Quick wins first, then profile**

**Phase 1 (Days 10-11): Quick Wins**
1. **Numba JIT on simulation loop**
   - Expected: 2-5x speedup
   - Effort: 4 hours
   - Risk: Low (well-documented pattern)

2. **Vectorize where possible**
   - Growth/death decisions are vectorizable
   - Expected: 1.5x additional
   - Effort: 2 hours

3. **Multiprocessing for population evaluation**
   - Already partially implemented
   - Fix: proper seeding, clean pool management
   - Expected: Nx speedup (N=cores)
   - Effort: 4 hours

**Phase 2 (Day 12): Profile and Optimize**
- Use cProfile to find remaining bottlenecks
- Optimize top 3 hotspots
- Expected: 1.5-2x additional
- Effort: 6 hours

**Measurement:**
- Baseline: Fixed simulation (1000 steps, 100k molecules)
- Baseline: FDDC 20 generations, population 50
- Report: Absolute runtime + speedup factor
- Include in BASELINE_BENCHMARKS.md

**Success Criteria:**
- Simulation: ≥5x faster
- FDDC total: ≥3x faster (includes overhead)
- Thesis can claim "order of magnitude improvement"

---

## ADR-007: Documentation Scope (2026-01-05)

**Status:** DECIDED

**Context:**
- Must accompany scientific paper
- Must be thesis-worthy
- 14-day timeline
- Audience: chemists + developers

**Decision:** **Minimum viable docs, high quality**

**Must Have:**
1. **README.md** - Quickstart, installation, examples
2. **THEORY.md** - Model equations, parameter meanings (from thesis.txt)
3. **API_REFERENCE.md** - Core functions + examples
4. **REPRODUCIBILITY.md** - How to reproduce results
5. **Docstrings** - All public functions (Google style)

**Nice to Have (if time):**
6. **TUTORIAL.md** - Step-by-step fitting workflow
7. **PERFORMANCE.md** - Optimization guide

**Deferred to v1.1:**
- Full Sphinx/MkDocs site
- Developer guide
- Architecture diagrams

**Format:**
- Markdown (simple, versionable)
- Code examples tested (copy-paste from working scripts)
- Equations in LaTeX (for paper compatibility)

---

## ADR-008: Dependency Management (2026-01-05)

**Status:** DECIDED

**Context:**
- Legacy code has loose dependencies
- Must be reproducible
- Pip-installable package

**Decision:** **Pin versions in pyproject.toml**

**Core Dependencies:**
```toml
[project]
dependencies = [
    "numpy>=1.21.0,<2.0",
    "pandas>=1.3.0,<2.0",
    "matplotlib>=3.4.0,<4.0",
    "scipy>=1.7.0,<2.0",
    "numba>=0.54.0,<1.0",  # for JIT
    "click>=8.0.0,<9.0",    # for CLI
]
```

**Development Dependencies:**
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=3.0",
    "ruff>=0.1.0",
    "mypy>=0.990",
]
```

**Rationale:**
- Pin major versions only (allow minor updates)
- Numba required for performance
- Click for CLI (better than argparse)
- No GUI framework beyond Tkinter (stdlib)

---

## ADR-009: Git Workflow (2026-01-05)

**Status:** DECIDED

**Context:**
- Solo developer (thesis project)
- Need clear history for thesis
- Need ability to show before/after

**Decision:** **Feature branches + merge commits**

**Workflow:**
```bash
# Day 1-2: Baseline
git checkout -b phase0-baseline
# ... commits ...
git checkout main
git merge --no-ff phase0-baseline -m "Phase 0: Baseline capture"

# Day 3-4: Core refactor
git checkout -b phase1-core-refactor
# ... commits ...
git merge --no-ff phase1-core-refactor -m "Phase 1: Core simulation refactor"

# etc.
```

**Commit Message Format:**
```
[PHASE] Short description

- Bullet points of changes
- Reference to ENGINEERING_LOG.md entry

Measured impact: [metric]
```

**Branch Naming:**
- `phase0-*` - Baseline and infrastructure
- `phase1-*` - Core refactor
- `phase2-*` - GUI refactor
- `bugfix-*` - Bug fixes
- `docs-*` - Documentation

**Rationale:**
- Clear history for thesis ("show your work")
- Each phase is a distinct milestone
- Can cherry-pick or revert cleanly
- Merge commits mark milestones

---

*This document will be referenced in thesis and code review*