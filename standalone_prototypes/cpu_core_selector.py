"""
CPU CORE SELECTOR: Parallel Evaluation Configuration
=====================================================

This standalone module provides logic for selecting how many CPU cores
to dedicate to the optimization process.

Features:
- Auto-detect available cores
- Let user choose number of workers
- Safety limits (leave cores for system)
- Integration-ready for GUI

This is a STANDALONE prototype - does NOT modify production code.
"""

import os
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


@dataclass
class CPUConfig:
    """Configuration for CPU core usage."""
    n_workers: Optional[int] = None  # None = auto-detect
    leave_free: int = 1  # Leave this many cores free for system
    use_processes: bool = False  # Use ProcessPoolExecutor vs ThreadPoolExecutor
    max_workers_override: Optional[int] = None  # Hard limit


def get_cpu_info() -> dict:
    """
    Get detailed CPU information for display in GUI.

    Returns dict with:
    - total_cores: Physical + logical cores available
    - physical_cores: Physical cores only (if detectable)
    - recommended: Recommended worker count
    - min_workers: Minimum sensible workers
    - max_workers: Maximum safe workers
    """
    try:
        total_cores = os.cpu_count() or 1
    except:
        total_cores = 1

    try:
        # Try to get physical cores (not always available)
        physical_cores = multiprocessing.cpu_count()
    except:
        physical_cores = total_cores

    # Recommendations
    # - Leave at least 1 core free for system/GUI
    # - For I/O bound tasks (like our simulation), can use more threads than cores
    # - For CPU bound, use physical cores

    recommended = max(1, total_cores - 1)
    min_workers = 1
    max_workers = total_cores

    return {
        'total_cores': total_cores,
        'physical_cores': physical_cores,
        'recommended': recommended,
        'min_workers': min_workers,
        'max_workers': max_workers,
    }


def validate_worker_count(n_workers: Optional[int], config: CPUConfig = None) -> int:
    """
    Validate and normalize worker count.

    Args:
        n_workers: Requested worker count (None = auto)
        config: Optional CPUConfig for constraints

    Returns:
        Valid worker count
    """
    if config is None:
        config = CPUConfig()

    cpu_info = get_cpu_info()

    if n_workers is None:
        # Auto: use recommended
        n_workers = cpu_info['recommended']
    elif n_workers <= 0:
        # Invalid: use recommended
        n_workers = cpu_info['recommended']

    # Apply constraints
    n_workers = max(cpu_info['min_workers'], n_workers)
    n_workers = min(cpu_info['max_workers'], n_workers)

    # Leave some free if requested
    if config.leave_free > 0:
        max_with_free = cpu_info['total_cores'] - config.leave_free
        n_workers = min(n_workers, max(1, max_with_free))

    # Apply hard override if set
    if config.max_workers_override is not None:
        n_workers = min(n_workers, config.max_workers_override)

    return n_workers


class ParallelEvaluator:
    """
    Parallel evaluation helper for FDDC optimizer.

    Handles parallel execution of simulation evaluations across
    multiple CPU cores.
    """

    def __init__(self, n_workers: Optional[int] = None,
                 use_processes: bool = False):
        """
        Initialize parallel evaluator.

        Args:
            n_workers: Number of worker threads/processes (None = auto)
            use_processes: Use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)
                          ThreadPool is usually better for our use case since
                          the GIL is released during numpy operations.
        """
        self.n_workers = validate_worker_count(n_workers)
        self.use_processes = use_processes
        self._executor = None

    def __enter__(self):
        """Context manager entry."""
        if self.use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def evaluate_batch(self, objective_fn, param_list: list,
                       sigma_list: list = None) -> list:
        """
        Evaluate a batch of parameter sets in parallel.

        Args:
            objective_fn: Function that takes (params, sigma=None) and returns cost
            param_list: List of parameter sets to evaluate
            sigma_list: Optional list of sigma weights (same length as param_list)

        Returns:
            List of costs in same order as param_list
        """
        if sigma_list is None:
            sigma_list = [None] * len(param_list)

        assert len(param_list) == len(sigma_list)

        if self._executor is None:
            raise RuntimeError("ParallelEvaluator must be used as context manager")

        # Submit all tasks
        futures = {}
        for i, (params, sigma) in enumerate(zip(param_list, sigma_list)):
            future = self._executor.submit(objective_fn, params, sigma=sigma)
            futures[future] = i

        # Collect results in order
        results = [None] * len(param_list)
        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Warning: Evaluation {idx} failed: {e}")
                results[idx] = float('inf')  # Penalty for failed evaluation

        return results


# =============================================================================
# GUI INTEGRATION HELPERS
# =============================================================================

def get_worker_options() -> list:
    """
    Get list of worker count options for GUI dropdown/slider.

    Returns list of tuples: (display_text, value)
    """
    cpu_info = get_cpu_info()
    options = []

    # Auto option
    options.append((f"Auto ({cpu_info['recommended']} workers)", None))

    # Specific counts
    for n in range(1, cpu_info['max_workers'] + 1):
        if n == 1:
            label = "1 worker (sequential)"
        elif n == cpu_info['recommended']:
            label = f"{n} workers (recommended)"
        elif n == cpu_info['max_workers']:
            label = f"{n} workers (maximum)"
        else:
            label = f"{n} workers"
        options.append((label, n))

    return options


def format_worker_status(n_workers: int, cpu_info: dict = None) -> str:
    """
    Format worker status for display.

    Returns string like "Using 6/8 cores (75%)"
    """
    if cpu_info is None:
        cpu_info = get_cpu_info()

    total = cpu_info['total_cores']
    percentage = (n_workers / total) * 100

    return f"Using {n_workers}/{total} cores ({percentage:.0f}%)"


# =============================================================================
# DEMO
# =============================================================================

def demo_parallel_evaluation():
    """Demonstrate parallel evaluation."""
    import random
    import numpy as np

    print("=" * 60)
    print("CPU CORE SELECTOR DEMO")
    print("=" * 60)

    # Get CPU info
    cpu_info = get_cpu_info()
    print(f"\nSystem Information:")
    print(f"  Total cores: {cpu_info['total_cores']}")
    print(f"  Physical cores: {cpu_info['physical_cores']}")
    print(f"  Recommended workers: {cpu_info['recommended']}")

    # Show options for GUI
    print(f"\nGUI Worker Options:")
    for label, value in get_worker_options():
        print(f"  {label} -> {value}")

    # Dummy objective (simulates polymer simulation)
    def dummy_objective(params, sigma=None):
        time.sleep(0.1)  # Simulate computation time
        return sum(p ** 2 for p in params) + random.gauss(0, 1)

    # Generate test data
    n_evaluations = 20
    param_list = [[random.random() for _ in range(10)] for _ in range(n_evaluations)]

    # Compare sequential vs parallel
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON ({n_evaluations} evaluations)")
    print(f"{'='*60}")

    # Sequential
    print("\n[1] Sequential (1 worker):")
    start = time.time()
    sequential_results = [dummy_objective(p) for p in param_list]
    seq_time = time.time() - start
    print(f"  Time: {seq_time:.2f}s")

    # Parallel with different worker counts
    for n_workers in [2, 4, cpu_info['recommended']]:
        if n_workers > cpu_info['max_workers']:
            continue

        print(f"\n[{n_workers}] Parallel ({n_workers} workers):")
        start = time.time()

        with ParallelEvaluator(n_workers=n_workers) as evaluator:
            parallel_results = evaluator.evaluate_batch(dummy_objective, param_list)

        par_time = time.time() - start
        speedup = seq_time / par_time
        print(f"  Time: {par_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  {format_worker_status(n_workers, cpu_info)}")

    print(f"\n{'='*60}")
    print("INTEGRATION NOTES")
    print("=" * 60)
    print("""
To integrate into GUI (OptimizationTab):

1. Add worker selection widget:
   ```python
   self.worker_combo = QComboBox()
   for label, value in get_worker_options():
       self.worker_combo.addItem(label, value)
   ```

2. Pass to FDDCConfig:
   ```python
   n_workers = self.worker_combo.currentData()
   config = FDDCConfig(..., n_workers=n_workers)
   ```

3. Show status during optimization:
   ```python
   status = format_worker_status(config.n_workers)
   self.status_label.setText(status)
   ```
""")


if __name__ == "__main__":
    demo_parallel_evaluation()
