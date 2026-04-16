"""
Run Manager: Organize simulation and optimization outputs.

Creates structured output directories with timestamps for each run.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


class RunManager:
    """
    Manages organized output for simulation and optimization runs.

    Creates directory structure:
        runs/
        ├── 2024-02-25_14-30-00_simulation/
        │   ├── params.json
        │   ├── results.json
        │   ├── kinetics.csv
        │   └── distribution.npz
        └── 2024-02-25_14-35-00_optimization/
            ├── config.json
            ├── best_params.json
            ├── cost_history.csv
            └── final_distribution.npz
    """

    def __init__(self, base_dir: str = "runs"):
        """
        Initialize run manager.

        Args:
            base_dir: Base directory for all runs (default: "runs")
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_run_dir = None

    def start_run(self, run_type: str = "simulation", name: Optional[str] = None) -> Path:
        """
        Start a new run and create its directory.

        Args:
            run_type: Type of run ("simulation" or "optimization")
            name: Optional custom name suffix

        Returns:
            Path to the run directory
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if name:
            dir_name = f"{timestamp}_{run_type}_{name}"
        else:
            dir_name = f"{timestamp}_{run_type}"

        self.current_run_dir = self.base_dir / dir_name
        self.current_run_dir.mkdir(exist_ok=True)

        # Create run info file with system metadata
        import platform
        import sys as _sys

        info = {
            "timestamp": timestamp,
            "type": run_type,
            "name": name,
            "created": datetime.now().isoformat(),
            "system": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": _sys.version,
                "cpu_count": os.cpu_count(),
                "machine": platform.machine(),
            },
        }
        self._save_json("run_info.json", info)

        return self.current_run_dir

    def _save_json(self, filename: str, data: Dict[str, Any]):
        """Save data as JSON file."""
        if self.current_run_dir is None:
            raise RuntimeError("No run started. Call start_run() first.")

        path = self.current_run_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def save_simulation_params(self, params) -> Path:
        """Save simulation parameters."""
        data = {
            "time_sim": params.time_sim,
            "number_of_molecules": params.number_of_molecules,
            "monomer_pool": params.monomer_pool,
            "p_growth": params.p_growth,
            "p_death": params.p_death,
            "p_dead_react": params.p_dead_react,
            "l_exponent": params.l_exponent,
            "d_exponent": params.d_exponent,
            "l_naked": params.l_naked,
            "kill_spawns_new": params.kill_spawns_new,
        }
        return self._save_json("params.json", data)

    def save_simulation_results(self, distribution, seed: int = None) -> Path:
        """Save simulation results (stats + distribution)."""
        stats = distribution.stats()
        poly_stats = distribution.polymer_stats()

        results = {
            "seed": seed,
            "basic_stats": stats,
            "polymer_stats": poly_stats,
        }
        self._save_json("results.json", results)

        # Also save raw distribution as npz
        npz_path = self.current_run_dir / "distribution.npz"
        np.savez(
            npz_path,
            living=distribution.living,
            dead=distribution.dead,
            coupled=distribution.coupled
        )

        return self.current_run_dir / "results.json"

    def save_kinetics(self, kinetics) -> Path:
        """Save kinetics data to CSV."""
        if self.current_run_dir is None:
            raise RuntimeError("No run started. Call start_run() first.")

        csv_path = self.current_run_dir / "kinetics.csv"
        kinetics.to_csv(str(csv_path))

        # Also save as Excel for chemists
        xlsx_path = self.current_run_dir / "kinetics.xlsx"
        try:
            kinetics.to_excel(str(xlsx_path))
        except Exception:
            pass  # Excel export is optional

        return csv_path

    def save_optimization_config(self, config) -> Path:
        """Save FDDC optimization config."""
        data = {
            "population_size": config.population_size,
            "max_generations": config.max_generations,
            "memory_size": config.memory_size,
            "n_encounters": config.n_encounters,
            "n_children": config.n_children,
            "mutation_rate": config.mutation_rate,
            "mutation_strength": config.mutation_strength,
            "crossover_type": config.crossover_type,
            "n_workers": config.n_workers,
            "enable_fddc": config.enable_fddc,
            "rank_selection_power": config.rank_selection_power,
        }
        return self._save_json("config.json", data)

    def save_optimization_results(self, result, param_names: list = None) -> Path:
        """Save optimization results."""
        if param_names is None:
            param_names = [
                "time_sim", "number_of_molecules", "monomer_pool",
                "p_growth", "p_death", "p_dead_react",
                "l_exponent", "d_exponent", "l_naked", "kill_spawns_new"
            ]

        # Convert best params to dict
        best_params_dict = {}
        for i, name in enumerate(param_names):
            val = result.best_params[i]
            # Convert numpy types to Python native
            if hasattr(val, 'item'):
                val = val.item()
            best_params_dict[name] = val

        data = {
            "best_cost": float(result.best_cost),
            "generations_run": result.generation,
            "convergence_generation": result.convergence_generation,
            "best_params": best_params_dict,
        }
        self._save_json("optimization_results.json", data)

        # Save cost history as CSV
        import pandas as pd
        cost_df = pd.DataFrame({
            "generation": list(range(1, len(result.cost_history) + 1)),
            "best_cost": result.cost_history
        })
        cost_path = self.current_run_dir / "cost_history.csv"
        cost_df.to_csv(cost_path, index=False)

        return self.current_run_dir / "optimization_results.json"

    def save_experimental_data_info(self, data_path: str, chain_lengths, values) -> Path:
        """Save info about experimental data used."""
        data = {
            "source_file": str(data_path),
            "n_points": len(chain_lengths),
            "chain_length_range": [float(chain_lengths.min()), float(chain_lengths.max())],
            "value_range": [float(values.min()), float(values.max())],
        }
        return self._save_json("experimental_data_info.json", data)

    def get_run_summary(self) -> str:
        """Get a text summary of the current run."""
        if self.current_run_dir is None:
            return "No run in progress."

        summary = f"Run Directory: {self.current_run_dir}\n"
        summary += f"Files:\n"

        for f in sorted(self.current_run_dir.iterdir()):
            size = f.stat().st_size
            summary += f"  - {f.name} ({size:,} bytes)\n"

        return summary

    @classmethod
    def list_runs(cls, base_dir: str = "runs") -> list:
        """List all available runs."""
        base = Path(base_dir)
        if not base.exists():
            return []

        runs = []
        for d in sorted(base.iterdir(), reverse=True):
            if d.is_dir():
                info_file = d / "run_info.json"
                if info_file.exists():
                    with open(info_file) as f:
                        info = json.load(f)
                    runs.append({
                        "path": str(d),
                        "name": d.name,
                        **info
                    })
                else:
                    runs.append({
                        "path": str(d),
                        "name": d.name,
                    })

        return runs


# Convenience functions for quick use
_default_manager = None

def get_run_manager(base_dir: str = "runs") -> RunManager:
    """Get or create the default run manager."""
    global _default_manager
    if _default_manager is None or str(_default_manager.base_dir) != base_dir:
        _default_manager = RunManager(base_dir)
    return _default_manager


def save_simulation_run(params, distribution, kinetics=None, seed=None, name=None):
    """
    Convenience function to save a complete simulation run.

    Args:
        params: SimulationParams
        distribution: Distribution result
        kinetics: Optional KineticsData
        seed: Random seed used
        name: Optional run name

    Returns:
        Path to run directory
    """
    manager = get_run_manager()
    manager.start_run("simulation", name)
    manager.save_simulation_params(params)
    manager.save_simulation_results(distribution, seed)

    if kinetics is not None:
        manager.save_kinetics(kinetics)

    print(f"Run saved to: {manager.current_run_dir}")
    return manager.current_run_dir


def save_optimization_run(config, result, experimental_data_path=None,
                          chain_lengths=None, values=None, name=None):
    """
    Convenience function to save a complete optimization run.

    Args:
        config: FDDCConfig
        result: OptimizationResult
        experimental_data_path: Path to experimental data file
        chain_lengths: Experimental chain lengths array
        values: Experimental values array
        name: Optional run name

    Returns:
        Path to run directory
    """
    manager = get_run_manager()
    manager.start_run("optimization", name)
    manager.save_optimization_config(config)
    manager.save_optimization_results(result)

    if experimental_data_path and chain_lengths is not None and values is not None:
        manager.save_experimental_data_info(experimental_data_path, chain_lengths, values)

    print(f"Run saved to: {manager.current_run_dir}")
    return manager.current_run_dir
