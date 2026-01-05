"""Command-line interface main module."""

import click
import numpy as np
from pathlib import Path

from polymer_growth.core import simulate, SimulationParams, Distribution
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig


@click.group()
@click.version_option()
def cli():
    """
    Polymer Growth Simulation and Optimization.

    A tool for simulating agent-based polymer chain growth and fitting
    parameters to experimental data using genetic algorithms.
    """
    pass


@cli.command()
@click.option('--time', '-t', default=1000, help='Simulation timesteps')
@click.option('--molecules', '-n', default=10000, help='Initial number of molecules')
@click.option('--monomer-pool', '-m', default=1000000, help='Initial monomer pool (-1 for infinite)')
@click.option('--seed', '-s', default=42, help='Random seed for reproducibility')
@click.option('--output', '-o', type=click.Path(), help='Save results to file')
@click.option('--stats/--no-stats', default=True, help='Print distribution statistics')
def simulate_cmd(time, molecules, monomer_pool, seed, output, stats):
    """
    Run a single polymer growth simulation.

    Example:
        polymer-sim simulate --time 1000 --molecules 10000 --seed 42
    """
    click.echo(f"Running simulation (seed={seed})...")

    # Create parameters (using defaults from thesis for other params)
    params = SimulationParams(
        time_sim=time,
        number_of_molecules=molecules,
        monomer_pool=monomer_pool,
        p_growth=0.72,
        p_death=0.000084,
        p_dead_react=0.73,
        l_exponent=0.41,
        d_exponent=0.75,
        l_naked=0.24,
        kill_spawns_new=True
    )

    # Run simulation
    rng = np.random.default_rng(seed)
    dist = simulate(params, rng)

    # Print stats
    if stats:
        click.echo("\nDistribution Statistics:")
        for key, value in dist.stats().items():
            click.echo(f"  {key}: {value}")

    # Save if requested
    if output:
        output_path = Path(output)
        np.savez(
            output_path,
            living=dist.living,
            dead=dist.dead,
            coupled=dist.coupled,
            params=params.to_dict()
        )
        click.echo(f"\nResults saved to {output_path}")

    click.echo("\n✓ Simulation complete")


@cli.command()
@click.argument('experimental_data', type=click.Path(exists=True))
@click.option('--generations', '-g', default=20, help='Number of generations')
@click.option('--population', '-p', default=50, help='Population size')
@click.option('--seed', '-s', default=42, help='Random seed')
@click.option('--output', '-o', type=click.Path(), help='Save best parameters to file')
def fit(experimental_data, generations, population, seed, output):
    """
    Fit simulation parameters to experimental data using FDDC.

    EXPERIMENTAL_DATA: Path to Excel file with experimental polymer distribution

    Example:
        polymer-sim fit data/experimental.xlsx --generations 20 --seed 42
    """
    click.echo(f"Loading experimental data from {experimental_data}...")

    # Load data
    exp_lengths, exp_values = load_experimental_data(experimental_data)
    click.echo(f"  Loaded {len(exp_values)} datapoints")
    click.echo(f"  Max chain length: {exp_lengths.max()}")

    # Setup objective function
    objective = MinMaxV2ObjectiveFunction(exp_values)

    # Define parameter bounds (from thesis)
    bounds = np.array([
        [100, 10000],          # time_sim
        [1000, 100000],        # number_of_molecules
        [10000, 100000000],    # monomer_pool
        [0.1, 0.99],          # p_growth
        [0.00001, 0.01],      # p_death
        [0.1, 0.99],          # p_dead_react
        [0.1, 0.99],          # l_exponent
        [0.1, 0.99],          # d_exponent
        [0.1, 0.99],          # l_naked
        [0, 1]                # kill_spawns_new (0 or 1)
    ])

    # Create wrapper for objective function that runs simulation
    def objective_wrapper(params_array):
        """Convert parameter array to SimulationParams and evaluate."""
        # Convert numpy array elements to Python scalars using .item()
        # This is more robust than float() for numpy arrays
        params = SimulationParams(
            time_sim=int(np.asarray(params_array[0]).item()),
            number_of_molecules=int(np.asarray(params_array[1]).item()),
            monomer_pool=int(np.asarray(params_array[2]).item()),
            p_growth=float(np.asarray(params_array[3]).item()),
            p_death=float(np.asarray(params_array[4]).item()),
            p_dead_react=float(np.asarray(params_array[5]).item()),
            l_exponent=float(np.asarray(params_array[6]).item()),
            d_exponent=float(np.asarray(params_array[7]).item()),
            l_naked=float(np.asarray(params_array[8]).item()),
            kill_spawns_new=bool(round(float(np.asarray(params_array[9]).item())))
        )

        # Run simulation
        rng = np.random.default_rng()  # Will be seeded properly in production
        dist = simulate(params, rng)

        # Compute cost
        return objective.compute_cost(dist)

    # Setup optimizer
    config = FDDCConfig(
        population_size=population,
        max_generations=generations
    )

    def progress_callback(gen, cost):
        """Print progress updates."""
        click.echo(f"Generation {gen}/{generations}: Best cost = {cost:.6f}")

    optimizer = FDDCOptimizer(
        bounds=bounds,
        objective_function=objective_wrapper,
        config=config,
        callback=progress_callback
    )

    # Run optimization
    click.echo(f"\nStarting FDDC optimization (seed={seed})...")
    click.echo(f"  Population: {population}")
    click.echo(f"  Generations: {generations}")
    click.echo()

    result = optimizer.optimize(seed=seed)

    # Print results
    click.echo("\n" + "=" * 60)
    click.echo("OPTIMIZATION COMPLETE")
    click.echo("=" * 60)
    click.echo(f"Best cost: {result.best_cost:.6f}")
    click.echo("\nBest parameters:")
    param_names = [
        'time_sim', 'number_of_molecules', 'monomer_pool',
        'p_growth', 'p_death', 'p_dead_react',
        'l_exponent', 'd_exponent', 'l_naked', 'kill_spawns_new'
    ]
    for name, value in zip(param_names, result.best_params):
        if name in ['time_sim', 'number_of_molecules', 'monomer_pool']:
            click.echo(f"  {name}: {int(value)}")
        elif name == 'kill_spawns_new':
            click.echo(f"  {name}: {bool(round(value))}")
        else:
            click.echo(f"  {name}: {value:.6f}")

    # Save if requested
    if output:
        output_path = Path(output)
        np.savez(
            output_path,
            best_params=result.best_params,
            best_cost=result.best_cost,
            cost_history=result.cost_history,
            generation=result.generation
        )
        click.echo(f"\nResults saved to {output_path}")

    click.echo("\n✓ Optimization complete")


@cli.command()
def gui():
    """Launch the graphical user interface."""
    click.echo("Launching GUI...")
    # Import here to avoid loading GUI dependencies when using CLI
    try:
        from polymer_growth.gui.app import main as gui_main
        gui_main()
    except ImportError:
        click.echo("Error: GUI dependencies not installed. Install with: pip install polymer-growth[gui]")
    except Exception as e:
        click.echo(f"Error launching GUI: {e}")


if __name__ == '__main__':
    cli()