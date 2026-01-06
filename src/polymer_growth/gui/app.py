"""Main GUI application for polymer growth simulation and optimization."""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QMessageBox
)
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QFont

from polymer_growth.core import simulate, SimulationParams, Distribution
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from polymer_growth.gui.plotting import PlotWidget


class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking the UI."""

    finished = Signal(object)  # Emits Distribution
    error = Signal(str)

    def __init__(self, params: SimulationParams, seed: int):
        super().__init__()
        self.params = params
        self.seed = seed

    def run(self):
        """Run simulation in background thread."""
        try:
            rng = np.random.default_rng(self.seed)
            dist = simulate(self.params, rng)
            self.finished.emit(dist)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationWorker(QThread):
    """Worker thread for running FDDC optimization without blocking the UI."""

    progress = Signal(int, float)  # generation, cost
    console_message = Signal(str)  # console output
    finished = Signal(object)  # OptimizationResult
    error = Signal(str)

    def __init__(
        self,
        experimental_data_path: str,
        config: FDDCConfig,
        bounds: np.ndarray,
        seed: int
    ):
        super().__init__()
        self.experimental_data_path = experimental_data_path
        self.config = config
        self.bounds = bounds
        self.seed = seed
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of optimization."""
        self._is_cancelled = True

    def run(self):
        """Run optimization in background thread."""
        try:
            # Load experimental data
            exp_lengths, exp_values = load_experimental_data(self.experimental_data_path)
            objective = MinMaxV2ObjectiveFunction(exp_values)

            # Create objective wrapper
            def objective_wrapper(params_array, sigma=None):
                if self._is_cancelled:
                    raise InterruptedError("Optimization cancelled")

                # Convert entire array to Python list to ensure native types
                # This is the most robust approach for co-evolution mode
                params_list = [float(x) for x in params_array]

                params = SimulationParams(
                    time_sim=int(params_list[0]),
                    number_of_molecules=int(params_list[1]),
                    monomer_pool=int(params_list[2]),
                    p_growth=params_list[3],
                    p_death=params_list[4],
                    p_dead_react=params_list[5],
                    l_exponent=params_list[6],
                    d_exponent=params_list[7],
                    l_naked=params_list[8],
                    kill_spawns_new=bool(round(params_list[9]))
                )

                rng = np.random.default_rng()
                dist = simulate(params, rng)
                return objective.compute_cost(dist, sigma=sigma)

            # Progress callback
            def progress_callback(gen, cost):
                if not self._is_cancelled:
                    self.progress.emit(gen, cost)

            # Console callback
            def console_callback(message):
                if not self._is_cancelled:
                    self.console_message.emit(message)

            # Run optimization
            optimizer = FDDCOptimizer(
                bounds=self.bounds,
                objective_function=objective_wrapper,
                config=self.config,
                callback=progress_callback,
                console_callback=console_callback
            )

            result = optimizer.optimize(seed=self.seed)

            if not self._is_cancelled:
                self.finished.emit(result)

        except InterruptedError:
            self.error.emit("Optimization cancelled by user")
        except Exception as e:
            self.error.emit(str(e))


class SimulationTab(QWidget):
    """Tab for running single simulations."""

    def __init__(self):
        super().__init__()
        self.worker: Optional[SimulationWorker] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the simulation tab UI."""
        layout = QHBoxLayout()  # Changed to horizontal for side-by-side

        # Left side: Parameters
        left_layout = QVBoxLayout()

        # Parameter input group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()

        # Create input widgets
        self.time_input = QSpinBox()
        self.time_input.setRange(100, 10000)
        self.time_input.setValue(1000)

        self.molecules_input = QSpinBox()
        self.molecules_input.setRange(1000, 100000)
        self.molecules_input.setValue(10000)

        self.monomer_input = QSpinBox()
        self.monomer_input.setRange(-1, 100000000)
        self.monomer_input.setValue(1000000)

        self.p_growth_input = QDoubleSpinBox()
        self.p_growth_input.setRange(0.0, 0.99)
        self.p_growth_input.setSingleStep(0.01)
        self.p_growth_input.setValue(0.72)

        self.p_death_input = QDoubleSpinBox()
        self.p_death_input.setRange(0.00001, 0.01)
        self.p_death_input.setSingleStep(0.00001)
        self.p_death_input.setDecimals(5)
        self.p_death_input.setValue(0.000084)

        self.p_dead_react_input = QDoubleSpinBox()
        self.p_dead_react_input.setRange(0.1, 0.99)
        self.p_dead_react_input.setSingleStep(0.01)
        self.p_dead_react_input.setValue(0.73)

        self.l_exponent_input = QDoubleSpinBox()
        self.l_exponent_input.setRange(0.1, 0.99)
        self.l_exponent_input.setSingleStep(0.01)
        self.l_exponent_input.setValue(0.41)

        self.d_exponent_input = QDoubleSpinBox()
        self.d_exponent_input.setRange(0.1, 0.99)
        self.d_exponent_input.setSingleStep(0.01)
        self.d_exponent_input.setValue(0.75)

        self.l_naked_input = QDoubleSpinBox()
        self.l_naked_input.setRange(0.1, 0.99)
        self.l_naked_input.setSingleStep(0.01)
        self.l_naked_input.setValue(0.24)

        self.kill_spawns_input = QCheckBox()
        self.kill_spawns_input.setChecked(True)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(42)

        # Add to form
        params_layout.addRow("Simulation Time:", self.time_input)
        params_layout.addRow("Number of Molecules:", self.molecules_input)
        params_layout.addRow("Monomer Pool:", self.monomer_input)
        params_layout.addRow("P(Growth):", self.p_growth_input)
        params_layout.addRow("P(Death):", self.p_death_input)
        params_layout.addRow("P(Dead React):", self.p_dead_react_input)
        params_layout.addRow("Living Exponent:", self.l_exponent_input)
        params_layout.addRow("Death Exponent:", self.d_exponent_input)
        params_layout.addRow("Living Naked:", self.l_naked_input)
        params_layout.addRow("Kill Spawns New:", self.kill_spawns_input)
        params_layout.addRow("Random Seed:", self.seed_input)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        left_layout.addStretch()

        # Right side: Plot
        right_layout = QVBoxLayout()
        plot_group = QGroupBox("Distribution Plot")
        plot_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        # Combine left and right
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)
        self.setLayout(layout)

    @Slot()
    def run_simulation(self):
        """Run simulation in background thread."""
        # Get parameters
        params = SimulationParams(
            time_sim=self.time_input.value(),
            number_of_molecules=self.molecules_input.value(),
            monomer_pool=self.monomer_input.value(),
            p_growth=self.p_growth_input.value(),
            p_death=self.p_death_input.value(),
            p_dead_react=self.p_dead_react_input.value(),
            l_exponent=self.l_exponent_input.value(),
            d_exponent=self.d_exponent_input.value(),
            l_naked=self.l_naked_input.value(),
            kill_spawns_new=self.kill_spawns_input.isChecked()
        )

        seed = self.seed_input.value()

        # Disable button during simulation
        self.run_btn.setEnabled(False)
        self.results_text.setText("Running simulation...")

        # Start worker thread
        self.worker = SimulationWorker(params, seed)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    @Slot(object)
    def on_simulation_finished(self, dist: Distribution):
        """Handle simulation completion."""
        self.run_btn.setEnabled(True)

        # Display results
        stats = dist.stats()
        results = "Simulation Complete!\n\n"
        results += f"Living chains: {len(dist.living)}\n"
        results += f"Dead chains: {len(dist.dead)}\n"
        results += f"Coupled chains: {len(dist.coupled)}\n"
        results += f"\nStatistics:\n"
        for key, value in stats.items():
            results += f"  {key}: {value}\n"

        self.results_text.setText(results)

        # Plot distribution
        self.plot_widget.canvas.plot_distribution(dist)

    @Slot(str)
    def on_simulation_error(self, error_msg: str):
        """Handle simulation error."""
        self.run_btn.setEnabled(True)
        self.results_text.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Simulation Error", error_msg)


class OptimizationTab(QWidget):
    """Tab for running FDDC optimization."""

    def __init__(self):
        super().__init__()
        self.worker: Optional[OptimizationWorker] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the optimization tab UI."""
        main_layout = QHBoxLayout()  # Changed to horizontal for side-by-side
        layout = QVBoxLayout()  # Left side layout

        # Data file selection
        data_group = QGroupBox("Experimental Data")
        data_layout = QHBoxLayout()
        self.data_path_input = QLineEdit()
        self.data_path_input.setPlaceholderText("Select Excel file with experimental data...")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_data_file)
        data_layout.addWidget(self.data_path_input)
        data_layout.addWidget(self.browse_btn)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # FDDC configuration
        config_group = QGroupBox("FDDC Configuration")
        config_layout = QFormLayout()

        self.population_input = QSpinBox()
        self.population_input.setRange(10, 200)
        self.population_input.setValue(50)

        self.generations_input = QSpinBox()
        self.generations_input.setRange(5, 100)
        self.generations_input.setValue(20)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(42)

        self.fast_mode_checkbox = QCheckBox("Fast Test Mode (disables co-evolution)")
        self.fast_mode_checkbox.setChecked(False)
        self.fast_mode_checkbox.setToolTip(
            "Disables co-evolution for faster testing (~10x speedup).\n"
            "⚠️ Use only for testing - disable for real optimization!"
        )

        config_layout.addRow("Population Size:", self.population_input)
        config_layout.addRow("Max Generations:", self.generations_input)
        config_layout.addRow("Random Seed:", self.seed_input)
        config_layout.addRow("", self.fast_mode_checkbox)  # Empty label for checkbox

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Progress display
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready to start optimization")
        progress_layout.addWidget(self.progress_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()

        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setMaximumHeight(150)
        self.console_output.setFont(QFont("Courier", 9))  # Monospace font for console
        console_layout.addWidget(self.console_output)

        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Optimization")
        self.start_btn.clicked.connect(self.start_optimization)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_optimization)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Results display
        results_group = QGroupBox("Best Parameters")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

        # Right side: Plot
        right_layout = QVBoxLayout()
        plot_group = QGroupBox("Convergence Plot")
        plot_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        # Combine left and right
        main_layout.addLayout(layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    @Slot()
    def browse_data_file(self):
        """Open file dialog to select experimental data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Experimental Data",
            str(Path.home()),
            "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.data_path_input.setText(file_path)

    @Slot()
    def start_optimization(self):
        """Start FDDC optimization in background thread."""
        # Validate data path
        data_path = self.data_path_input.text()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(
                self,
                "Invalid Data File",
                "Please select a valid experimental data file."
            )
            return

        # Get configuration
        config = FDDCConfig(
            population_size=self.population_input.value(),
            max_generations=self.generations_input.value(),
            enable_coevolution=not self.fast_mode_checkbox.isChecked()  # Disable if fast mode
        )

        # Default parameter bounds (from thesis)
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
            [0, 1]                # kill_spawns_new
        ])

        seed = self.seed_input.value()

        # Disable/enable buttons
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting optimization...")
        self.results_text.clear()
        self.console_output.clear()

        # Start worker thread
        self.worker = OptimizationWorker(data_path, config, bounds, seed)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.console_message.connect(self.on_console_message)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.error.connect(self.on_optimization_error)
        self.worker.start()

    @Slot()
    def cancel_optimization(self):
        """Cancel running optimization."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")

    @Slot(int, float)
    def on_progress_update(self, generation: int, cost: float):
        """Update progress display."""
        max_gen = self.generations_input.value()
        progress = int((generation / max_gen) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(
            f"Generation {generation}/{max_gen} - Best cost: {cost:.6f}"
        )

    @Slot(str)
    def on_console_message(self, message: str):
        """Append message to console output."""
        self.console_output.append(message)
        # Auto-scroll to bottom
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )

    @Slot(object)
    def on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Optimization complete!")

        # Display best parameters
        param_names = [
            'time_sim', 'number_of_molecules', 'monomer_pool',
            'p_growth', 'p_death', 'p_dead_react',
            'l_exponent', 'd_exponent', 'l_naked', 'kill_spawns_new'
        ]

        results = f"Best Cost: {result.best_cost:.6f}\n\n"
        results += "Best Parameters:\n"
        for name, value in zip(param_names, result.best_params):
            if name in ['time_sim', 'number_of_molecules', 'monomer_pool']:
                results += f"  {name}: {int(value)}\n"
            elif name == 'kill_spawns_new':
                results += f"  {name}: {bool(round(value))}\n"
            else:
                results += f"  {name}: {value:.6f}\n"

        results += f"\nConvergence: {len(result.cost_history)} generations\n"

        self.results_text.setText(results)

        # Plot convergence
        if result.cost_history:
            self.plot_widget.canvas.plot_convergence(result.cost_history)

    @Slot(str)
    def on_optimization_error(self, error_msg: str):
        """Handle optimization error."""
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText("Optimization failed")
        self.results_text.setText(f"Error: {error_msg}")

        if "cancelled" not in error_msg.lower():
            QMessageBox.critical(self, "Optimization Error", error_msg)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("Polymer Growth Simulation & Optimization")
        self.setMinimumSize(800, 600)

        # Create tab widget
        tabs = QTabWidget()
        tabs.addTab(SimulationTab(), "Simulation")
        tabs.addTab(OptimizationTab(), "Optimization")

        self.setCentralWidget(tabs)


def main():
    """Launch the GUI application."""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()