"""Main GUI application for polymer growth simulation and optimization."""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFileDialog, QTextEdit, QProgressBar, QGroupBox, QFormLayout,
    QMessageBox, QComboBox, QToolButton, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QFont, QIcon

from polymer_growth.core import simulate, SimulationParams, Distribution
from polymer_growth.core.run_manager import RunManager
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from polymer_growth.gui.plotting import PlotWidget
from polymer_growth.gui.tooltips import PARAM_INFO, METRIC_INFO
from polymer_growth.gui.queue_tab import TaskQueueTab


def _info_button(tooltip_html: str) -> QToolButton:
    """Create a small info button with a rich-text tooltip."""
    btn = QToolButton()
    btn.setText("?")
    btn.setFixedSize(20, 20)
    btn.setStyleSheet(
        "QToolButton {"
        "  border: 1px solid #999; border-radius: 10px;"
        "  background: #e8f4fd; color: #2980b9;"
        "  font-weight: bold; font-size: 11px;"
        "}"
        "QToolButton:hover { background: #d0e8f7; }"
    )
    btn.setToolTip(tooltip_html)
    return btn


def _labeled_row(label_text: str, widget, info_key: str = None):
    """Create a horizontal layout with label, widget, and optional info button."""
    row = QHBoxLayout()
    label = QLabel(label_text)
    label.setFixedWidth(140)
    row.addWidget(label)
    row.addWidget(widget, 1)
    if info_key and info_key in PARAM_INFO:
        row.addWidget(_info_button(PARAM_INFO[info_key]["tooltip"]))
    elif info_key and info_key in METRIC_INFO:
        row.addWidget(_info_button(METRIC_INFO[info_key]["tooltip"]))
    return row


class SimulationWorker(QThread):
    """Worker thread for running simulations without blocking the UI."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, params: SimulationParams, seed: int, track_kinetics: bool = False):
        super().__init__()
        self.params = params
        self.seed = seed
        self.track_kinetics = track_kinetics

    def run(self):
        try:
            rng = np.random.default_rng(self.seed)
            result = simulate(self.params, rng, track_kinetics=self.track_kinetics)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationWorker(QThread):
    """Worker thread for running FDDC optimization without blocking the UI."""

    progress = Signal(int, float)
    console_message = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, experimental_data_path: str, config: FDDCConfig,
                 bounds: np.ndarray, seed: int):
        super().__init__()
        self.experimental_data_path = experimental_data_path
        self.config = config
        self.bounds = bounds
        self.seed = seed
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            exp_lengths, exp_values = load_experimental_data(self.experimental_data_path)
            objective = MinMaxV2ObjectiveFunction(exp_values)

            def objective_wrapper(params_array, sigma=None):
                if self._is_cancelled:
                    raise InterruptedError("Optimization cancelled")

                params_array = np.asarray(params_array).flatten()
                params_list = params_array.tolist()

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

            def progress_callback(gen, cost):
                if not self._is_cancelled:
                    self.progress.emit(gen, cost)

            def console_callback(message):
                if not self._is_cancelled:
                    self.console_message.emit(message)

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
    """Tab for running single simulations with interactive visualization."""

    def __init__(self):
        super().__init__()
        self.worker: Optional[SimulationWorker] = None
        self.last_result = None
        self.last_kinetics = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        # --- Left side: Parameters ---
        left_layout = QVBoxLayout()

        params_group = QGroupBox("Simulation Parameters")
        params_layout = QVBoxLayout()

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

        self.track_kinetics_input = QCheckBox()
        self.track_kinetics_input.setChecked(True)

        # Add rows with info icons
        params_layout.addLayout(_labeled_row("Simulation Time:", self.time_input, "time_sim"))
        params_layout.addLayout(_labeled_row("Molecules:", self.molecules_input, "number_of_molecules"))
        params_layout.addLayout(_labeled_row("Monomer Pool:", self.monomer_input, "monomer_pool"))
        params_layout.addLayout(_labeled_row("P(Growth):", self.p_growth_input, "p_growth"))
        params_layout.addLayout(_labeled_row("P(Death):", self.p_death_input, "p_death"))
        params_layout.addLayout(_labeled_row("P(Dead React):", self.p_dead_react_input, "p_dead_react"))
        params_layout.addLayout(_labeled_row("Living Exponent:", self.l_exponent_input, "l_exponent"))
        params_layout.addLayout(_labeled_row("Death Exponent:", self.d_exponent_input, "d_exponent"))
        params_layout.addLayout(_labeled_row("L Naked:", self.l_naked_input, "l_naked"))
        params_layout.addLayout(_labeled_row("Kill Spawns New:", self.kill_spawns_input, "kill_spawns_new"))
        params_layout.addLayout(_labeled_row("Random Seed:", self.seed_input, "seed"))
        params_layout.addLayout(_labeled_row("Track Kinetics:", self.track_kinetics_input))

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        self.export_btn = QPushButton("Export Kinetics")
        self.export_btn.clicked.connect(self.export_kinetics)
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Export per-timestep Mn/Mw/PDI data")
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        left_layout.addLayout(btn_layout)

        # Results display
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(220)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        left_layout.addStretch()

        # --- Right side: Plots ---
        right_layout = QVBoxLayout()

        # Distribution plot
        dist_group = QGroupBox("Distribution Plot")
        dist_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        dist_layout.addWidget(self.plot_widget)
        dist_group.setLayout(dist_layout)
        right_layout.addWidget(dist_group, 3)

        # Kinetics plot (shown when kinetics tracking is enabled)
        self.kinetics_group = QGroupBox("Kinetics Over Time")
        kinetics_layout = QVBoxLayout()

        # Plot selector for kinetics
        self.kinetics_selector = QComboBox()
        self.kinetics_selector.addItem("Mn / Mw / PDI", "kinetics")
        self.kinetics_selector.addItem("Chain Populations + Conversion", "populations")
        self.kinetics_selector.currentIndexChanged.connect(self._update_kinetics_plot)
        kinetics_layout.addWidget(self.kinetics_selector)

        self.kinetics_plot = PlotWidget()
        kinetics_layout.addWidget(self.kinetics_plot)
        self.kinetics_group.setLayout(kinetics_layout)
        self.kinetics_group.setVisible(False)
        right_layout.addWidget(self.kinetics_group, 2)

        # Combine
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 2)
        self.setLayout(layout)

    @Slot()
    def run_simulation(self):
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
        self.run_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.results_text.setText("Running simulation...")

        track_kinetics = self.track_kinetics_input.isChecked()
        self.worker = SimulationWorker(params, seed, track_kinetics=track_kinetics)
        self.worker.finished.connect(self.on_simulation_finished)
        self.worker.error.connect(self.on_simulation_error)
        self.worker.start()

    @Slot(object)
    def on_simulation_finished(self, result):
        self.run_btn.setEnabled(True)

        from polymer_growth.core.simulation import SimulationResult
        if isinstance(result, SimulationResult):
            dist = result.distribution
            self.last_kinetics = result.kinetics
            self.export_btn.setEnabled(self.last_kinetics is not None)
        else:
            dist = result
            self.last_kinetics = None
            self.export_btn.setEnabled(False)

        self.last_result = dist

        # Display results
        stats = dist.stats()
        poly = dist.polymer_stats()
        results = "Simulation Complete!\n\n"
        results += f"Living chains: {stats['n_living']:,}\n"
        results += f"Dead chains: {stats['n_dead']:,}\n"
        results += f"Coupled chains: {stats['n_coupled']:,}\n"
        results += f"\nPolymer Metrics:\n"
        results += f"  Mn: {poly['Mn']:,.1f} g/mol\n"
        results += f"  Mw: {poly['Mw']:,.1f} g/mol\n"
        results += f"  PDI: {poly['PDI']:.3f}\n"
        results += f"  DP_n: {poly['DP_n']:.1f}\n"
        results += f"  DP_w: {poly['DP_w']:.1f}\n"

        if self.last_kinetics is not None:
            results += f"\nKinetics tracked: {len(self.last_kinetics.timesteps)} timesteps"
            conv = self.last_kinetics.monomer_conversion[-1] * 100
            results += f"\nFinal conversion: {conv:.1f}%"

        self.results_text.setText(results)

        # Plot distribution
        self.plot_widget.canvas.plot_distribution(dist)

        # Show kinetics plots if tracked
        if self.last_kinetics is not None:
            self.kinetics_group.setVisible(True)
            self._update_kinetics_plot()
        else:
            self.kinetics_group.setVisible(False)

    def _update_kinetics_plot(self):
        """Update kinetics plot based on selector."""
        if self.last_kinetics is None:
            return
        mode = self.kinetics_selector.currentData()
        if mode == "populations":
            self.kinetics_plot.canvas.plot_chain_populations(self.last_kinetics)
        else:
            self.kinetics_plot.canvas.plot_kinetics(self.last_kinetics)

    @Slot()
    def export_kinetics(self):
        if self.last_kinetics is None:
            QMessageBox.warning(self, "No Data",
                                "No kinetics data available. Run with 'Track Kinetics' enabled.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Kinetics Data", "kinetics.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    self.last_kinetics.to_excel(file_path)
                else:
                    self.last_kinetics.to_csv(file_path)
                QMessageBox.information(self, "Export Complete",
                                        f"Kinetics data saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    @Slot(str)
    def on_simulation_error(self, error_msg: str):
        self.run_btn.setEnabled(True)
        self.results_text.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Simulation Error", error_msg)


class OptimizationTab(QWidget):
    """Tab for running FDDC optimization."""

    def __init__(self):
        super().__init__()
        self.worker: Optional[OptimizationWorker] = None
        self.run_manager = RunManager()
        self.current_config: Optional[FDDCConfig] = None
        self.current_data_path: Optional[str] = None
        self.cost_history: list = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()

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
        config_layout = QVBoxLayout()

        self.population_input = QSpinBox()
        self.population_input.setRange(10, 200)
        self.population_input.setValue(50)

        self.generations_input = QSpinBox()
        self.generations_input.setRange(5, 100)
        self.generations_input.setValue(20)

        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999)
        self.seed_input.setValue(42)

        # Worker count selector
        cpu_count = os.cpu_count() or 4
        self.workers_input = QComboBox()
        self.workers_input.addItem(f"Auto ({max(1, cpu_count - 1)} workers)", None)
        for i in range(1, cpu_count + 1):
            label = f"{i} worker{'s' if i > 1 else ''}"
            if i == 1:
                label += " (sequential)"
            elif i == cpu_count:
                label += " (max)"
            self.workers_input.addItem(label, i)
        self.workers_input.setCurrentIndex(0)

        config_layout.addLayout(_labeled_row("Population Size:", self.population_input))
        config_layout.addLayout(_labeled_row("Max Generations:", self.generations_input))
        config_layout.addLayout(_labeled_row("Random Seed:", self.seed_input, "seed"))
        config_layout.addLayout(_labeled_row("CPU Workers:", self.workers_input))
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Progress
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
        self.console_output.setMaximumHeight(120)
        self.console_output.setFont(QFont("Courier", 9))
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

        # Buttons
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

        # Results
        results_group = QGroupBox("Best Parameters")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

        # --- Right side: Convergence plot ---
        right_layout = QVBoxLayout()
        plot_group = QGroupBox("Convergence Plot")
        plot_layout = QVBoxLayout()
        self.plot_widget = PlotWidget()
        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        main_layout.addLayout(layout, 1)
        main_layout.addLayout(right_layout, 2)
        self.setLayout(main_layout)

    @Slot()
    def browse_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Experimental Data",
            str(Path.home()), "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.data_path_input.setText(file_path)

    @Slot()
    def start_optimization(self):
        data_path = self.data_path_input.text()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(self, "Invalid Data File",
                                "Please select a valid experimental data file.")
            return

        # Check if queue is running
        main_win = self.window()
        if hasattr(main_win, 'queue_tab') and main_win.queue_tab.is_running():
            reply = QMessageBox.warning(
                self, "Queue Running",
                "The Optimization Queue is currently running. "
                "Starting another optimization will compete for CPU resources.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        n_workers = self.workers_input.currentData()
        config = FDDCConfig(
            population_size=self.population_input.value(),
            max_generations=self.generations_input.value(),
            n_workers=n_workers
        )

        self.current_config = config
        self.current_data_path = data_path
        self.cost_history = []

        # Thomas's exact bounds (from 2020 thesis)
        bounds = np.array([
            [100, 3000],          # time_sim
            [10000, 120000],      # number_of_molecules
            [1000000, 5000000],   # monomer_pool
            [0.1, 0.99],          # p_growth
            [0.0001, 0.002],      # p_death
            [0.1, 0.9],           # p_dead_react
            [0.1, 0.9],           # l_exponent
            [0.1, 0.9],           # d_exponent
            [0.1, 1.0],           # l_naked
            [0, 1]                # kill_spawns_new
        ])

        seed = self.seed_input.value()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting optimization...")
        self.results_text.clear()
        self.console_output.clear()

        self.worker = OptimizationWorker(data_path, config, bounds, seed)
        self.worker.progress.connect(self.on_progress_update)
        self.worker.console_message.connect(self.on_console_message)
        self.worker.finished.connect(self.on_optimization_finished)
        self.worker.error.connect(self.on_optimization_error)
        self.worker.start()

    @Slot()
    def cancel_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.progress_label.setText("Cancelling...")

    @Slot(int, float)
    def on_progress_update(self, generation: int, cost: float):
        max_gen = self.generations_input.value()
        progress = int((generation / max_gen) * 100)
        self.progress_bar.setValue(progress)
        self.progress_label.setText(
            f"Generation {generation}/{max_gen} - Best cost: {cost:.6f}"
        )

        # Live-update convergence plot
        self.cost_history.append(cost)
        self.plot_widget.canvas.plot_convergence(self.cost_history)

    @Slot(str)
    def on_console_message(self, message: str):
        self.console_output.append(message)
        self.console_output.verticalScrollBar().setValue(
            self.console_output.verticalScrollBar().maximum()
        )

    @Slot(object)
    def on_optimization_finished(self, result):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)

        run_dir = None
        try:
            data_name = Path(self.current_data_path).stem if self.current_data_path else None
            self.run_manager.start_run("optimization", data_name)
            self.run_manager.save_optimization_config(self.current_config)
            self.run_manager.save_optimization_results(result)
            run_dir = self.run_manager.current_run_dir
            self.progress_label.setText(f"Complete! Saved to: {run_dir.name}")
        except Exception as e:
            self.progress_label.setText(f"Complete! (save failed: {e})")

        param_names = [
            'time_sim', 'number_of_molecules', 'monomer_pool',
            'p_growth', 'p_death', 'p_dead_react',
            'l_exponent', 'd_exponent', 'l_naked', 'kill_spawns_new'
        ]

        results = f"Best Cost: {result.best_cost:.6f}\n"
        if hasattr(result, 'convergence_generation') and result.convergence_generation:
            results += f"(achieved at generation {result.convergence_generation})\n\n"
        else:
            results += "\n"

        results += "Best Parameters:\n"
        for name, value in zip(param_names, result.best_params):
            if name in ['time_sim', 'number_of_molecules', 'monomer_pool']:
                results += f"  {name}: {int(value)}\n"
            elif name == 'kill_spawns_new':
                results += f"  {name}: {bool(round(value))}\n"
            else:
                results += f"  {name}: {value:.6f}\n"

        results += f"\nTotal generations: {len(result.cost_history)}\n"
        if run_dir:
            results += f"\nSaved to:\n  {run_dir}\n"

        self.results_text.setText(results)

        if result.cost_history:
            self.plot_widget.canvas.plot_convergence(result.cost_history)

    @Slot(str)
    def on_optimization_error(self, error_msg: str):
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
        self.setWindowTitle("Polymer Growth Simulation & Optimization")
        self.setMinimumSize(1100, 800)

        self.tabs = QTabWidget()
        self.sim_tab = SimulationTab()
        self.opt_tab = OptimizationTab()
        self.queue_tab = TaskQueueTab()
        self.tabs.addTab(self.sim_tab, "Simulation")
        self.tabs.addTab(self.opt_tab, "Optimization")
        self.tabs.addTab(self.queue_tab, "Optimization Queue")

        self.setCentralWidget(self.tabs)


def main():
    """Launch the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()