"""Optimization Queue tab for scheduling and running multiple FDDC optimizations.

Designed for long-running jobs (30-40 min) with rich live feedback:
- Queue management (add, remove, clear)
- Live console output (initial eval progress, generation logs)
- Live convergence plot
- Elapsed time and ETA tracking
- Results summary when each task completes
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSpinBox,
    QLineEdit, QFileDialog, QTextEdit, QMessageBox, QAbstractItemView,
    QSplitter
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QFont, QColor, QBrush

from polymer_growth.core import simulate, SimulationParams
from polymer_growth.core.run_manager import RunManager
from polymer_growth.objective import MinMaxV2ObjectiveFunction, load_experimental_data
from polymer_growth.optimizers import FDDCOptimizer, FDDCConfig
from polymer_growth.gui.plotting import PlotWidget


class TaskStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class QueueTask:
    """A task in the queue."""
    task_id: int
    task_type: str
    name: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    current_info: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    cost_history: List[float] = field(default_factory=list)

    @property
    def elapsed(self) -> Optional[timedelta]:
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return timedelta(seconds=int(end - self.start_time))

    @property
    def eta(self) -> Optional[timedelta]:
        if self.start_time is None or self.progress <= 0:
            return None
        elapsed = time.time() - self.start_time
        rate = elapsed / self.progress
        remaining = rate * (100 - self.progress)
        return timedelta(seconds=int(remaining))


_next_task_id = 0


def _new_task_id() -> int:
    global _next_task_id
    _next_task_id += 1
    return _next_task_id


class QueueWorker(QThread):
    """Runs optimization tasks from the queue one at a time."""

    task_started = Signal(int)              # task_id
    task_progress = Signal(int, int, str)   # task_id, progress%, info
    task_cost_update = Signal(int, float)   # task_id, cost
    task_console = Signal(int, str)         # task_id, console message
    task_finished = Signal(int, object)     # task_id, result
    task_failed = Signal(int, str)          # task_id, error
    queue_finished = Signal()

    def __init__(self, tasks: List[QueueTask]):
        super().__init__()
        self._tasks = tasks
        self._is_cancelled = False
        self._current_task_cancelled = False

    def cancel_all(self):
        self._is_cancelled = True
        self._current_task_cancelled = True

    def cancel_current(self):
        self._current_task_cancelled = True

    def run(self):
        for task in self._tasks:
            if self._is_cancelled:
                break
            if task.status != TaskStatus.PENDING:
                continue

            self._current_task_cancelled = False
            self.task_started.emit(task.task_id)

            try:
                result = self._run_optimization(task)

                if not self._current_task_cancelled:
                    self.task_finished.emit(task.task_id, result)
            except InterruptedError:
                self.task_failed.emit(task.task_id, "Cancelled by user")
            except Exception as e:
                self.task_failed.emit(task.task_id, str(e))

        self.queue_finished.emit()

    def _run_optimization(self, task: QueueTask):
        p = task.params
        data_path = p["data_path"]
        seed = p.get("seed", 42)

        exp_lengths, exp_values = load_experimental_data(data_path)
        objective = MinMaxV2ObjectiveFunction(exp_values)

        config = FDDCConfig(
            population_size=p.get("population_size", 50),
            max_generations=p.get("max_generations", 20),
            n_workers=p.get("n_workers", None)
        )

        max_gen = config.max_generations

        def objective_wrapper(params_array, sigma=None):
            if self._current_task_cancelled:
                raise InterruptedError("Task cancelled")

            params_array = np.asarray(params_array).flatten().tolist()
            sim_params = SimulationParams(
                time_sim=int(params_array[0]),
                number_of_molecules=int(params_array[1]),
                monomer_pool=int(params_array[2]),
                p_growth=params_array[3],
                p_death=params_array[4],
                p_dead_react=params_array[5],
                l_exponent=params_array[6],
                d_exponent=params_array[7],
                l_naked=params_array[8],
                kill_spawns_new=bool(round(params_array[9]))
            )
            rng = np.random.default_rng()
            dist = simulate(sim_params, rng)
            return objective.compute_cost(dist, sigma=sigma)

        def progress_callback(gen, cost):
            if not self._current_task_cancelled:
                pct = int((gen / max_gen) * 100)
                info = f"Gen {gen}/{max_gen} - Cost: {cost:.6f}"
                self.task_progress.emit(task.task_id, pct, info)
                self.task_cost_update.emit(task.task_id, cost)

        def console_callback(message):
            if not self._current_task_cancelled:
                self.task_console.emit(task.task_id, message)

        # Thomas's exact bounds (from 2020 thesis)
        optimizer = FDDCOptimizer(
            bounds=np.array([
                [100, 3000], [10000, 120000], [1000000, 5000000],
                [0.1, 0.99], [0.0001, 0.002], [0.1, 0.9],
                [0.1, 0.9], [0.1, 0.9], [0.1, 1.0], [0, 1]
            ]),
            objective_function=objective_wrapper,
            config=config,
            callback=progress_callback,
            console_callback=console_callback,
        )

        result = optimizer.optimize(seed=seed)

        # Save run
        manager = RunManager()
        data_name = Path(data_path).stem
        manager.start_run("optimization", f"{task.name}_{data_name}")
        manager.save_optimization_config(config)
        manager.save_optimization_results(result)

        return {"result": result, "run_dir": str(manager.current_run_dir)}


class TaskQueueTab(QWidget):
    """Tab for managing a queue of FDDC optimization tasks."""

    def __init__(self):
        super().__init__()
        self.tasks: List[QueueTask] = []
        self.worker: Optional[QueueWorker] = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_table)
        self.init_ui()

    def is_running(self) -> bool:
        """Check if any optimization is currently running."""
        return self.worker is not None and self.worker.isRunning()

    def init_ui(self):
        main_layout = QVBoxLayout()

        splitter = QSplitter(Qt.Vertical)

        # --- Top: Add task + queue table ---
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # Quick add controls -- optimization only
        add_group = QGroupBox("Add Optimization to Queue")
        add_layout = QVBoxLayout()

        # Row 1: Name + data file
        row1 = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Run name (optional)")
        self.name_input.setFixedWidth(160)

        self.opt_data_input = QLineEdit()
        self.opt_data_input.setPlaceholderText("Select experimental data file...")
        self.opt_browse_btn = QPushButton("Browse")
        self.opt_browse_btn.setFixedWidth(70)
        self.opt_browse_btn.clicked.connect(self._browse_data)

        row1.addWidget(QLabel("Name:"))
        row1.addWidget(self.name_input)
        row1.addSpacing(10)
        row1.addWidget(QLabel("Data:"))
        row1.addWidget(self.opt_data_input, 1)
        row1.addWidget(self.opt_browse_btn)
        add_layout.addLayout(row1)

        # Row 2: FDDC params -- compact with fixed widths
        row2 = QHBoxLayout()

        self.opt_pop_input = QSpinBox()
        self.opt_pop_input.setRange(10, 200)
        self.opt_pop_input.setValue(50)
        self.opt_pop_input.setFixedWidth(70)

        self.opt_gen_input = QSpinBox()
        self.opt_gen_input.setRange(5, 100)
        self.opt_gen_input.setValue(20)
        self.opt_gen_input.setFixedWidth(70)

        self.opt_seed_input = QSpinBox()
        self.opt_seed_input.setRange(0, 999999)
        self.opt_seed_input.setValue(42)
        self.opt_seed_input.setFixedWidth(80)

        cpu_count = os.cpu_count() or 4
        self.opt_workers_input = QComboBox()
        self.opt_workers_input.addItem(f"Auto ({max(1, cpu_count - 1)})", None)
        for i in range(1, cpu_count + 1):
            self.opt_workers_input.addItem(str(i), i)
        self.opt_workers_input.setFixedWidth(110)

        row2.addWidget(QLabel("Pop:"))
        row2.addWidget(self.opt_pop_input)
        row2.addSpacing(12)
        row2.addWidget(QLabel("Gens:"))
        row2.addWidget(self.opt_gen_input)
        row2.addSpacing(12)
        row2.addWidget(QLabel("Seed:"))
        row2.addWidget(self.opt_seed_input)
        row2.addSpacing(12)
        row2.addWidget(QLabel("Workers:"))
        row2.addWidget(self.opt_workers_input)
        row2.addStretch()
        add_layout.addLayout(row2)

        # Row 3: Add button + batch tools
        row3 = QHBoxLayout()
        self.add_btn = QPushButton("Add to Queue")
        self.add_btn.clicked.connect(self._add_task)

        self.add_batch_btn = QPushButton("Add Batch (vary seed)")
        self.add_batch_btn.setToolTip("Add multiple runs with different seeds")
        self.add_batch_btn.clicked.connect(self._add_batch)

        self.batch_count = QSpinBox()
        self.batch_count.setRange(2, 50)
        self.batch_count.setValue(5)
        self.batch_count.setPrefix("x")
        self.batch_count.setFixedWidth(55)

        row3.addWidget(self.add_btn)
        row3.addWidget(self.add_batch_btn)
        row3.addWidget(self.batch_count)
        row3.addStretch()
        add_layout.addLayout(row3)

        add_group.setLayout(add_layout)
        top_layout.addWidget(add_group)

        # Queue table
        queue_group = QGroupBox("Task Queue")
        queue_layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#", "Name", "Status", "Progress", "Elapsed", "ETA"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.currentCellChanged.connect(
            lambda row, _col, _prev_row, _prev_col: self._on_row_selected(row)
        )
        queue_layout.addWidget(self.table)

        # Queue controls
        ctrl_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Queue")
        self.run_btn.clicked.connect(self._run_queue)
        self.cancel_btn = QPushButton("Cancel Current")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_current)
        self.cancel_all_btn = QPushButton("Cancel All")
        self.cancel_all_btn.setEnabled(False)
        self.cancel_all_btn.clicked.connect(self._cancel_all)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.clear_btn = QPushButton("Clear Finished")
        self.clear_btn.clicked.connect(self._clear_finished)

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.cancel_all_btn)
        ctrl_layout.addWidget(self.remove_btn)
        ctrl_layout.addWidget(self.clear_btn)
        ctrl_layout.addStretch()

        self.overall_label = QLabel("Queue: 0 tasks")
        ctrl_layout.addWidget(self.overall_label)

        queue_layout.addLayout(ctrl_layout)
        queue_group.setLayout(queue_layout)
        top_layout.addWidget(queue_group)

        splitter.addWidget(top_widget)

        # --- Bottom: Detail + console + live plot ---
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)

        # Left: detail + console stacked
        left_bottom = QVBoxLayout()

        detail_group = QGroupBox("Task Details")
        detail_layout = QVBoxLayout()
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Courier", 9))
        self.detail_text.setMaximumHeight(140)
        detail_layout.addWidget(self.detail_text)
        detail_group.setLayout(detail_layout)
        left_bottom.addWidget(detail_group)

        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Courier", 9))
        self.console_output.setMaximumHeight(140)
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        left_bottom.addWidget(console_group)

        left_wrapper = QWidget()
        left_wrapper.setLayout(left_bottom)
        bottom_layout.addWidget(left_wrapper, 1)

        # Right: live convergence plot
        plot_group = QGroupBox("Live Convergence")
        plot_layout = QVBoxLayout()
        self.live_plot = PlotWidget()
        plot_layout.addWidget(self.live_plot)
        plot_group.setLayout(plot_layout)
        bottom_layout.addWidget(plot_group, 2)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([400, 300])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _browse_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Experimental Data",
            str(Path.home()), "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.opt_data_input.setText(file_path)

    def _add_task(self):
        data_path = self.opt_data_input.text().strip()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(self, "Missing Data",
                                "Select an experimental data file first.")
            return

        name = self.name_input.text().strip() or f"Opt {_new_task_id() + 1}"
        params = {
            "data_path": data_path,
            "population_size": self.opt_pop_input.value(),
            "max_generations": self.opt_gen_input.value(),
            "seed": self.opt_seed_input.value(),
            "n_workers": self.opt_workers_input.currentData(),
        }

        task = QueueTask(
            task_id=_new_task_id(),
            task_type="optimization",
            name=name,
            params=params
        )
        self.tasks.append(task)
        self._refresh_table()
        self._update_overall_label()

    def _add_batch(self):
        """Add multiple optimization runs with varying seeds."""
        count = self.batch_count.value()
        base_seed = self.opt_seed_input.value()

        for i in range(count):
            self.opt_seed_input.setValue(base_seed + i)
            name = self.name_input.text().strip() or "Batch"
            self.name_input.setText(f"{name} (seed={base_seed + i})")
            self._add_task()

        self.name_input.clear()
        self.opt_seed_input.setValue(base_seed)

    def _remove_selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.tasks):
            task = self.tasks[row]
            if task.status == TaskStatus.RUNNING:
                QMessageBox.warning(self, "Cannot Remove",
                                    "Cannot remove a running task. Cancel it first.")
                return
            self.tasks.pop(row)
            self._refresh_table()
            self._update_overall_label()

    def _clear_finished(self):
        self.tasks = [t for t in self.tasks
                      if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
        self._refresh_table()
        self._update_overall_label()

    def _run_queue(self):
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if not pending:
            QMessageBox.information(self, "Nothing to Run", "No pending tasks in queue.")
            return

        # Check if optimization tab is running
        main_win = self.window()
        if hasattr(main_win, 'opt_tab'):
            opt_tab = main_win.opt_tab
            if opt_tab.worker and opt_tab.worker.isRunning():
                reply = QMessageBox.warning(
                    self, "Optimization Running",
                    "An optimization is already running in the Optimization tab. "
                    "Running the queue will compete for CPU resources.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_all_btn.setEnabled(True)
        self.console_output.clear()

        self.worker = QueueWorker(self.tasks)
        self.worker.task_started.connect(self._on_task_started)
        self.worker.task_progress.connect(self._on_task_progress)
        self.worker.task_cost_update.connect(self._on_task_cost)
        self.worker.task_console.connect(self._on_console_message)
        self.worker.task_finished.connect(self._on_task_finished)
        self.worker.task_failed.connect(self._on_task_failed)
        self.worker.queue_finished.connect(self._on_queue_finished)
        self.worker.start()

        self._timer.start(1000)

    def _cancel_current(self):
        if self.worker:
            self.worker.cancel_current()

    def _cancel_all(self):
        if self.worker:
            self.worker.cancel_all()
            for t in self.tasks:
                if t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.CANCELLED

    def _find_task(self, task_id: int) -> Optional[QueueTask]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def _find_task_row(self, task_id: int) -> int:
        for i, t in enumerate(self.tasks):
            if t.task_id == task_id:
                return i
        return -1

    @Slot(int)
    def _on_task_started(self, task_id: int):
        task = self._find_task(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            task.cost_history = []
            self._refresh_table()
            self._update_overall_label()
            self.console_output.clear()
            self.console_output.append(f"--- Starting: {task.name} ---")
            # Auto-select the running task row
            row = self._find_task_row(task_id)
            if row >= 0:
                self.table.selectRow(row)

    @Slot(int, int, str)
    def _on_task_progress(self, task_id: int, progress: int, info: str):
        task = self._find_task(task_id)
        if task:
            task.progress = progress
            task.current_info = info
            self._refresh_table()
            self._update_detail_for_task(task)

    @Slot(int, float)
    def _on_task_cost(self, task_id: int, cost: float):
        task = self._find_task(task_id)
        if task:
            task.cost_history.append(cost)
            # Always update plot for the running task
            if len(task.cost_history) >= 1:
                self.live_plot.canvas.plot_convergence(task.cost_history)

    @Slot(int, str)
    def _on_console_message(self, task_id: int, message: str):
        self.console_output.append(message)
        # Auto-scroll
        sb = self.console_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(int, object)
    def _on_task_finished(self, task_id: int, result):
        task = self._find_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            task.progress = 100
            task.result = result
            task.current_info = "Completed"
            self._refresh_table()
            self._update_overall_label()
            self._update_detail_for_task(task)
            self.console_output.append(f"--- Completed: {task.name} ---\n")

    @Slot(int, str)
    def _on_task_failed(self, task_id: int, error: str):
        task = self._find_task(task_id)
        if task:
            task.status = (TaskStatus.CANCELLED if "cancel" in error.lower()
                           else TaskStatus.FAILED)
            task.end_time = time.time()
            task.error = error
            task.current_info = error
            self._refresh_table()
            self._update_overall_label()
            self.console_output.append(f"--- Failed: {error} ---\n")

    @Slot()
    def _on_queue_finished(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_all_btn.setEnabled(False)
        self._timer.stop()
        self._refresh_table()
        self._update_overall_label()

        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        QMessageBox.information(
            self, "Queue Complete",
            f"Queue finished: {completed} completed, {failed} failed."
        )

    def _on_row_selected(self, row):
        if 0 <= row < len(self.tasks):
            task = self.tasks[row]
            self._update_detail_for_task(task)
            if task.cost_history:
                self.live_plot.canvas.plot_convergence(task.cost_history)

    def _update_detail_for_task(self, task: QueueTask):
        lines = [
            f"Task #{task.task_id}: {task.name}",
            f"Status: {task.status.value}",
            f"Data: {Path(task.params.get('data_path', '')).name}",
            f"Pop: {task.params.get('population_size')}  "
            f"Gens: {task.params.get('max_generations')}  "
            f"Seed: {task.params.get('seed')}",
        ]

        if task.elapsed:
            lines.append(f"Elapsed: {task.elapsed}")
        if task.eta and task.status == TaskStatus.RUNNING:
            lines.append(f"ETA: {task.eta}")

        if task.status == TaskStatus.RUNNING and task.current_info:
            lines.append(f"\n{task.current_info}")

        # Show results summary for completed tasks
        if task.status == TaskStatus.COMPLETED and task.result:
            res = task.result
            if isinstance(res, dict):
                opt_result = res.get("result")
                if opt_result and hasattr(opt_result, 'best_cost'):
                    lines.append(f"\n--- Results ---")
                    lines.append(f"Best cost: {opt_result.best_cost:.6f}")
                    if hasattr(opt_result, 'convergence_generation') and opt_result.convergence_generation:
                        lines.append(f"Converged at gen: {opt_result.convergence_generation}")
                    lines.append(f"Generations run: {len(opt_result.cost_history)}")

                    # Best parameters
                    param_names = [
                        'time_sim', 'n_molecules', 'monomer_pool',
                        'p_growth', 'p_death', 'p_dead_react',
                        'l_exp', 'd_exp', 'l_naked', 'kill_spawns'
                    ]
                    lines.append(f"\nBest parameters:")
                    for pname, val in zip(param_names, opt_result.best_params):
                        if pname in ('time_sim', 'n_molecules', 'monomer_pool'):
                            lines.append(f"  {pname}: {int(val)}")
                        elif pname == 'kill_spawns':
                            lines.append(f"  {pname}: {bool(round(val))}")
                        else:
                            lines.append(f"  {pname}: {val:.6f}")

                run_dir = res.get("run_dir", "")
                if run_dir:
                    lines.append(f"\nSaved: {Path(run_dir).name}")

        if task.cost_history:
            best = min(task.cost_history)
            best_gen = task.cost_history.index(best) + 1
            lines.append(f"\nBest cost so far: {best:.6f} (gen {best_gen})")

        self.detail_text.setText("\n".join(lines))

    def _update_overall_label(self):
        total = len(self.tasks)
        pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)
        done = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks
                     if t.status in (TaskStatus.FAILED, TaskStatus.CANCELLED))
        self.overall_label.setText(
            f"Queue: {total} total | {pending} pending | {running} running | "
            f"{done} done | {failed} failed"
        )

    def _refresh_table(self):
        self.table.setRowCount(len(self.tasks))

        status_colors = {
            TaskStatus.PENDING: QColor("#f0f0f0"),
            TaskStatus.RUNNING: QColor("#fff3cd"),
            TaskStatus.COMPLETED: QColor("#d4edda"),
            TaskStatus.FAILED: QColor("#f8d7da"),
            TaskStatus.CANCELLED: QColor("#e2e3e5"),
        }

        for row, task in enumerate(self.tasks):
            color = status_colors.get(task.status, QColor("white"))
            brush = QBrush(color)

            items = [
                str(task.task_id),
                task.name,
                task.status.value,
                (f"{task.progress}% - {task.current_info}"
                 if task.current_info else f"{task.progress}%"),
                str(task.elapsed or "-"),
                str(task.eta or "-") if task.status == TaskStatus.RUNNING else "-",
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setBackground(brush)
                self.table.setItem(row, col, item)
