"""
TASK QUEUE SYSTEM: Batch Optimization Manager
=============================================

This standalone module provides a task queue system for:
- Queueing multiple optimization runs
- Running them sequentially or in parallel
- Progress tracking and status updates
- Results collection and comparison

Use cases:
- Run multiple datasets through optimization
- Compare different configurations
- Batch parameter sweeps
- Overnight batch processing

This is a STANDALONE prototype - does NOT modify production code.
"""

import uuid
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Any, Dict
from enum import Enum
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OptimizationTask:
    """Represents a single optimization task in the queue."""

    # Task identification
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    # Configuration
    data_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0-100%
    current_generation: int = 0
    best_cost: float = float('inf')

    # Timing
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'OptimizationTask':
        """Create from dictionary."""
        d = d.copy()
        d['status'] = TaskStatus(d['status'])
        return cls(**d)

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None

        start = datetime.fromisoformat(self.started_at)

        if self.completed_at:
            end = datetime.fromisoformat(self.completed_at)
        else:
            end = datetime.now()

        return (end - start).total_seconds()


class TaskQueueCallbacks:
    """Callbacks for task queue events (for GUI integration)."""

    def on_task_added(self, task: OptimizationTask):
        """Called when a task is added to the queue."""
        pass

    def on_task_started(self, task: OptimizationTask):
        """Called when a task starts running."""
        pass

    def on_task_progress(self, task: OptimizationTask, generation: int, cost: float):
        """Called when task progress updates."""
        pass

    def on_task_completed(self, task: OptimizationTask):
        """Called when a task completes (success or failure)."""
        pass

    def on_queue_empty(self):
        """Called when queue becomes empty."""
        pass


class TaskQueue:
    """
    Task queue manager for batch optimization runs.

    Features:
    - Add/remove tasks
    - Start/stop/pause queue processing
    - Progress tracking
    - Results collection
    - Persistence (save/load queue state)
    """

    def __init__(self, callbacks: TaskQueueCallbacks = None,
                 max_concurrent: int = 1):
        """
        Initialize task queue.

        Args:
            callbacks: Optional callbacks for GUI integration
            max_concurrent: Max tasks to run concurrently (1 = sequential)
        """
        self.tasks: List[OptimizationTask] = []
        self.callbacks = callbacks or TaskQueueCallbacks()
        self.max_concurrent = max_concurrent

        self._running = False
        self._paused = False
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def add_task(self, name: str, data_path: str, config: dict) -> OptimizationTask:
        """
        Add a new task to the queue.

        Args:
            name: Human-readable task name
            data_path: Path to target data file
            config: FDDCConfig parameters as dict

        Returns:
            Created OptimizationTask
        """
        task = OptimizationTask(
            name=name,
            data_path=data_path,
            config=config,
        )

        with self._lock:
            self.tasks.append(task)

        self.callbacks.on_task_added(task)
        return task

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue (only if pending)."""
        with self._lock:
            for i, task in enumerate(self.tasks):
                if task.id == task_id:
                    if task.status == TaskStatus.PENDING:
                        self.tasks.pop(i)
                        return True
                    elif task.status == TaskStatus.RUNNING:
                        # Cancel running task
                        if task_id in self._futures:
                            self._futures[task_id].cancel()
                            task.status = TaskStatus.CANCELLED
                            return True
        return False

    def get_task(self, task_id: str) -> Optional[OptimizationTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_pending_tasks(self) -> List[OptimizationTask]:
        """Get all pending tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_running_tasks(self) -> List[OptimizationTask]:
        """Get all running tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def get_completed_tasks(self) -> List[OptimizationTask]:
        """Get all completed tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def clear_completed(self):
        """Remove all completed tasks."""
        with self._lock:
            self.tasks = [t for t in self.tasks
                         if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)]

    def start(self, run_task_fn: Callable[[OptimizationTask, Callable], None]):
        """
        Start processing the queue.

        Args:
            run_task_fn: Function that runs a task. Signature:
                         run_task_fn(task, progress_callback)
                         progress_callback(generation, cost)
        """
        if self._running:
            return

        self._running = True
        self._paused = False
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        # Start processing thread
        threading.Thread(target=self._process_queue, args=(run_task_fn,),
                        daemon=True).start()

    def stop(self):
        """Stop processing the queue."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def pause(self):
        """Pause queue processing (current tasks continue)."""
        self._paused = True

    def resume(self):
        """Resume queue processing."""
        self._paused = False

    def _process_queue(self, run_task_fn: Callable):
        """Main queue processing loop."""
        while self._running:
            if self._paused:
                time.sleep(0.5)
                continue

            # Get next pending task
            pending = self.get_pending_tasks()
            running = self.get_running_tasks()

            if len(running) >= self.max_concurrent or not pending:
                # Wait for running tasks or new tasks
                time.sleep(0.5)

                # Check if queue is empty and no tasks running
                if not pending and not running:
                    self.callbacks.on_queue_empty()

                continue

            # Start next task
            task = pending[0]
            self._start_task(task, run_task_fn)

    def _start_task(self, task: OptimizationTask,
                    run_task_fn: Callable):
        """Start a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        self.callbacks.on_task_started(task)

        def progress_callback(generation: int, cost: float):
            task.current_generation = generation
            task.best_cost = cost
            # Estimate progress (assuming max_generations in config)
            max_gen = task.config.get('max_generations', 100)
            task.progress = min(100, (generation / max_gen) * 100)
            self.callbacks.on_task_progress(task, generation, cost)

        def run_wrapper():
            try:
                result = run_task_fn(task, progress_callback)
                task.result = result
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
            finally:
                task.completed_at = datetime.now().isoformat()
                task.progress = 100
                self.callbacks.on_task_completed(task)

        future = self._executor.submit(run_wrapper)
        self._futures[task.id] = future

    def save(self, path: str):
        """Save queue state to file."""
        data = {
            'tasks': [t.to_dict() for t in self.tasks],
            'saved_at': datetime.now().isoformat(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load queue state from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.tasks = [OptimizationTask.from_dict(t) for t in data['tasks']]

        # Reset running tasks to pending (interrupted)
        for task in self.tasks:
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PENDING


# =============================================================================
# GUI INTEGRATION: PyQt5 Widget Example
# =============================================================================

PYQT_WIDGET_EXAMPLE = '''
"""
PyQt5 Task Queue Widget - Integration Example

Copy this to the GUI when ready to integrate.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QProgressBar, QLabel, QHeaderView, QComboBox,
    QSpinBox, QFileDialog, QGroupBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from task_queue import TaskQueue, TaskQueueCallbacks, OptimizationTask, TaskStatus


class TaskQueueWidget(QWidget):
    """Widget for managing optimization task queue."""

    task_started = pyqtSignal(str)  # task_id
    task_completed = pyqtSignal(str, dict)  # task_id, result

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_queue()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        controls = QHBoxLayout()

        self.add_btn = QPushButton("Add Task")
        self.add_btn.clicked.connect(self._add_task_dialog)

        self.start_btn = QPushButton("Start Queue")
        self.start_btn.clicked.connect(self._toggle_queue)

        self.clear_btn = QPushButton("Clear Completed")
        self.clear_btn.clicked.connect(self._clear_completed)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 8)
        self.workers_spin.setValue(1)
        self.workers_spin.setPrefix("Workers: ")

        controls.addWidget(self.add_btn)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.clear_btn)
        controls.addWidget(self.workers_spin)
        controls.addStretch()

        layout.addLayout(controls)

        # Task table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Name", "Status", "Progress", "Generation", "Best Cost", "Duration"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Status bar
        self.status_label = QLabel("Queue: Idle")
        layout.addWidget(self.status_label)

    def _setup_queue(self):
        callbacks = TaskQueueCallbacks()
        callbacks.on_task_added = self._on_task_added
        callbacks.on_task_started = self._on_task_started
        callbacks.on_task_progress = self._on_task_progress
        callbacks.on_task_completed = self._on_task_completed
        callbacks.on_queue_empty = self._on_queue_empty

        self.queue = TaskQueue(callbacks=callbacks)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_table)
        self.update_timer.start(1000)

    def _add_task_dialog(self):
        """Show dialog to add new task."""
        # In real implementation, show dialog to select data file and config
        pass

    def add_task(self, name: str, data_path: str, config: dict):
        """Add task to queue."""
        self.queue.add_task(name, data_path, config)

    def _toggle_queue(self):
        """Start or stop queue processing."""
        if self.queue._running:
            self.queue.stop()
            self.start_btn.setText("Start Queue")
        else:
            self.queue.max_concurrent = self.workers_spin.value()
            self.queue.start(self._run_optimization_task)
            self.start_btn.setText("Stop Queue")

    def _run_optimization_task(self, task: OptimizationTask, progress_cb):
        """Run a single optimization task."""
        # This would call the actual FDDC optimizer
        # For now, simulate
        import time
        import random

        max_gen = task.config.get('max_generations', 100)
        cost = 1000.0

        for gen in range(max_gen):
            time.sleep(0.1)  # Simulate work
            cost *= 0.95 + random.random() * 0.1
            progress_cb(gen, cost)

            if cost < 10:
                break

        return {'best_cost': cost, 'generations': gen}

    def _clear_completed(self):
        self.queue.clear_completed()
        self._update_table()

    def _update_table(self):
        """Update task table."""
        self.table.setRowCount(len(self.queue.tasks))

        for row, task in enumerate(self.queue.tasks):
            self.table.setItem(row, 0, QTableWidgetItem(task.name))
            self.table.setItem(row, 1, QTableWidgetItem(task.status.value))

            # Progress bar
            progress = QProgressBar()
            progress.setValue(int(task.progress))
            self.table.setCellWidget(row, 2, progress)

            self.table.setItem(row, 3, QTableWidgetItem(str(task.current_generation)))
            self.table.setItem(row, 4, QTableWidgetItem(f"{task.best_cost:.4f}"))

            duration = task.duration
            if duration:
                self.table.setItem(row, 5, QTableWidgetItem(f"{duration:.1f}s"))
            else:
                self.table.setItem(row, 5, QTableWidgetItem("-"))

    def _on_task_added(self, task):
        self._update_table()

    def _on_task_started(self, task):
        self.task_started.emit(task.id)
        self.status_label.setText(f"Running: {task.name}")

    def _on_task_progress(self, task, gen, cost):
        self._update_table()

    def _on_task_completed(self, task):
        self.task_completed.emit(task.id, task.result or {})
        self._update_table()

    def _on_queue_empty(self):
        self.status_label.setText("Queue: Idle")
        self.start_btn.setText("Start Queue")
'''


# =============================================================================
# DEMO
# =============================================================================

def demo_task_queue():
    """Demonstrate the task queue system."""
    import random

    print("=" * 60)
    print("TASK QUEUE SYSTEM DEMO")
    print("=" * 60)

    # Create callbacks for console output
    class ConsoleCallbacks(TaskQueueCallbacks):
        def on_task_added(self, task):
            print(f"[+] Task added: {task.name} ({task.id})")

        def on_task_started(self, task):
            print(f"[>] Task started: {task.name}")

        def on_task_progress(self, task, gen, cost):
            print(f"    Gen {gen}: cost={cost:.4f}", end='\r')

        def on_task_completed(self, task):
            status = "SUCCESS" if task.status == TaskStatus.COMPLETED else "FAILED"
            print(f"\n[{status}] Task completed: {task.name}")
            if task.duration:
                print(f"    Duration: {task.duration:.1f}s")
            if task.result:
                print(f"    Result: {task.result}")

        def on_queue_empty(self):
            print("\n[*] Queue empty - all tasks completed")

    # Create queue
    queue = TaskQueue(callbacks=ConsoleCallbacks(), max_concurrent=2)

    # Add some tasks
    print("\nAdding tasks to queue...")

    queue.add_task(
        name="Dataset A (5K)",
        data_path="data/5K.csv",
        config={'max_generations': 50, 'population_size': 30}
    )

    queue.add_task(
        name="Dataset B (10K)",
        data_path="data/10K.csv",
        config={'max_generations': 50, 'population_size': 30}
    )

    queue.add_task(
        name="Dataset C (20K)",
        data_path="data/20K.csv",
        config={'max_generations': 50, 'population_size': 30}
    )

    # Define task runner (simulates optimization)
    def run_task(task: OptimizationTask, progress_cb: Callable):
        max_gen = task.config.get('max_generations', 50)
        cost = 1000.0

        for gen in range(max_gen):
            time.sleep(0.05)  # Simulate work
            cost *= 0.9 + random.random() * 0.15
            progress_cb(gen, cost)

            if cost < 10:
                break

        return {'best_cost': cost, 'generations': gen}

    # Start queue
    print("\nStarting queue processing (2 concurrent tasks)...")
    queue.start(run_task)

    # Wait for completion
    while queue.get_pending_tasks() or queue.get_running_tasks():
        time.sleep(0.5)

    queue.stop()

    # Show summary
    print("\n" + "=" * 60)
    print("TASK SUMMARY")
    print("=" * 60)

    for task in queue.tasks:
        print(f"\n{task.name}:")
        print(f"  Status: {task.status.value}")
        print(f"  Duration: {task.duration:.1f}s")
        if task.result:
            print(f"  Best cost: {task.result.get('best_cost', 'N/A'):.4f}")
            print(f"  Generations: {task.result.get('generations', 'N/A')}")

    # Save queue state
    queue.save("/tmp/task_queue_state.json")
    print("\nQueue state saved to /tmp/task_queue_state.json")


if __name__ == "__main__":
    demo_task_queue()
