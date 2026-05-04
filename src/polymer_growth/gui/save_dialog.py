"""Shared save-location dialog and helpers for the GUI.

Used by both the Optimization tab (single run) and the Queue tab (batch runs)
to ask the user *before* starting where results should be saved.
"""

import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog,
)
from PySide6.QtCore import QStandardPaths
from PySide6.QtGui import QFont

from polymer_growth.core.run_manager import RunManager


# ---------------------------------------------------------------------------
# Default results directory
# ---------------------------------------------------------------------------

def default_results_dir() -> Path:
    """Platform-appropriate default: ~/Documents/PolymerGrowth/.

    Works identically from Finder, a terminal, or an IDE -- no CWD dependency.
    """
    docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
    results = Path(docs) / "PolymerGrowth"
    results.mkdir(parents=True, exist_ok=True)
    return results


# ---------------------------------------------------------------------------
# Save-location dialog
# ---------------------------------------------------------------------------

class SaveLocationDialog(QDialog):
    """Ask the user for a folder name and parent directory before a run."""

    def __init__(self, parent=None, suggested_name="", title="Save Results",
                 start_label="Start", description=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)

        layout = QVBoxLayout()

        desc_text = description or (
            "Choose a name and location for this run.\n"
            "A folder with this name will be created to hold all output files."
        )
        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        layout.addSpacing(8)

        # Folder name
        name_row = QHBoxLayout()
        name_lbl = QLabel("Folder name:")
        name_lbl.setFixedWidth(90)
        name_row.addWidget(name_lbl)
        self.name_input = QLineEdit(suggested_name)
        self.name_input.setPlaceholderText("e.g. optimization_5k_no_BB")
        self.name_input.textChanged.connect(self._update_preview)
        name_row.addWidget(self.name_input)
        layout.addLayout(name_row)

        # Parent directory
        dir_row = QHBoxLayout()
        dir_lbl = QLabel("Save to:")
        dir_lbl.setFixedWidth(90)
        dir_row.addWidget(dir_lbl)
        self.dir_input = QLineEdit(str(default_results_dir()))
        self.dir_input.textChanged.connect(self._update_preview)
        dir_row.addWidget(self.dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        # Path preview
        layout.addSpacing(4)
        self.preview = QLabel()
        self.preview.setStyleSheet("color: #555; padding: 4px;")
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        self._update_preview()

        # Buttons
        layout.addSpacing(8)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton(start_label)
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    # -- helpers --

    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Choose save location", self.dir_input.text()
        )
        if path:
            self.dir_input.setText(path)

    def _update_preview(self):
        base = self.dir_input.text().strip()
        name = self.name_input.text().strip()
        if base and name:
            self.preview.setText(f"Results will be saved to:\n{Path(base) / name}")
        else:
            self.preview.setText("Enter a folder name to continue.")

    # -- public API --

    def save_path(self) -> Path:
        """Full path = parent / folder_name."""
        return Path(self.dir_input.text().strip()) / self.name_input.text().strip()

    def folder_name(self) -> str:
        return self.name_input.text().strip()


# ---------------------------------------------------------------------------
# Saving helpers (used by both tabs)
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Turn an arbitrary string into a safe directory name."""
    return re.sub(r'[^\w\-.()\s]', '_', name).strip()


def save_optimization_to_dir(save_dir: Path, config, result,
                             seed=None, data_path=None,
                             bounds=None, elapsed_sec=None):
    """Write config, results, and cost history into *save_dir*.

    Uses RunManager's save methods for format consistency with the CLI,
    but writes directly to *save_dir* instead of creating a timestamped
    subdirectory.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    info = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "data_file": str(data_path) if data_path else None,
        "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
        "bounds": bounds.tolist() if isinstance(bounds, np.ndarray) else bounds,
        "system": {
            "platform": platform.system(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "machine": platform.machine(),
        },
    }
    with open(save_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    # Reuse RunManager's serialisation logic
    manager = RunManager(base_dir=str(save_dir))
    manager.current_run_dir = save_dir
    manager.save_optimization_config(config)
    manager.save_optimization_results(result)
