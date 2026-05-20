"""Graphical user interface for polymer-growth package."""

try:
    from polymer_growth.gui.app import main
except ImportError:
    def main():
        raise ImportError(
            "GUI requires PySide6. Install with: pip install polymer-growth[gui]"
        )

__all__ = ["main"]