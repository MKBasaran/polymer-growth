"""Frozen-app entry point for PyInstaller bundles.

`multiprocessing.freeze_support()` MUST be called before anything else so that
spawn-based worker processes (Windows; macOS fallback) can re-enter cleanly
instead of relaunching the GUI. On macOS/Linux the optimizer forces 'fork',
so this call is effectively a no-op there, but it is harmless and required
for the Windows build.
"""

import multiprocessing
import sys


def main() -> None:
    multiprocessing.freeze_support()

    # Defer imports until after freeze_support so worker re-entry stays cheap.
    from polymer_growth.gui.app import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
    sys.exit(0)
