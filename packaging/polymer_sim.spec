# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Polymer Growth Simulator GUI.

Builds a one-folder bundle:
- macOS:  dist/Polymer Growth Simulator.app
- Windows: dist/Polymer Growth Simulator/Polymer Growth Simulator.exe

Build with:
    pyinstaller packaging/polymer_sim.spec --noconfirm --clean
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

APP_NAME = "Polymer Growth Simulator"
ENTRY = "packaging/polymer_sim_app.py"
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

# Be aggressive about pulling in optional deps that PySide6/matplotlib/scipy
# load lazily — missing any of these is what makes a frozen build "fine on
# main tab, broken on the others." We err on the side of including everything.
hidden = []
hidden += collect_submodules("polymer_growth")
hidden += collect_submodules("scipy")
hidden += collect_submodules("scipy.special")
hidden += collect_submodules("scipy.optimize")
hidden += collect_submodules("matplotlib.backends")
hidden += [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "openpyxl",
    "openpyxl.cell._writer",
    "pkg_resources.py2_warn",
]

datas = []
datas += collect_data_files("matplotlib", subdir="mpl-data")

block_cipher = None


a = Analysis(
    [ENTRY],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "IPython",
        "pytest",
        "black",
        "ruff",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME if not IS_WIN else "PolymerGrowthSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="com.mkbasaran.polymergrowth",
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "11.0",
        },
    )
