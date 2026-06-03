# Packaging

Standalone GUI bundle build pipeline for macOS and Windows. The bundle
embeds the Python interpreter and all dependencies; no Python install is
required on the target machine.

## Prerequisites

| Requirement   | macOS                       | Windows                                            |
| ------------- | --------------------------- | -------------------------------------------------- |
| Python        | 3.11 (see note below)       | 3.11 via `py -3.11` (see note below)               |
| DMG/installer | `hdiutil` (ships with OS)   | [Inno Setup 6](https://jrsoftware.org/isdl.php)    |

Python 3.10.0 ships bytecode that PyInstaller 6.20's modulegraph cannot
parse (`IndexError: tuple index out of range` in `dis._get_const_info`).
The build path is therefore pinned to 3.11. The project itself runs on
3.10 for source-install dev and test.

## Build

From repo root.

### macOS

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[gui,dev]"
bash packaging/build_macos.sh
```

### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -e ".[gui,dev]"
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

Output lands in `dist/` (gitignored). `src/` is not modified.

## Outputs

| Platform | Path                                                          | Form               |
| -------- | ------------------------------------------------------------- | ------------------ |
| macOS    | `dist/Polymer Growth Simulator.app`                           | Application bundle |
| macOS    | `dist/PolymerGrowthSimulator-0.1.0.dmg`                       | Disk image         |
| Windows  | `dist\Polymer Growth Simulator\PolymerGrowthSimulator.exe`    | Portable folder    |
| Windows  | `dist\PolymerGrowthSimulator-Setup-0.1.0.exe`                 | Inno Setup installer |

## CI builds

`.github/workflows/installer.yml` builds both platforms on push to the
`installer` branch and uploads the artifacts to the workflow run.

## End-user install

| Platform | Artifact                                  | Install action                                     | Install location                                                                                                          |
| -------- | ----------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| macOS    | `PolymerGrowthSimulator-0.1.0.dmg`        | Open the DMG, drag `.app` to Applications.         | `/Applications/Polymer Growth Simulator.app`                                                                              |
| Windows  | `PolymerGrowthSimulator-Setup-0.1.0.exe`  | Run the installer.                                 | `C:\Program Files\Polymer Growth Simulator\` (admin) or `%LOCALAPPDATA%\Polymer Growth Simulator\` (no admin). Start Menu shortcut included. |
| Windows  | Portable folder                           | Unzip, run `PolymerGrowthSimulator.exe`.           | Wherever the user unzipped it.                                                                                            |

Bundles are not code-signed. First-launch warnings:

- macOS: right-click, Open, Open. Alternatively
  `xattr -dr com.apple.quarantine "/Applications/Polymer Growth Simulator.app"`.
- Windows: SmartScreen, "More info", "Run anyway".

Signed distribution requires an Apple Developer ID with notarization
(macOS) and an Authenticode certificate (Windows). Neither is included.

## Files

| File                                  | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `polymer_sim_app.py`                  | Frozen-app entry. Calls `multiprocessing.freeze_support()` first.      |
| `polymer_sim.spec`                    | Cross-platform PyInstaller spec.                                       |
| `build_macos.sh`                      | macOS local build script. Produces `.app` and `.dmg`.                  |
| `build_windows.ps1`                   | Windows local build script. Produces portable folder and `.exe` installer. |
| `polymer_sim.iss`                     | Inno Setup script invoked by `build_windows.ps1`.                      |
| `../.github/workflows/installer.yml`  | CI build for both platforms.                                           |

## Fork vs spawn note

`fddc.py` exposes worker callables as module-level globals
(`_worker_objective`, `_worker_simulate`, `_worker_cost`). The
multiprocessing `Pool` invokes those globals from worker processes.

On macOS the default start method for frozen applications is `spawn`,
which re-imports the module per worker, leaving the globals at `None`.
The first worker call then raises `TypeError: 'NoneType' object is not
callable`. The Simulation tab uses `QThread` only and is unaffected;
the Optimization and Queue tabs route through `Pool` and fail.

The optimizer therefore forces `fork` at module-import time on
macOS and Linux, including when the executable is frozen. On Windows
`fork` is unavailable, so the build relies on `spawn` plus
`multiprocessing.freeze_support()` in the entry script.
