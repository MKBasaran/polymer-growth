# Packaging

Standalone installers for the Polymer Growth Simulator GUI.

## Files

| File                                  | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `polymer_sim_app.py`                  | Frozen-app entry; calls `multiprocessing.freeze_support()` first.      |
| `polymer_sim.spec`                    | Cross-platform PyInstaller spec.                                       |
| `build_macos.sh`                      | macOS local build → `.app` + `.dmg`.                                   |
| `build_windows.ps1`                   | Windows local build → portable folder + Inno Setup installer.          |
| `polymer_sim.iss`                     | Inno Setup script used by `build_windows.ps1`.                         |
| `../.github/workflows/installer.yml`  | CI: builds both platforms on push to `installer` branch.               |

## Why the previous bundle only ran one of three tabs

`fddc.py` registers worker callables as module-level globals
(`_worker_objective` etc.) which the multiprocessing Pool calls from worker
processes. The macOS default start method for frozen apps is `spawn`, which
re-imports the module per worker — so those globals are `None`, and the
first call raises `TypeError: 'NoneType' object is not callable`. The
Simulation tab uses `QThread` only and was unaffected; Optimization + Queue
both went through the Pool and crashed.

Fix is one line: force `fork` even when frozen on macOS/Linux. On Windows
there is no `fork`, so it keeps `spawn` plus `multiprocessing.freeze_support()`.

## Build locally on macOS (Kaan's dev box)

```bash
bash packaging/build_macos.sh
# outputs:
#   dist/Polymer Growth Simulator.app
#   dist/PolymerGrowthSimulator-0.1.0.dmg
```

Open the bundle directly:

```bash
open "dist/Polymer Growth Simulator.app"
```

## Build locally on Windows

Install Python 3.10/3.11 and [Inno Setup 6](https://jrsoftware.org/isdl.php), then:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

## Build in CI (recommended for Windows since I'm on macOS)

Push to `installer`. The workflow at `.github/workflows/installer.yml` builds
both macOS and Windows artifacts. Grab them from the run page on GitHub
once it finishes (~10–15 min).

## Distribution

- Not code-signed. macOS Gatekeeper will block on first launch — either
  right-click → Open → Open, or:
  `xattr -dr com.apple.quarantine "dist/Polymer Growth Simulator.app"`
- Windows SmartScreen will warn for the same reason: "More info" → "Run anyway".
- Proper distribution needs an Apple Developer ID + notarization (out of
  scope for the thesis deliverable).

— Kaan Basaran
