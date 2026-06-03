# Packaging

Builds standalone installers for the Polymer Growth Simulator GUI.

## What's here

| File                     | Purpose                                                                   |
| ------------------------ | ------------------------------------------------------------------------- |
| `polymer_sim_app.py`     | Frozen-app entry point. Calls `multiprocessing.freeze_support()` first.   |
| `polymer_sim.spec`       | Cross-platform PyInstaller spec (produces `.app` on macOS, folder on Win) |
| `build_macos.sh`         | macOS local build → `.app` + `.dmg`                                       |
| `build_windows.ps1`      | Windows local build → folder bundle + Inno Setup installer                |
| `polymer_sim.iss`        | Inno Setup script used by `build_windows.ps1`                             |
| `../.github/workflows/installer.yml` | CI build for both platforms on push to `installer` branch     |

## Why the previous bundle only ran one of three tabs

`fddc.py` registers worker callables as module-level globals (`_worker_objective` etc.)
which the multiprocessing Pool calls from worker processes. The macOS default
start method for frozen apps is `spawn`, which re-imports the module per
worker — so those globals are `None`, and the first call raises
`TypeError: 'NoneType' object is not callable`. The Simulation tab uses
QThread only and was unaffected; Optimization + Queue both went through the
Pool and crashed.

Fix is one line: force the `fork` start method even when frozen (macOS/Linux).
On Windows there is no `fork`, so we keep `spawn` and rely on
`multiprocessing.freeze_support()` plus PyInstaller's normal hooks. The pool
codepath is wrapped in closures that don't survive `spawn` pickling either,
but that hits Windows only — handle it there separately if it becomes a real
problem (workaround: set workers to 1 in the GUI dropdown).

## Build locally on macOS

```bash
# from repo root, with venv ready
bash packaging/build_macos.sh

# outputs:
#   dist/Polymer Growth Simulator.app
#   dist/PolymerGrowthSimulator-0.1.0.dmg
```

Test the bundle directly:

```bash
open "dist/Polymer Growth Simulator.app"
```

## Build locally on Windows

Install Python 3.10 and [Inno Setup 6](https://jrsoftware.org/isdl.php), then:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
```

## Build in CI (recommended for the Windows .exe since dev is on macOS)

Push to `installer`. The workflow at `.github/workflows/installer.yml` builds
both macOS and Windows artifacts. Download them from the run page on GitHub
once it completes (~10–15 min).

## Distribution notes

- **Not code-signed.** macOS will refuse the `.app` on first launch via
  Gatekeeper. Either:
  - Right-click → Open → Open (one-time bypass), or
  - `xattr -dr com.apple.quarantine "dist/Polymer Growth Simulator.app"`
- Real distribution needs an Apple Developer ID and notarization; out of scope
  for the thesis deliverable.
- The Windows installer will show a SmartScreen warning for the same reason.
  Users can click "More info" → "Run anyway".
