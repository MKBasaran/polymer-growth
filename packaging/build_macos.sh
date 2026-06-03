#!/usr/bin/env bash
# Build the macOS .app and wrap it into a .dmg.
#
# Prereqs:
#   - macOS (tested on Sequoia)
#   - Python 3.11 venv at .venv with the project installed via pip install -e ".[gui,dev]"
#     (3.10.0 ships bytecode PyInstaller 6.20 cannot parse — use 3.11)
#
# Output:
#   dist/Polymer Growth Simulator.app
#   dist/PolymerGrowthSimulator-0.1.0.dmg
#
# Run from repo root:
#   bash packaging/build_macos.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Polymer Growth Simulator"
VERSION="0.1.0"
DMG_NAME="PolymerGrowthSimulator-${VERSION}.dmg"

if [[ ! -d ".venv" ]]; then
    echo "ERROR: .venv not found. Run 'python3.11 -m venv .venv && .venv/bin/pip install -e \".[gui,dev]\"' first." >&2
    exit 1
fi

# Reject 3.10 explicitly — PyInstaller's modulegraph chokes on 3.10.0 bytecode.
VENV_PY_VERSION=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$VENV_PY_VERSION" != "3.11" ]]; then
    echo "ERROR: .venv uses Python ${VENV_PY_VERSION}, but the build path requires 3.11." >&2
    echo "       Delete .venv and run: python3.11 -m venv .venv && .venv/bin/pip install -e \".[gui,dev]\"" >&2
    exit 1
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"

echo "==> Installing PyInstaller into venv (if needed)..."
"$PIP" install --quiet "pyinstaller>=6.6,<7.0"

echo "==> Cleaning prior build/dist..."
rm -rf build dist

echo "==> Running PyInstaller..."
"$PY" -m PyInstaller packaging/polymer_sim.spec --noconfirm --clean --log-level WARN

APP_PATH="dist/${APP_NAME}.app"
if [[ ! -d "$APP_PATH" ]]; then
    echo "ERROR: ${APP_PATH} was not produced. PyInstaller failed." >&2
    exit 1
fi

echo "==> Built ${APP_PATH}"
echo "    Size: $(du -sh "$APP_PATH" | cut -f1)"

echo "==> Building .dmg..."
DMG_PATH="dist/${DMG_NAME}"
rm -f "$DMG_PATH"

# hdiutil ships with macOS — no extra tool required.
# Build a staging dir so the DMG window opens with a clean view containing
# the .app and an Applications symlink.
STAGING="dist/dmg_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "$APP_PATH" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "$STAGING" \
    -ov \
    -format UDZO \
    "$DMG_PATH" \
    > /dev/null

rm -rf "$STAGING"

echo "==> Built ${DMG_PATH}"
echo "    Size: $(du -sh "$DMG_PATH" | cut -f1)"
echo
echo "Done. Test with: open \"${APP_PATH}\""
echo "Distribute:     ${DMG_PATH}"
