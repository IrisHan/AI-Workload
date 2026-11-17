#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required; set PYTHON=/path/to/python3.11")
try:
    import pyexpat
    import sqlite3
except ImportError as exc:
    raise SystemExit(
        "Python cannot import pyexpat or sqlite3 (%s).\n"
        "On Homebrew Python for macOS 26, pyexpat may be linked against the system libexpat,\n"
        "which is missing XML_SetAllocTrackerActivationThreshold. Install a newer expat and\n"
        "relink the pyexpat extension, then run make setup again:\n"
        "  brew install expat\n"
        "  install_name_tool -change /usr/lib/libexpat.1.dylib \"$(brew --prefix expat)/lib/libexpat.1.dylib\" <pyexpat.so>\n"
        "  codesign --force --sign - <pyexpat.so>"
        % exc
    ) from exc
PY
"$PYTHON" -m venv .venv
mkdir -p data
.venv/bin/python -m pip install -r backend/requirements.lock
.venv/bin/python -m pip install --no-deps -e './backend[dev]'
if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found. Install Node.js 22+ before building the UI." >&2
  exit 1
fi
npm --prefix frontend ci
npm --prefix frontend run build
echo "Setup finished. Next: make init-db && make start"
