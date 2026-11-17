#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8666}"
exec .venv/bin/python scripts/service.py start --port "$PORT"
