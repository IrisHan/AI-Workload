#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Create the environment first: make setup" >&2
  exit 1
fi
mkdir -p data
if [[ -e data/planner.db ]]; then
  echo "data/planner.db already exists. Leaving it unchanged."
  exit 0
fi
.venv/bin/python scripts/migration_db.py restore --bundle database/bundle --sqlite data/planner.db
.venv/bin/python scripts/migration_db.py verify --bundle database/bundle --sqlite data/planner.db
echo "Initialized data/planner.db from database/bundle."
