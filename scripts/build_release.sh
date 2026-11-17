#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=0.1.0
NAME="ai-workload-${VERSION}"
STAGE="release/staging/${NAME}"
ARCHIVE="release/${NAME}.tar.gz"

if [[ ! -f frontend/dist/index.html ]]; then
  echo "frontend/dist is missing. Run: make build" >&2
  exit 1
fi

rm -rf release/staging
mkdir -p "$STAGE/backend" "$STAGE/frontend" "$STAGE/scripts" "$STAGE/database"
cp backend/pyproject.toml backend/requirements.lock backend/alembic.ini "$STAGE/backend/"
cp -R backend/app backend/migrations "$STAGE/backend/"
rm -rf "$STAGE/backend/app/__pycache__" "$STAGE/backend/migrations/__pycache__" \
  "$STAGE/backend/migrations/versions/__pycache__"
find "$STAGE/backend" -type d -name '__pycache__' -exec rm -rf {} +
cp -R frontend/dist "$STAGE/frontend/dist"
cp -R database/bundle "$STAGE/database/bundle"
cp scripts/service.py scripts/migration_db.py scripts/init_db.sh "$STAGE/scripts/"
cp start.sh stop.sh "$STAGE/"
chmod 755 "$STAGE/start.sh" "$STAGE/stop.sh" "$STAGE/scripts/init_db.sh"

cat > "$STAGE/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ required; set PYTHON=/path/to/python3.11")
try:
    import pyexpat, sqlite3
except ImportError as exc:
    raise SystemExit("This Python cannot import pyexpat or sqlite3: %s" % exc)
PY
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.lock
.venv/bin/python -m pip install --no-deps -e './backend'
bash scripts/init_db.sh
echo
echo "Installed. Start with: bash start.sh"
echo "Open http://127.0.0.1:8666"
EOF
chmod 755 "$STAGE/install.sh"

cat > "$STAGE/README.md" <<EOF
# AI Workload ${VERSION}

Runnable package for macOS and Linux. Python 3.11 or newer is required. Node.js is not required. Windows and Docker are not supported.

The application is still in development. Version 1.0 is planned for 2027.

\`\`\`bash
bash install.sh
bash start.sh
\`\`\`

Open http://127.0.0.1:8666 . API reference: http://127.0.0.1:8666/docs .

\`bash start.sh\` listens on 127.0.0.1 port 8666. Choose another port with \`PORT=9000 bash start.sh\`. Stop with \`bash stop.sh\`. Status is \` .venv/bin/python scripts/service.py status\`.

The database file is \`data/planner.db\`. \`install.sh\` creates it from \`database/bundle/\` when the file is absent. Set \`DATABASE_URL=sqlite:////absolute/path/planner.db\` before start to use another file. There is no other configuration file.
EOF

tar -czf "$ARCHIVE" -C release/staging "$NAME"
rm -rf release/staging
echo "Wrote $ARCHIVE"
ls -lh "$ARCHIVE"
