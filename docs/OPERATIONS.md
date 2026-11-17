# Operations

This note is the current run path for AI Workload 0.1.0. Historical design notes and delivery reports are not part of the published tree.

## Scripts

| Command | What it does |
|---|---|
| `make setup` | Create `.venv`, install `backend/requirements.lock`, install the backend editable, `npm ci`, build the UI |
| `make init-db` | If `data/planner.db` is absent, restore `database/bundle/` into it |
| `make build` | Rebuild `frontend/dist` with TypeScript and Vite |
| `make start` | Start one local process on `127.0.0.1` and the selected port |
| `make stop` | Stop the process recorded in `data/service.json` |
| `make status` | Print pid, port, URL, and whether `/api/health` returns `ok` |
| `make release` | Rebuild `release/ai-workload-0.1.0.tar.gz` from the current tree and `frontend/dist` |
| `bash start.sh` / `bash stop.sh` | Same start and stop commands without Make |

`scripts/smoke.py` checks the running preview: health, a known model parameter count, a known GPU figure, OpenAPI, and the static UI.

```bash
.venv/bin/python scripts/smoke.py --base-url http://127.0.0.1:8666
```

## Ports and URLs

The listen port is not read from a config file.

- Preview and API: `http://127.0.0.1:8666`
- Override: `PORT=9000 make start`
- The process binds `127.0.0.1` only
- If the requested port is taken, start searches the next 20 ports and prints the URL
- Dev UI, only while `npm --prefix frontend run dev` is running: `http://127.0.0.1:5173`
- That dev server proxies `/api`, `/docs`, and `/openapi.json` to `http://127.0.0.1:8666`
- Health: `GET /api/health`
- API reference: `http://127.0.0.1:8666/docs`

`DATABASE_URL` selects the database. When it is unset, the file is `data/planner.db` under the repository root. Example:

```bash
DATABASE_URL=sqlite:////absolute/path/planner.db make start
```

`deployment/env.example` lists the variables. Empty means “use the default”.

## Database

SQLite only. Alembic revision `0003` is the schema. `database/bundle/` is the published initial data: catalog rows, structure snapshots, and workload history. `make init-db` loads it once. A later start migrates to head and applies idempotent seeds; it does not wipe the file.

The live file, its WAL, `data/server.log`, and `data/service.json` stay on the machine. They are listed in `.gitignore`.

## Python

Supported interpreter: CPython 3.11 or newer. `make setup` checks the version and imports `pyexpat` and `sqlite3` before it creates `.venv`.

Locked install: `backend/requirements.lock`. Optional override for the package index: `PIP_INDEX_URL`.

Node.js 22+ is required to build the UI from source. The release tarball already contains `frontend/dist`, so a machine that only runs the tarball does not need Node.js.

## Release package

`release/ai-workload-0.1.0.tar.gz` is the 0.1.0 package for macOS and Linux. After unpacking, `bash install.sh` then `bash start.sh` is enough. The archive contains the backend, the built UI, migrations, and `database/bundle/`. It omits the TypeScript source, tests, and this repository's git history.

Rebuild it from a source checkout with `make build && make release`.
