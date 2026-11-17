# AI Workload

AI Workload is a local planning workbench for inference deployment. It combines a model catalog, a GPU catalog, and an inference designer that estimates memory, parallelism, and prefill/decode performance.

The project is still in development. Version 1.0 is planned for 2027. This repository publishes the current 0.1.0 line so it can be run, reviewed, and iterated in the open.

The application has no login. It listens only on `127.0.0.1`.

![Uploading 截屏2026-11-18 01.16.22.png…]()## Choose how to run it

| Goal | Start here |
|---|---|
| Use the app without building the UI | [Release package 0.1.0](release/ai-workload-0.1.0.tar.gz) |
| Change the source | Sections below |

Windows and Docker are not supported. The supported hosts are macOS and Linux.

## Release package

The file `release/ai-workload-0.1.0.tar.gz` is a runnable tree. It contains the Python application, the already-built web UI, the database schema, and the initial data. It does not ask you to compile the frontend.

```bash
tar -xzf ai-workload-0.1.0.tar.gz
cd ai-workload-0.1.0
bash install.sh
bash start.sh
```

Then open **http://127.0.0.1:8666**. API reference: **http://127.0.0.1:8666/docs**.

`install.sh` needs Python 3.11 or newer and network access to install Python packages. It does not need Node.js.

## Develop from source

Requirements:

- Python 3.11 or newer
- Node.js 22 or newer and npm, only when you build or change the UI
- A CPython build whose `pyexpat` and `sqlite3` modules import cleanly

```bash
make setup
make init-db
make start
make status
```

`make setup` creates `.venv`, installs the locked Python dependencies, installs the backend in editable mode, and builds the UI. `make init-db` creates `data/planner.db` from `database/bundle/` when that file does not already exist. Startup then opens the same file.

Stop the preview with `make stop`.

Day-to-day UI development uses two processes. Stop the preview first, because it does not reload code:

```bash
make stop
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8666 --reload
npm --prefix frontend run dev
```

The Vite server is **http://127.0.0.1:5173**. Its `/api`, `/docs`, and `/openapi.json` requests are proxied to port 8666.

## Where the service listens

There is no configuration file for the listen address. Defaults are in the start command.

| Item | Default | How to change it |
|---|---|---|
| Preview URL | http://127.0.0.1:8666 | `make start PORT=9000` or `PORT=9000 bash start.sh` |
| Bind address | `127.0.0.1` | Fixed in `scripts/service.py`. The process is not exposed beyond the machine. |
| API docs | http://127.0.0.1:8666/docs | Same port as the preview |
| Vite dev URL | http://127.0.0.1:5173 | `frontend/vite.config.ts` (`server.port`) |
| Vite API proxy | http://127.0.0.1:8666 | `frontend/vite.config.ts` (`server.proxy`) |
| SQLite file | `data/planner.db` | `DATABASE_URL=sqlite:////absolute/path/planner.db` |
| Health check | `GET /api/health` | `make status` |

If 8666 is busy, `start.sh` tries the next 20 ports and prints the URL it actually bound. `make status` prints that URL again. The chosen port is recorded in `data/service.json`, which is local runtime state and is not committed.

An example of the environment variables is in `deployment/env.example`. Copy values into your shell. Do not commit a file that contains a real database password. The SQLite path does not need one.

## Database

The local database is SQLite. The app creates the schema with Alembic on startup (`backend/migrations`, revision `0003`) and then loads idempotent model and hardware seeds.

`make init-db` is the way to load the published snapshot before the first start. The snapshot lives in `database/bundle/` and includes the schema plus the catalog, structure snapshots, and workload history exported for this 0.1.0 line. The script refuses to replace an existing `data/planner.db`.

Check the schema version:

```bash
.venv/bin/alembic -c backend/alembic.ini current
```

PostgreSQL and Docker Compose are not part of this release. Workload execution locks are implemented for SQLite only.

## Python environment

`make setup` is the supported installer. It uses `python3` unless you set `PYTHON` to a 3.11+ interpreter:

```bash
PYTHON=python3.11 make setup
```

Pinned packages are in `backend/requirements.lock`. A slow package index can be replaced for that one command:

```bash
PIP_INDEX_URL=https://pypi.org/simple PYTHON=python3.11 make setup
```

No extra system library is required for SQLite. On macOS, a Homebrew Python 3.11 build can fail while creating the virtual environment if `pyexpat` cannot load the system `libexpat`. `make setup` stops with the import error. The usual repair is to install Homebrew `expat` 2.7 or newer and point `pyexpat.so` at that library, then run `make setup` again.

## Layout

| Path | Role |
|---|---|
| `backend/` | FastAPI application, engines, Alembic migrations |
| `frontend/` | React UI source. The preview serves `frontend/dist` |
| `database/bundle/` | Schema and initial rows used by `make init-db` |
| `scripts/` | Setup, database restore, process control, release build |
| `release/ai-workload-0.1.0.tar.gz` | Prebuilt 0.1.0 package for macOS and Linux |
| `data/` | Local SQLite file, log, and pid. Not committed |
| `docs/OPERATIONS.md` | Short operator notes for ports, database, and scripts |

Rebuild the release archive after a UI or backend change:

```bash
make build
make release
```

More detail is in [docs/OPERATIONS.md](docs/OPERATIONS.md).
