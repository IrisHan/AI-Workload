"""Start/stop only this project's local preview, with HTTP readiness checks."""

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data" / "service.json"
LOG = ROOT / "data" / "server.log"


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def owned(state):
    try:
        command = subprocess.check_output(
            ["ps", "-p", str(state["pid"]), "-o", "command="], text=True
        )
    except (subprocess.CalledProcessError, KeyError):
        return False
    if "app.main:app" not in command:
        return False
    # venv/bin/python is a symlink. Homebrew's framework binary replaces argv0,
    # so ps shows Python.app rather than the venv path.
    venv_python = str(ROOT / ".venv" / "bin" / "python")
    app_dir = f"--app-dir {ROOT / 'backend'}"
    return venv_python in command or app_dir in command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    parser.add_argument("--port", type=int, default=8666)
    args = parser.parse_args()
    ROOT.joinpath("data").mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if args.action == "stop":
        if state and owned(state):
            os.kill(state["pid"], signal.SIGTERM)
            for _ in range(30):
                if not owned(state):
                    break
                time.sleep(0.2)
            else:
                raise SystemExit("Process has not stopped; inspect data/server.log")
            STATE.unlink(missing_ok=True)
            print("Planner stopped.")
        else:
            print("No owned Planner process to stop.")
        return
    if args.action == "status":
        good = bool(state and owned(state) and healthy(state["port"]))
        print(json.dumps({"healthy": good, **state}))
        raise SystemExit(0 if good else 1)
    if state and owned(state):
        if healthy(state["port"]):
            print(state["url"])
            return
        raise SystemExit(
            "Owned process is unhealthy. Inspect data/server.log, then stop and start."
        )
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("Build frontend first: npm --prefix frontend run build")
    port = args.port
    for candidate in range(port, port + 20):
        with socket.socket() as sock:
            try:
                # Match Uvicorn's reuse semantics without overlooking a live listener.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", candidate))
                sock.listen(1)
                port = candidate
                break
            except OSError:
                continue
    else:
        raise SystemExit("No free port in requested range")
    with LOG.open("ab") as output:
        process = subprocess.Popen(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                str(ROOT / "backend"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {"pid": process.pid, "port": port, "url": f"http://127.0.0.1:{port}"}
    STATE.write_text(json.dumps(state, indent=2))
    for _ in range(40):
        if process.poll() is not None:
            raise SystemExit(f"Planner exited. Inspect {LOG}")
        if healthy(port):
            with urlopen(state["url"], timeout=3) as response:
                if b'<div id="root">' not in response.read():
                    raise SystemExit("SPA readiness failed")
            print(state["url"])
            return
        time.sleep(0.25)
    process.terminate()
    process.wait(timeout=10)
    raise SystemExit("HTTP readiness failed. Inspect data/server.log")


if __name__ == "__main__":
    main()
