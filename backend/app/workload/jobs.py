"""One local coordinator; CPU execution is isolated in a spawned process."""
import logging
import multiprocessing
import threading
import time
from uuid import uuid4

from app.engines.inference.diagnostics import event_memory, representative_layers
from app.engines.inference.explain import memory_explanation, summary_explanation
from app.engines.inference.ir import Candidate
from app.engines.inference.memory import admission_memory
from app.engines.inference.metrics import sensitivity
from app.engines.inference.normalize import normalize
from app.engines.inference.scheduler import simulate
from app.engines.inference.search import search

from .domain import FrozenSources, canonical
from .repository import Repository
from .schemas import DesignInput, Search

logger = logging.getLogger(__name__)


def restore(frozen):
    source = frozen["sources"]
    return FrozenSources(**source, ir=normalize(source["model"], source["structure"]))


def execute(frozen, cancel, progress=None):
    sources = restore(frozen)
    design = DesignInput.model_validate(frozen["design"])
    options = Search.model_validate(frozen["search"])
    result = search(sources, design, options, cancelled=cancel, progress=progress)
    # Diagnostics have their own documented bounded replay, never change ranking.
    deadline = time.monotonic() + 20
    for plan in result["plans"]:
        if cancel():
            result["status"] = "cancelled"
            break
        candidate = Candidate(**{**plan["candidate"], "boundaries": tuple(plan["candidate"]["boundaries"])})
        memory = admission_memory(sources.ir, candidate, design, sources.hardware, sources.assumptions)
        plan["explanation"] = summary_explanation(sources, candidate, design, plan)
        for rank in memory["ranks"]:
            rank["explanation"] = memory_explanation(sources, candidate, design, rank)
        memory["explanation"] = memory["ranks"][memory["worst_rank"]]["explanation"]
        plan["memory"] = {k: v for k, v in memory.items() if k != "ranks"}
        simulation = simulate(sources, candidate, design, min(options.event_budget, 20000), details=True,
                              cancelled=lambda: cancel() or time.monotonic() >= deadline)
        events = event_memory(sources, candidate, design, simulation)
        operators, communication = representative_layers(sources, candidate, design, simulation)
        by_layer = {}
        for operator in operators:
            key = (operator["rank"], operator["layer_index"], operator["phase"])
            by_layer.setdefault(key, []).append(operator)
        layer_rows = []
        for rank in memory["ranks"]:
            for layer in rank.pop("layers"):
                source = sources.ir.layers[layer["layer_index"]]
                for phase in ("prefill", "decode"):
                    ops = by_layer.get((rank["rank"], source.index, phase), [])
                    layer_rows.append({
                        **layer, "rank": rank["rank"], "stage": rank["p"], "phase": phase,
                        "template_id": source.template_id,
                        "weight_storage_refs": [w.storage_id for w in source.weights],
                        "local_shapes": [w for w in rank["weights"] if w["storage_id"] in {
                            weight.storage_id for weight in source.weights}],
                        "operators": ops, "flops": sum(op["flops"] for op in ops) if ops else None,
                        "hbm_traffic_bytes": sum(op["hbm_bytes"] for op in ops) if ops else None,
                        "scope": "representative_wave", "timing_available": bool(ops)})
            rank["event_memory"] = next(row for row in events["ranks"] if row["rank"] == rank["rank"])
        plan["event_memory"] = {k: v for k, v in events.items() if k != "ranks"}
        plan["details"] = {"layers": layer_rows, "ranks": memory["ranks"],
                           "groups": plan["groups"], "communication": communication,
                           "events": simulation["waves"]}
        plan["diagnostic_replay"] = {"status": simulation["status"], "trace_complete": simulation["trace_complete"],
                                     "events_processed": simulation["events_processed"]}
        plan["sensitivity"] = sensitivity(sources, candidate, design, deadline, cancel)
    # Bound persisted output independently of event count / parameter limits.
    if len(canonical(result).encode()) > 64 * 1024 * 1024:
        for plan in result["plans"]:
            plan["details"].pop("events", None)
            plan["details"]["layers"] = plan["details"]["layers"][:1000]
            plan["diagnostic_replay"]["output_truncated"] = True
        if len(canonical(result).encode()) > 64 * 1024 * 1024:
            raise ValueError("Result exceeds 64 MiB output budget")
    return result


def child(frozen, connection, cancel):
    last = 0.0

    def progress(value):
        nonlocal last
        if time.monotonic() - last >= 1:
            connection.send(("progress", value))
            last = time.monotonic()
    try:
        connection.send(("result", execute(frozen, cancel.is_set, progress)))
    except Exception as exc:
        logger.exception("Workload child failed")
        connection.send(("error", {"code": "ENGINE_FAILURE", "message": str(exc)[:2000]}))
    finally:
        connection.close()


class Worker:
    def __init__(self, sessions):
        self.sessions = sessions
        self.owner = uuid4().hex
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="workload-coordinator", daemon=True)
        self.last_error = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)

    def loop(self):
        while not self.stop_event.is_set():
            try:
                with self.sessions() as session:
                    row = Repository(session).claim(self.owner)
                    if row:
                        run_id, frozen = row.id, row.frozen_input
                if row:
                    self.compute(run_id, frozen)
                else:
                    self.stop_event.wait(0.25)
            except Exception as exc:
                logger.exception("Workload coordinator failed")
                self.last_error = str(exc)[:1000]
                self.stop_event.wait(1)

    def compute(self, run_id, frozen):
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        cancel = context.Event()
        process = context.Process(target=child, args=(frozen, sender, cancel), daemon=True)
        process.start()
        sender.close()
        progress, result, error = {}, None, None
        began, heartbeat, cancelled_at = time.monotonic(), 0, None
        try:
            while process.is_alive() or receiver.poll():
                now = time.monotonic()
                if self.stop_event.is_set():
                    cancel.set()
                if now - heartbeat >= 1:
                    with self.sessions() as session:
                        if not Repository(session).heartbeat(run_id, self.owner, progress):
                            cancel.set()
                    heartbeat = now
                if cancel.is_set():
                    cancelled_at = cancelled_at or now
                    if now - cancelled_at > 2:
                        process.terminate()
                        break
                if now - began > frozen["search"]["max_seconds"] + 30:
                    error = {"code": "WORKER_TIMEOUT", "message": "Search and diagnostic wall-time limit exceeded"}
                    process.terminate()
                    break
                if receiver.poll(0.1):
                    try:
                        kind, value = receiver.recv()
                    except EOFError:
                        break
                    if kind == "progress":
                        progress = value
                    elif kind == "result":
                        result = value
                    else:
                        error = value
            process.join(timeout=1)
            with self.sessions() as session:
                Repository(session).finish(run_id, self.owner, result, error,
                                           interrupted=self.stop_event.is_set())
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            receiver.close()
