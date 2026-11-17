"""Isolated DB/API/worker acceptance; never writes the active Planner database."""
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from app.application.planner_service import seed
from app.hardware.repository import HardwareRepository
from app.infrastructure.database import StructureRow, database, migrate
from app.repositories.catalog import CatalogRepository
from app.workload.jobs import Worker
from app.workload.repository import Repository, StoreError, timestamp
from app.workload.router import router
from app.workload.schemas import DesignInput
from app.workload.service import Service
from app.workload.tables import PlanRow, RunRow
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select

PREFIX = "/api/workloads"


@pytest.fixture
def api(tmp_path):
    engine, sessions = database(f"sqlite:///{tmp_path}/workload.db")
    migrate(engine)
    with sessions() as session:
        seed(CatalogRepository(session))
        HardwareRepository(session).seed()
        model = CatalogRepository(session).models()[0]
    worker = Worker(sessions)

    @asynccontextmanager
    async def lifespan(app):
        worker.start()
        yield
        worker.stop()
        engine.dispose()

    app = FastAPI(lifespan=lifespan)
    app.state.sessions, app.state.workload_worker = sessions, worker
    app.include_router(router)
    spec = DesignInput.model_validate({
        "model_id": model.id, "hardware_id": "nvidia-h100-sxm-80gb-20260913",
        "mode": "offline", "load": {"prompt_tokens": 5, "output_tokens": 3, "concurrency": 1,
                                  "request_count": 1, "request_rate": None, "arrival": None},
        "runtime": {"max_batch_sequences": 1, "prefill_chunk_tokens": 5},
        "resources": {"gpus_per_node": 1, "topology_template": "single_gpu-v1"}}).model_dump()
    with TestClient(app) as client:
        yield client, sessions, spec, worker


def create(client, spec, key=None):
    response = client.post(PREFIX + "/designs", json={"name": "Isolated acceptance", "input": spec},
                           headers={"Idempotency-Key": key or uuid4().hex})
    assert response.status_code in (200, 201), response.text
    return response.json()


def submit(client, design, key=None):
    response = client.post(PREFIX + "/runs", json={"design_revision_id": design["revision"]["id"],
                           "search": {"max_candidates": 10, "max_seconds": 5.0, "event_budget": 5000}},
                           headers={"Idempotency-Key": key or uuid4().hex})
    assert response.status_code == 202, response.text
    return response.json()


def wait(client, run_id):
    until = time.monotonic() + 15
    while time.monotonic() < until:
        result = client.get(PREFIX + "/runs/" + run_id).json()
        if result["status"] not in ("running", "queued", "cancel_requested"):
            return result
        time.sleep(.05)
    raise AssertionError("Worker did not finish within 15 seconds")


def test_design_idempotency_revision_cas_and_pagination(api):
    client, sessions, spec, _ = api
    first = create(client, spec, "design-key")
    assert create(client, spec, "design-key") == first
    conflict = client.post(PREFIX + "/designs", json={"name": "Different", "input": spec},
                           headers={"Idempotency-Key": "design-key"})
    assert conflict.status_code == 409
    service = Service(sessions)
    from app.workload.schemas import SaveRevision
    def save():
        try:
            return service.save(first["id"], SaveRevision.model_validate({"input": spec}), first["etag"])
        except StoreError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: save(), range(2)))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert "REVISION_CONFLICT" in results
    assert client.get(f'{PREFIX}/designs/{first["id"]}/revisions/1').json()["input"] == spec
    create(client, spec)
    page = client.get(PREFIX + "/designs?limit=1").json()
    second = client.get(PREFIX + "/designs", params={"limit": 1, "cursor": page["next_cursor"]}).json()
    assert page["items"][0]["id"] != second["items"][0]["id"]
    assert second["next_cursor"] is None


def test_spawned_worker_full_chain_export_and_compare(api):
    client, sessions, spec, _ = api
    design = create(client, spec)
    queued = submit(client, design, "run-key")
    repeated = submit(client, design, "run-key")
    assert repeated["id"] == queued["id"]
    run = wait(client, queued["id"])
    assert run["status"] == "completed", run
    assert run["summary"]["outcome"] == "feasible"
    plans = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()
    assert len(plans["items"]) == 1
    plan = plans["items"][0]
    base = f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}'
    ranks = client.get(base + "/ranks").json()
    assert ranks["result_hash"] == plan["result_hash"]
    rank = ranks["items"][0]
    assert rank["peak_bytes"] == sum(rank["components"].values())
    event = rank["event_memory"]
    assert event["peak_bytes"] == sum(event["component_at_peak"].values())
    assert event["peak_bytes"] <= rank["peak_bytes"]
    assert client.get(base + "/layers?limit=1").json()["next_offset"] == 1
    graph = client.get(base + "/graph?view=logical&layer=0").json()
    assert len(graph["nodes"]) > 1 and graph["result_hash"] == plan["result_hash"]
    export = client.get(f'{PREFIX}/runs/{run["id"]}/export')
    assert export.status_code == 200
    payload = export.json()
    assert payload["frozen_input"]["design"] == spec
    assert payload["plans"][0]["details"]["layers"]
    comparison = client.post(PREFIX + "/compare", json={
        "plan_refs": [{"run_id": run["id"], "plan_id": plan["id"]}] * 2})
    assert comparison.status_code == 200, comparison.text
    assert comparison.json()["input_differences"] == []
    assert client.get(f'{PREFIX}/runs/wrong/plans/{plan["id"]}').status_code == 404
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PlanRow)) == 1
    assert client.delete(f'{PREFIX}/designs/{design["id"]}',
                         headers={"If-Match": design["etag"]}).status_code == 204
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(PlanRow)) == 0
        assert session.scalar(select(func.count()).select_from(RunRow)) == 0


@pytest.mark.parametrize("change", [
    {"unexpected": True}, {"resources": {"node_count": True}}, {"resources": {"node_count": 1.5}},
    {"resources": {"node_count": 256, "gpus_per_node": 256}},
    {"runtime": {"weight_dtype": "fp8"}},
    {"overrides": [{"path": "model.hidden_size", "value": 1, "reason": "invalid"}]},
])
def test_strict_boundary_errors(api, change):
    client, _, spec, _ = api
    result = client.post(PREFIX + "/designs", json={"name": "Invalid", "input": {**spec, **change}},
                         headers={"Idempotency-Key": uuid4().hex})
    assert result.status_code == 422
    assert result.json()["error"]["code"] == "INVALID_CONSTRAINT"


def test_preview_readonly_and_not_green(api):
    client, sessions, spec, _ = api
    with sessions() as session:
        before = session.scalar(select(func.count()).select_from(StructureRow))
    result = client.post(PREFIX + "/preview", json={"input": spec, "client_revision": 12})
    assert result.status_code == 200, result.text
    assert result.json()["capacity_status"] == "unknown"
    assert result.json()["client_revision"] == 12
    assert result.json()["elapsed_ms"] < 500
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(StructureRow)) == before


def test_cancel_retry_and_expired_lease(api):
    client, sessions, spec, worker = api
    worker.stop()
    design = create(client, spec)
    from app.workload.schemas import SubmitRun
    service = Service(sessions)
    body = SubmitRun(design_revision_id=design["revision"]["id"])
    queued = service.submit(body, "first")
    with sessions() as session:
        repo = Repository(session)
        claimed = repo.claim("lost-worker")
        assert claimed.id == queued["id"]
    with sessions() as session:
        row = session.get(RunRow, queued["id"])
        row.lease_expires_at = timestamp(-1)
        session.commit()
    with sessions() as session:
        assert Repository(session).claim("replacement") is None
    interrupted = service.run(queued["id"])
    assert interrupted["status"] == "interrupted"
    retried = service.retry(queued["id"], "retry")
    assert retried["input_hash"] == queued["input_hash"]
    assert service.cancel(retried["id"])["status"] == "cancelled"
    assert service.cancel(retried["id"])["status"] == "cancelled"
    assert service.run(queued["id"])["status"] == "interrupted"


def test_queue_limit_and_cancel_beats_publication(api):
    client, sessions, spec, worker = api
    worker.stop()
    design = create(client, spec)
    service = Service(sessions)
    from app.workload.schemas import SubmitRun
    body = SubmitRun(design_revision_id=design["revision"]["id"])
    runs = [service.submit(body, str(i)) for i in range(4)]
    with pytest.raises(StoreError, match="four"):
        service.submit(body, "fifth")
    with pytest.raises(StoreError, match="active"):
        service.delete(design["id"], design["etag"])
    with sessions() as session:
        row = Repository(session).claim("owner")
        identifier = row.id
    assert service.cancel(identifier)["status"] == "cancel_requested"
    with sessions() as session:
        Repository(session).finish(identifier, "owner", {"status": "completed", "plans": [], "outcome": "infeasible"})
    assert service.run(identifier)["status"] == "cancelled"
    assert len(runs) == 4
