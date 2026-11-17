import json

import httpx
import pytest
from app.application.planner_service import DATA
from app.engines.model_analyzer import validate_legacy_config
from app.integrations.huggingface import HuggingFaceImporter, SourceUnavailable
from app.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from test_workload_api import create, submit, wait


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 'planner.db'}")) as client:
        yield client


def design_input(client, **overrides):
    body = {
        "model_id": client.get("/api/models").json()[0]["id"],
        "hardware_id": "nvidia-h100-sxm-80gb-20260913",
        "mode": "offline",
        "load": {"prompt_tokens": 5, "output_tokens": 3, "concurrency": 1,
                 "request_count": 1, "request_rate": None, "arrival": None},
        "runtime": {"max_batch_sequences": 1, "prefill_chunk_tokens": 5},
        "resources": {"gpus_per_node": 1, "topology_template": "single_gpu-v1"},
    }
    body.update(overrides)
    return body


def test_seed_golden_parameters(client):
    models = client.get("/api/models").json()
    assert [m["parameters"] for m in models] == [494032768, 1543714304]
    for model in models:
        analysis = client.post(f"/api/models/{model['id']}/analyze").json()
        assert sum(analysis["parameter_breakdown"].values()) == model["parameters"]
        assert analysis["parameter_breakdown"]["output_head"] == 0
        assert len(model["revision"]) == 40
        assert len(model["config_sha256"]) == 64


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_run_snapshot_and_accounting(client, precision):
    spec = design_input(client)
    spec["runtime"].update(weight_dtype=precision, activation_dtype=precision, kv_dtype=precision)
    design = create(client, spec)
    run = wait(client, submit(client, design)["id"])
    assert run["status"] == "completed", run
    payload = client.get(f"/api/workloads/runs/{run['id']}/export").json()
    assert payload["frozen_input"]["design"] == design["revision"]["input"]
    rank = payload["plans"][0]["details"]["ranks"][0]
    assert rank["peak_bytes"] == sum(rank["components"].values())
    assert sum(w["bytes"] for w in rank["weights"]) == 494032768 * (2 if precision == "bf16" else 4)
    phases = {row["phase"] for row in payload["plans"][0]["details"]["layers"]}
    assert phases == {"prefill", "decode"}
    assert payload["frozen_input"]["sources"]["model"]["config_sha256"]


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/workloads"), ("POST", "/api/workloads"),
    ("GET", "/api/workloads/old"), ("POST", "/api/workloads/old/profile"),
    ("GET", "/api/workloads/old/profiles"),
])
def test_legacy_routes_gone(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "LEGACY_WORKLOAD_REMOVED"


@pytest.mark.parametrize(
    "override",
    [
        {"runtime": {"max_batch_sequences": 0}},
        {"runtime": {"max_batch_sequences": 1.5}},
        {"runtime": {"max_batch_sequences": True}},
        {"load": {"prompt_tokens": -1}},
        {"load": {"prompt_tokens": 16777217}},
        {"mode": "training"},
        {"runtime": {"weight_dtype": "int4"}},
        {"runtime": {"activation_checkpoint": True}},
        {"unknown_field": 1},
    ],
)
def test_invalid_workloads(client, override):
    response = client.post("/api/workloads/designs",
                           json={"name": "Invalid", "input": design_input(client, **override)},
                           headers={"Idempotency-Key": "invalid"})
    assert response.status_code == 422
    assert client.get("/api/workloads/designs").json()["items"] == []


def test_missing_and_api_fallback(client):
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/api/models/absent").status_code == 404
    assert client.post("/api/workloads/preview", json={
        "input": design_input(client, model_id="absent"), "client_revision": 1}).status_code == 404
    assert client.get("/api/workloads/runs/absent").status_code == 404
    assert client.post("/api/models/import", json={"repo_id": "https://localhost/test"}).status_code == 422


def test_oom_not_false_fit(client):
    spec = design_input(client)
    spec["load"].update(concurrency=4096, request_count=4096, prompt_tokens=32000)
    spec["runtime"].update(max_batch_sequences=4096)
    run = wait(client, submit(client, create(client, spec))["id"])
    assert run["summary"]["outcome"] == "infeasible", run
    assert client.get(f"/api/workloads/runs/{run['id']}/plans").json()["items"] == []


def test_restart_persistence_and_migrations(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    with TestClient(create_app(url)) as client:
        w = create(client, design_input(client))
        p = wait(client, submit(client, w)["id"])
    app = create_app(url)
    with TestClient(app) as client:
        assert len(client.get("/api/models").json()) == 2
        assert client.get(f"/api/workloads/designs/{w['id']}").json()["revision"] == w["revision"]
        assert client.get(f"/api/workloads/runs/{p['id']}").json() == p
        tables = inspect(app.state.engine).get_table_names()
        assert "alembic_version" in tables and "workload_runs" in tables
        assert "workloads" not in tables and "workload_profiles" not in tables


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "qwen2_moe"},
        {"hidden_size": 897},
        {"num_key_value_heads": 3},
        {"use_sliding_window": True},
        {"auto_map": {"AutoConfig": "evil"}},
        {"quantization_config": {"bits": 4}},
        {"num_local_experts": 8},
        {"rope_scaling": {"factor": 2}},
        {"head_dim": 128},
        {"vocab_size": True},
        {"tie_word_embeddings": "false"},
    ],
)
def test_legacy_workload_unsupported_config(changes):
    config = json.loads((DATA / "qwen-0.5b-config.json").read_text())
    config.update(changes)
    with pytest.raises(ValueError):
        validate_legacy_config(config)


def test_import_fixed_revision_and_idempotence(client):
    raw = (DATA / "qwen-0.5b-config.json").read_bytes()
    sha = "060db6499f32faf8b98477b0a26969ef7d8b9987"
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if "/api/" in request.url.path:
            return httpx.Response(200, json={"sha": sha, "safetensors": {"total": 494032768}})
        return httpx.Response(200, content=raw)

    client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(handler))
    imported = client.post("/api/models/import", json={"repo_id": "Qwen/Qwen2.5-0.5B"})
    assert imported.status_code == 200
    assert imported.json()["revision"] == sha
    assert sha in urls[-1]
    assert len(client.get("/api/models").json()) == 2


@pytest.mark.parametrize("status", [401, 404, 500])
def test_import_source_failure(client, status):
    client.app.state.importer = HuggingFaceImporter(
        httpx.MockTransport(lambda request: httpx.Response(status))
    )
    assert client.post("/api/models/import", json={"repo_id": "Qwen/missing"}).status_code == 502


def test_redirect_and_size_protection():
    importer = HuggingFaceImporter(
        httpx.MockTransport(
            lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        )
    )
    with pytest.raises(SourceUnavailable, match="redirect rejected"):
        importer.import_model("Qwen/test")
    importer = HuggingFaceImporter(
        httpx.MockTransport(lambda request: httpx.Response(200, content=b" " * 2_000_001))
    )
    with pytest.raises(SourceUnavailable, match="2 MB"):
        importer.import_model("Qwen/test")


def test_spa_deep_link_and_assets(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>Planner</html>")
    (dist / "assets" / "app.js").write_text("console.log('planner')")
    with TestClient(create_app("sqlite:///:memory:", static_dir=dist)) as client:
        assert client.get("/workloads/abc").text == "<html>Planner</html>"
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/backend/pyproject.toml").status_code == 404
