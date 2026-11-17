"""P1 integration: immutable upgrades, profile conservation, cross-API residency."""
import copy
import json
from pathlib import Path

import httpx
import pytest
from app.infrastructure.database import ModelRow
from app.integrations.huggingface import HuggingFaceImporter
from app.main import create_app
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2] / "docs/reports"
GPU = "nvidia-h100-sxm-80gb-20260913"


def import_model(client, group, repo):
    folder = ROOT / group / "evidence/sources" / repo
    raw = (folder / "config.json").read_bytes()
    meta = json.loads((folder / "metadata.json").read_text())
    client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(
        lambda request: httpx.Response(200, json={"sha": meta["sha"]})
        if "/api/" in request.url.path else httpx.Response(200, content=raw)))
    response = client.post("/api/models/import", json={"repo_id": meta["id"]})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_morphology_cost_conservation_and_runtime_guard(phase):
    with TestClient(create_app("sqlite:///:memory:")) as client:
        m = import_model(client, "2026-09-15-kimi-family", "moonshotai--Kimi-K2-Instruct")
        snapshot = client.post(f"/api/models/{m['id']}/structure").json()
        r = client.post(f"/api/models/{m['id']}/structures/{snapshot['id']}/cost",
                        json={"hardware_id": GPU, "phase": phase, "sequence_length": 128})
        assert r.status_code == 200, r.text
        p = r.json()
        assert sum(row["flops"] for row in p["rows"]) == pytest.approx(p["flops"], rel=1e-14)
        caps = client.get("/api/workloads/capabilities", params={"model_id": m["id"]}).json()
        assert caps["supported"] is False
        assert any(issue["task"] == "W13a" for issue in caps["issues"])


def test_old_model_capability_refresh_does_not_rewrite_payload():
    with TestClient(create_app("sqlite:///:memory:")) as client:
        m = import_model(client, "2026-09-14-qwen-family", "Qwen--Qwen3-0.6B")
        factory = client.app.state.sessions
        with factory() as session:
            row = session.get(ModelRow, m["id"])
            old = copy.deepcopy(row.payload)
            old.update(workload_supported=False, support_notes=["Old engine"], capability_version=None)
            row.payload = old
            session.commit()
        fresh = client.get(f"/api/models/{m['id']}").json()
        assert fresh["workload_supported"] is True
        assert fresh["capability_version"] == "model-capabilities-1.1.0"
        assert fresh["raw_config"] == old["raw_config"]
        assert fresh["parameters"] == old["parameters"]
        with factory() as session:
            assert session.get(ModelRow, m["id"]).payload == old
        assert next(x for x in client.get("/api/models").json() if x["id"] == m["id"]) == fresh


def test_k3_cost_memory_and_old_snapshot_rejection():
    with TestClient(create_app("sqlite:///:memory:")) as client:
        m = import_model(client, "2026-09-15-kimi-family", "moonshotai--Kimi-K3")
        prefix = f"/api/models/{m['id']}"
        snapshot = client.post(prefix + "/structure").json()
        for dtype in ("bf16", "fp32"):
            memory = client.post(prefix + "/memory", json=dict(batch_size=2, context_length=128, dtype=dtype))
            cost = client.post(prefix + f"/structures/{snapshot['id']}/cost", json=dict(
                hardware_id=GPU, phase="prefill", batch_size=2, sequence_length=128, dtype=dtype))
            assert memory.status_code == cost.status_code == 200
            assert cost.json()["kv_cache_bytes"] == memory.json()["kv_cache_bytes"]
            assert cost.json()["memory_fits"] is False
        for overrides in ({"batch_size": 0}, {"sequence_length": 1048577}, {"dtype": "fp8"}):
            bad = client.post(prefix + f"/structures/{snapshot['id']}/cost", json={
                "hardware_id": GPU, "phase": "prefill", **overrides})
            assert bad.status_code == 422
        assert client.get(prefix + f"/structures/{snapshot['id']}").json() == snapshot
