import hashlib
import json
from pathlib import Path

import pytest
from app.hardware.repository import DATA, HardwareRepository
from app.hardware.roofline import evaluate
from app.hardware.schemas import HardwareRecord, payload_hash
from app.main import create_app
from fastapi.testclient import TestClient
from test_planner import design_input
from test_workload_api import create, submit, wait

H100 = "nvidia-h100-sxm-80gb-20260913"
A100 = "nvidia-a100-pcie-80gb-20260913"


@pytest.fixture
def client():
    with TestClient(create_app("sqlite:///:memory:")) as client:
        yield client


def test_catalog_filters_versions_and_export(client):
    records = client.get("/api/hardware").json()
    assert len(records) == 11
    assert len(client.get("/api/hardware?include_legacy=true").json()) == 12
    assert len(client.get("/api/hardware?planner_only=true").json()) == 7
    assert len(client.get("/api/hardware?vendor=Huawei").json()) == 2
    assert len(client.get("/api/hardware?q=A100&form_factor=PCIe").json()) == 1
    assert client.get("/api/hardware?q=no-match").json() == []
    for record in client.get("/api/hardware?include_legacy=true").json():
        assert record == client.get(f"/api/hardware/{record['id']}").json()
        export = client.get(f"/api/hardware/{record['id']}/export")
        assert export.json() == record
        assert "attachment" in export.headers["content-disposition"]
        assert payload_hash(record) == record["content_sha256"]
    for url in ("/api/hardware/absent", "/api/hardware/absent/export", "/api/hardware/absent/roofline"):
        assert client.get(url).status_code == 404
    for query in ("vendor=invalid", "form_factor=invalid", "q=" + "a" * 101):
        assert client.get("/api/hardware?" + query).status_code == 422


@pytest.mark.parametrize("identifier,peak,bw", [(H100, 989.5, 3350), (A100, 312, 1935)])
def test_roofline_independent_golden(client, identifier, peak, bw):
    body = client.get(f"/api/hardware/{identifier}/roofline?intensity=2").json()
    assert body["ridge_point"] == pytest.approx(peak * 1000 / bw)
    assert body["operating_point"]["performance_tops"] == pytest.approx(2 * bw / 1000)
    assert body["bottleneck"] == "memory"
    result = client.get(f"/api/hardware/{identifier}/roofline?intensity=10000&sparse=true").json()
    assert result["operating_point"]["performance_tops"] == peak * 2
    assert result["bottleneck"] == "compute"
    assert all(p["performance_tops"] <= peak * 2 for p in result["curve"])


def test_integer_precision_and_blackwell_sparse_ratio(client):
    value = client.get(f"/api/hardware/{A100}/roofline?precision=int8").json()
    assert value["unit"] == "TOPS" and value["peak_tops"] == 624
    identifier = "nvidia-b300-sxm-270gb-20260913"
    dense = client.get(f"/api/hardware/{identifier}/roofline?precision=fp4").json()
    sparse = client.get(f"/api/hardware/{identifier}/roofline?precision=fp4&sparse=true").json()
    assert dense["peak_tops"] == 13500
    assert sparse["peak_tops"] == 18000
    assert sparse["peak_tops"] / dense["peak_tops"] == pytest.approx(4 / 3)
    assert client.get(f"/api/hardware/{identifier}/roofline?precision=int8").status_code == 422


@pytest.mark.parametrize("query", [
    "precision=fp64", "intensity=0", "intensity=-1", "intensity=NaN", "intensity=Infinity",
    "intensity=1e10", "intensity=abc", "precision=fp32&sparse=true",
])
def test_bad_roofline_inputs(client, query):
    assert client.get(f"/api/hardware/{H100}/roofline?{query}").status_code == 422


@pytest.mark.parametrize("identifier,precision", [
    (A100, "fp8"), ("nvidia-h20-sxm-96gb-20260913", "bf16"),
    ("nvidia-l20-pcie-48gb-20260913", "bf16"),
    ("huawei-910b2-module-64gb-20260913", "bf16"),
    ("huawei-910c-module-128gb-20260913", "fp16"),
])
def test_unknown_and_unverified_fail_closed(client, identifier, precision):
    assert client.get(f"/api/hardware/{identifier}/roofline?precision={precision}").status_code == 422


def test_workload_guard_and_snapshot(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 'hardware.db'}")) as client:
        model = client.get("/api/models").json()[0]
        spec = design_input(client)
        run = wait(client, submit(client, create(client, spec))["id"])
        frozen = client.get(f"/api/workloads/runs/{run['id']}/export").json()["frozen_input"]
        assert frozen["sources"]["hardware"]["content_sha256"] == client.get(f"/api/hardware/{H100}").json()["content_sha256"]
        spec["hardware_id"] = "nvidia-h20-sxm-96gb-20260913"
        preview = client.post("/api/workloads/preview", json={"input": spec, "client_revision": 0}).json()
        assert preview["capacity_status"] == "fail"
        assert preview["diagnostics"][0]["code"] == "PRECISION_UNVERIFIED"
        blocked = create(client, spec)
        response = client.post("/api/workloads/runs",
                               json={"design_revision_id": blocked["revision"]["id"]},
                               headers={"Idempotency-Key": "unverified-hardware"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "PRECISION_UNVERIFIED"
        structure = client.post(f"/api/models/{model['id']}/structure").json()
        url = f"/api/models/{model['id']}/structures/{structure['id']}/cost"
        assert client.post(url, json={"hardware_id": spec["hardware_id"]}).status_code == 422
        response = client.post(url, json={"hardware_id": H100})
        assert response.status_code == 200
        assert response.json()["input_snapshot"]["hardware"]["content_sha256"]


def test_legacy_persistence_and_conflict(client):
    identifier = "nvidia-h100-sxm-80gb-v1"
    with client.app.state.sessions() as session:
        repo = HardwareRepository(session)
        before = repo.calculation_input(identifier).model_dump()
        repo.seed()
        repo.seed()
        assert repo.calculation_input(identifier).model_dump() == before
        assert "schema_version" not in before
        from app.infrastructure.database import HardwareRow
        row = session.get(HardwareRow, H100)
        row.payload = {**row.payload, "name": "conflicting update"}
        session.commit()
        with pytest.raises(RuntimeError, match="Immutable"):
            repo.seed()


def test_packaged_data_sources_hashes_and_engine_boundary():
    archive = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-hardware-center/evidence/sources"
    for path in DATA.glob("*.json"):
        data = json.loads(path.read_text())
        record = HardwareRecord.model_validate(data)
        assert payload_hash(data) == record.content_sha256
        for source in record.sources:
            assert hashlib.sha256((archive / source.artifact).read_bytes()).hexdigest() == source.sha256
        if record.planner_precisions:
            with pytest.raises(ValueError):
                evaluate(record, "bf16", False, float("nan"))
    data["compute"]["bf16"]["dense"] = -1
    with pytest.raises(ValueError):
        HardwareRecord.model_validate(data)
