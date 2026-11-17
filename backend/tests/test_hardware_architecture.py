import hashlib

import pytest
from app.hardware.architecture.repository import content_hash, load_bundle
from app.infrastructure.database import ROOT
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with TestClient(create_app("sqlite:///:memory:")) as client:
        yield client


def get(client, sku):
    response = client.get(f"/api/hardware/{sku}/architecture")
    assert response.status_code == 200, response.text
    return response.json()


def test_all_skus_export_and_frozen_input(client):
    records = client.get("/api/hardware?include_legacy=true").json()
    for record in records:
        doc = get(client, record["id"])
        assert [s["number"] for s in doc["sections"]] == list(range(1, 9))
        assert doc["hardware_sha256"] == record["content_sha256"]
        assert content_hash(doc) == doc["content_sha256"]
        assert client.get(f"/api/hardware/{record['id']}").json() == record
        exported = client.get(f"/api/hardware/{record['id']}/architecture/export")
        assert exported.json() == doc
        assert "attachment" in exported.headers["content-disposition"]
    assert client.get("/api/hardware/missing/architecture").status_code == 404
    assert client.get("/api/hardware/missing/architecture/export").status_code == 404


def test_shared_hopper_template_and_sku_core_counts(client):
    sxm = get(client, "nvidia-h100-sxm-80gb-20260913")
    pcie = get(client, "nvidia-h100-pcie-80gb-20260913")
    h20 = get(client, "nvidia-h20-sxm-96gb-20260913")
    assert sxm["architecture"] == pcie["architecture"] == h20["architecture"]
    def facts(d):
        return {f["key"]: f["value"] for f in d["headline_metrics"]}
    assert facts(sxm)["cuda_count"] == "16,896"
    assert facts(pcie)["cuda_count"] == "14,592"
    assert facts(sxm)["tensor_count"] == "528"
    assert facts(pcie)["tensor_count"] == "456"
    assert facts(sxm)["rt_count"] == "0 / 不支持"
    assert "user-reference" in h20["headline_metrics"][0]["source_ids"]


def test_memory_scopes_and_blackwell_corrections(client):
    b200 = get(client, "nvidia-b200-sxm-180gb-20260913")
    b300 = get(client, "nvidia-b300-sxm-270gb-20260913")
    memory = {f["key"]: f for f in b200["memory_hierarchy"]}
    assert memory["register"]["value"] == "256 KiB"
    assert memory["tmem"]["value"] == "256 KiB"
    assert memory["l1_pool"]["value"] == "256 KiB"
    assert memory["shared_max"]["value"] == "228 KiB"
    assert memory["device_memory"]["value"] == "180 GB"
    assert memory["device_memory"]["status"] == "catalog"
    assert b200["architecture"]["id"] != b300["architecture"]["id"]
    assert b200["headline_metrics"][0]["value"] == "148"
    assert b300["headline_metrics"][0]["status"] == "reference"
    assert "128KB" in " ".join(b200["sections"][2]["notes"])
    assert "30 PFLOPS" in b300["sections"][3]["reference_text"]
    assert "13.5/18PF" in " ".join(b300["sections"][3]["notes"])


def test_ada_shared_not_hopper_and_missing_npu(client):
    l20 = get(client, "nvidia-l20-pcie-48gb-20260913")
    l40 = get(client, "nvidia-l40-pcie-48gb-20260913")
    assert l20["architecture"] == l40["architecture"]
    memory = {f["key"]: f["value"] for f in l20["memory_hierarchy"]}
    assert memory["shared_max"] == "100 KiB" and memory["l1_pool"] == "128 KiB"
    assert memory["tmem"] == "无"
    npu = get(client, "huawei-910c-module-128gb-20260913")
    assert npu["architecture"]["specs"] == []
    assert len(npu["memory_hierarchy"]) == 1
    assert npu["variant"]["source_family"] is None


def test_archived_reference_complete_and_verified_source_hashes():
    bundle = load_bundle()
    raw = (ROOT / "docs/reports/2026-09-14-hardware-architecture/evidence/six-gpu-reference.md").read_text()
    assert bundle["source_document"] == raw
    for family, sections in bundle["references"].items():
        assert len(sections) >= 7, family
        assert all(s["text"] and s["blocks"] for s in sections.values())
    assert len(bundle["references"]) == 6
    assert len(bundle["variants"]) == 11
    for source in bundle["sources"]:
        assert hashlib.sha256((ROOT / source["artifact"]).read_bytes()).hexdigest() == source["sha256"]
    for template in bundle["templates"]:
        assert content_hash(template) == template["content_sha256"]
