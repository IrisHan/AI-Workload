import json
from pathlib import Path

import httpx
import pytest
from app.integrations.huggingface import HuggingFaceImporter
from app.main import create_app
from fastapi.testclient import TestClient

SOURCES = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-13-model-center/evidence/sources"
RECORDS = json.loads((SOURCES / "manifest.json").read_text())


@pytest.mark.parametrize("record", RECORDS, ids=[r["repo_id"].split("/")[-1] for r in RECORDS])
def test_import_analysis_structure_memory_api(record):
    folder = SOURCES / record["repo_id"].replace("/", "--")
    raw = (folder / "config.json").read_bytes()
    requested = []

    def handler(request):
        requested.append(str(request.url))
        if "/api/" in request.url.path:
            return httpx.Response(200, json={"sha": record["revision"], "safetensors": record["safetensors"]})
        assert record["revision"] in str(request.url)
        assert request.url.path.endswith("/config.json")
        return httpx.Response(200, content=raw)

    with TestClient(create_app("sqlite:///:memory:")) as client:
        client.app.state.importer = HuggingFaceImporter(httpx.MockTransport(handler))
        response = client.post("/api/models/import", json={"repo_id": "https://huggingface.co/" + record["repo_id"]})
        assert response.status_code == 200, response.text
        model = response.json()
        prefix = f"/api/models/{model['id']}"
        assert model["raw_config"] == json.loads(raw)
        assert model["revision"] == record["revision"]
        assert len(requested) == 2
        analysis = client.post(prefix + "/analyze")
        assert analysis.status_code == 200, analysis.text
        assert analysis.json()["parameters"] == model["parameters"]
        structure = client.post(prefix + "/structure")
        assert structure.status_code == 200, structure.text
        assert structure.json()["comparison"]["matched"]
        snapshot = structure.json()
        assert snapshot["input_snapshot"] == model
        assert client.post(prefix + "/structure").json() == snapshot
        memory = client.post(prefix + "/memory", json={"context_length": 128})
        assert memory.status_code == 200, memory.text
        assert memory.json()["weight_bytes"] == 2*model["parameters"]
        assert memory.json()["input_snapshot"]["model"]["config_sha256"] == model["config_sha256"]
        assert client.post(prefix + "/memory", json={"context_length": True}).status_code == 422
        assert client.get(prefix + "/structures/" + snapshot["id"] + "/instances").status_code == 200
        assert client.post("/api/models/import", json={"repo_id": record["repo_id"]}).json() == model
        if not model["workload_supported"]:
            caps = client.get("/api/workloads/capabilities", params={"model_id": model["id"]})
            assert caps.status_code == 200
            assert caps.json()["supported"] is False
            assert caps.json()["issues"]


@pytest.mark.parametrize("url", [
    "https://huggingface.co.evil/model/name", "https://huggingface.co/name/repo/blob/main/config.json",
    "https://huggingface.co/name/repo?download=true", "http://huggingface.co/name/repo",
    "https://user@huggingface.co/name/repo", "https://huggingface.co/name/../repo",
])
def test_import_url_boundary_no_network(url):
    def no_network(_):
        pytest.fail("invalid URL must fail before network")
    with pytest.raises(ValueError):
        HuggingFaceImporter(httpx.MockTransport(no_network)).import_model(url)
