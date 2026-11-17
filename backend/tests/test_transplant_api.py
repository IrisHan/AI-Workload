import importlib
import json
from pathlib import Path

from app.lib.spec_studio.glossary import load_glossary
from app.main import create_app
from fastapi.testclient import TestClient


def test_transplant_model_options_paging_cost_and_ownership():
    with TestClient(create_app("sqlite:///:memory:")) as client:
        models = client.get("/api/models").json()
        model = models[1]
        prefix = f"/api/models/{model['id']}"
        before = client.post(prefix + "/structure").json()
        changed = client.post(prefix + "/structure", json={"kv_dtype": "fp32"}).json()
        assert before["id"] != changed["id"]
        assert before["analysis"]["serving"]["kv_layout"]["bytes_per_token"] == 28672
        assert changed["analysis"]["serving"]["kv_layout"]["bytes_per_token"] == 57344
        assert before["analysis"]["totals"] == changed["analysis"]["totals"]
        assert client.post(prefix + "/structure", json={"kv_dtype": "fp32"}).json() == changed
        assert len(client.get(prefix + "/structures").json()) == 2
        base = prefix + "/structures/" + before["id"]
        page = client.get(base + "/instances?offset=20&limit=20").json()
        assert [row["index"] for row in page["items"]] == list(range(20, 28))
        assert client.get(base + "/instances?limit=101").status_code == 422
        assert client.get(base + "/instances?offset=28").json()["items"] == []
        template = before["analysis"]["templates"][0]
        assert client.get(base + "/templates/" + template["id"]).json() == template
        gpu = client.get("/api/hardware").json()[0]
        body = {"hardware_id": gpu["id"], "phase": "decode", "context_length": 128}
        cost = client.post(base + "/cost", json=body)
        assert cost.status_code == 200, cost.text
        assert cost.json()["calibrated"] is False
        assert cost.json()["input_snapshot"]["hardware"]["version"] == gpu["version"]
        assert cost.json()["input_snapshot"]["structure_snapshot_id"] == before["id"]
        assert client.post(base + "/cost", json={**body, "hardware": {}}).status_code == 422
        assert client.post(base + "/cost", json={**body, "hardware_id": "missing"}).status_code == 404
        wrong = f"/api/models/{models[0]['id']}/structures/{before['id']}"
        assert client.get(wrong + "/instances").status_code == 404
        assert client.get(wrong + "/templates/" + template["id"]).status_code == 404
        assert client.post(wrong + "/cost", json=body).status_code == 404
        assert client.post(prefix + "/structure", json={"repo_id": "other/model"}).status_code == 422
        assert client.post(prefix + "/structure", json={"kv_dtype": "invalid"}).status_code == 422
        assert client.get("/api/structure/glossary").json()["gqa"]


def test_all_engine_modules_are_backend_owned_and_resources_packaged():
    root = Path(__file__).resolve().parents[1]
    package = root / "app/lib/spec_studio"
    for module in package.glob("*.py"):
        imported = importlib.import_module("app.lib.spec_studio." + module.stem)
        assert Path(imported.__file__).is_relative_to(root)
    assert load_glossary()["rope"]
    config = json.loads((Path(__file__).parent / "spec_studio/fixtures/deepseek-v3.json").read_text())
    from app.lib.spec_studio.grouped import inspect_grouped
    result = inspect_grouped(config)
    assert [group["count"] for group in result["groups"]] == [3, 58]
    assert result["totals"]["total"] == 671026419200
