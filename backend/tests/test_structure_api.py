from app.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import inspect
from test_planner import design_input
from test_workload_api import create, submit, wait


def test_structure_snapshot_idempotent_and_persistent(tmp_path):
    url = f"sqlite:///{tmp_path / 'structure.db'}"
    with TestClient(create_app(url)) as client:
        model = client.get("/api/models").json()[1]
        prefix = f"/api/models/{model['id']}"
        before = client.post(prefix + "/analyze").json()
        response = client.post(prefix + "/structure")
        assert response.status_code == 200, response.text
        snapshot = response.json()
        assert snapshot["comparison"]["matched"] is True
        assert all(value == 0 for value in snapshot["comparison"]["category_deltas"].values())
        assert snapshot["analysis"]["totals"]["total"] == 1543714304
        assert snapshot["input_snapshot"] == model
        assert client.post(prefix + "/structure").json() == snapshot
        assert client.post(prefix + "/analyze").json() == before
    app = create_app(url)
    with TestClient(app) as client:
        assert client.get(prefix + "/structures").json() == [snapshot]
        assert client.get(prefix + "/structures/" + snapshot["id"]).json() == snapshot
        other = client.get("/api/models").json()[0]["id"]
        assert client.get(f"/api/models/{other}/structures/{snapshot['id']}").status_code == 404
        assert "model_structure_snapshots" in inspect(app.state.engine).get_table_names()
        assert client.post("/api/models/missing/structure").status_code == 404


def test_structure_preserves_frozen_inference_run(tmp_path):
    with TestClient(create_app(f"sqlite:///{tmp_path / 'inference.db'}")) as client:
        model = client.get("/api/models").json()[0]
        run = wait(client, submit(client, create(client, design_input(client)))["id"])
        url = f"/api/workloads/runs/{run['id']}/export"
        frozen = client.get(url).json()
        assert client.post(f"/api/models/{model['id']}/structure").status_code == 200
        assert client.get(url).json() == frozen
