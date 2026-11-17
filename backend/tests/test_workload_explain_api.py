"""Published trace scope, immutable export and historical compatibility."""
# ruff: noqa: F811
from app.workload.domain import content_hash
from app.workload.tables import RunRow
from test_workload_api import PREFIX, api, create, submit, wait  # noqa: F401


def test_explanation_published_scope_export_and_history(api):
    client, sessions, spec, _ = api
    saved = create(client, spec)
    run = wait(client, submit(client, saved)["id"])
    assert run["status"] == "completed", run
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    base = f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}'
    scope = plan["summary"]["explanation"]["scope"]
    assert scope["run_id"] == run["id"]
    assert scope["plan_id"] == plan["id"]
    assert scope["input_hash"] == run["input_hash"]
    for t in plan["summary"]["explanation"]["traces"].values():
        assert t["scope"]["run_id"] == run["id"]
    rank = client.get(base + "/ranks").json()["items"][0]
    assert rank["explanation"]["traces"]["peak"]["result"]["value"] == rank["peak_bytes"]
    inspection = client.get(base + "/inspection?layer=0&rank=0").json()
    assert inspection["explanation"]["scope"]["plan_id"] == plan["id"]
    assert inspection["explanation"]["scope"]["kind"] == "representative_batch"
    assert inspection["explanation"]["scope"]["memory_kind"] == "reserved_admission_bound"
    assert inspection["explanation"]["nodes"]["q_proj"]["operators"]
    assert client.get(base + "/inspection?layer=99999&rank=0").status_code == 404
    assert client.get(f'{PREFIX}/runs/wrong/plans/{plan["id"]}/inspection?layer=0&rank=0').status_code == 404
    exported = client.get(f'{PREFIX}/runs/{run["id"]}/export').json()
    assert exported["export_hash"] == content_hash({k: v for k, v in exported.items() if k != "export_hash"})
    assert exported["plans"][0]["summary"]["explanation"] == plan["summary"]["explanation"]
    with sessions() as session:
        row = session.get(RunRow, run["id"])
        row.versions = {**row.versions, "engine": "inference-planner-1.2.0"}
        session.commit()
    historical = client.get(base + "/inspection?layer=0&rank=0").json()
    assert historical["explanation"] is None
    assert historical["timing"]["operators"] == []
    assert client.get(base).json() == plan
