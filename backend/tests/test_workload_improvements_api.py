"""Source defaults, frozen metrics and reusable design search integration."""
# pytest discovers the shared fixture by name.
# ruff: noqa: F811
from uuid import uuid4

import pytest
from app.workload.tables import RunRow
from test_workload_api import PREFIX, api, create, submit, wait  # noqa: F401


@pytest.mark.parametrize("sku,intra,inter", [
    ("nvidia-a100-sxm-80gb-20260913", 300e9, 25e9),
    ("nvidia-h100-sxm-80gb-20260913", 450e9, 50e9),
    ("nvidia-a100-pcie-80gb-20260913", None, None),
])
def test_capabilities_source_units_and_domains(api, sku, intra, inter):
    client, _, spec, _ = api
    response = client.get(PREFIX + "/capabilities", params={"model_id": spec["model_id"], "hardware_id": sku})
    assert response.status_code == 200, response.text
    caps = response.json()
    assert caps["hardware"]["id"] == sku
    defaults = caps["topology_defaults"]["switched_sxm_reference-v1"]
    assert defaults["intra_bandwidth_bytes_s"] == intra
    assert defaults["inter_bandwidth_bytes_s"] == inter
    assert caps["parallel_domains"]["tp"][0] == 1
    assert caps["parallel_domains"]["ep"] == [1]


def test_metrics_persist_and_old_engine_inspection_does_not_recalculate(api):
    client, sessions, spec, _ = api
    saved = create(client, spec)
    run = wait(client, submit(client, saved)["id"])
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    base = f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}'
    metrics = plan["summary"]["performance"]["metrics"]
    assert 0 < metrics["mfu"] < 1
    assert metrics["iteration_time_ms"] > 0
    assert metrics["prefill_tokens_s_per_gpu"] > 0
    assert metrics["cost_per_million_output_tokens"] is None
    assert metrics["energy_per_million_output_tokens"] is None
    timing = client.get(base + "/inspection?layer=0&rank=0").json()["timing"]
    operator = next(o for o in timing["operators"] if o["id"] == "q_proj")
    assert operator["seconds"] == pytest.approx(max(operator["compute_seconds"], operator["hbm_seconds"])
                                               + operator["launch_seconds"])
    reopened = client.get(f'{PREFIX}/designs/{saved["id"]}').json()
    assert reopened["runs"][0]["id"] == run["id"]
    assert reopened["model_name"] and reopened["hardware_name"]
    assert client.get(base).json() == plan
    with sessions() as session:
        row = session.get(RunRow, run["id"])
        row.versions = {**row.versions, "engine": "inference-planner-1.0.0"}
        session.commit()
    old = client.get(base + "/inspection?layer=0&rank=0").json()
    assert old["timing"]["operators"] == []
    assert "Historical" in old["timing"]["reason"]
    assert client.get(base).json() == plan


def test_recent_search_by_name_source_and_cursor_scope(api):
    client, _, spec, _ = api
    first = create(client, spec)
    second = create(client, spec)
    changed = {**spec, "load": {**spec["load"], "prompt_tokens": 6}}
    response = client.post(f'{PREFIX}/designs/{first["id"]}/revisions', json={"input": changed},
                           headers={"If-Match": first["etag"]})
    assert response.status_code == 201
    page = client.get(PREFIX + "/designs", params={"q": "acceptance", "limit": 1}).json()
    assert page["items"][0]["id"] == first["id"]
    next_page = client.get(PREFIX + "/designs", params={
        "q": "acceptance", "cursor": page["next_cursor"], "limit": 1}).json()
    assert next_page["items"][0]["id"] == second["id"]
    assert next_page["next_cursor"] is None
    assert client.get(PREFIX + "/designs", params={"cursor": page["next_cursor"]}).status_code == 422
    for query in (spec["hardware_id"], first["model_name"], "H100"):
        assert len(client.get(PREFIX + "/designs", params={"q": query}).json()["items"]) == 2
    assert client.get(PREFIX + "/designs", params={"q": uuid4().hex}).json()["items"] == []
    assert client.get(PREFIX + "/designs", params={"q": "%_"}).json()["items"] == []
