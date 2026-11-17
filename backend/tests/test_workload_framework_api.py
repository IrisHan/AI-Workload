"""Isolated API/worker evidence for framework layouts and persisted cycles."""
# ruff: noqa: F811
from copy import deepcopy
from pathlib import Path

import pytest
from app.integrations.huggingface import normalize_model
from app.repositories.catalog import CatalogRepository
from test_workload_api import PREFIX, api, create, submit, wait  # noqa: F401


def test_framework_cycle_roundtrip_and_preview(api):
    client, _, original, _ = api
    spec = deepcopy(original)
    spec["parallelism"].update(layout="vllm-v0.10.2", mode="manual",
                              manual_candidate={"tp": 2, "pp": 1, "ep": 1, "dp": 1, "boundaries": []})
    spec["resources"].update(gpus_per_node=2, topology_template="switched_sxm_reference-v1")
    spec["runtime"]["speculative"].update(enabled=True, draft_tokens=2, acceptance_rate=.7,
                                         draft_latency_ms=0, draft_memory_bytes=0, include_prefill=True)
    caps = client.get(PREFIX + "/capabilities", params={"model_id": spec["model_id"]}).json()
    assert set(caps["layouts"]) == {"orthogonal-tp-ep-v1", "vllm-v0.10.2", "sglang-v0.5.3"}
    assert caps["parallel_domains_by_layout"]["vllm-v0.10.2"]["tp"][0] == 1
    preview = client.post(PREFIX + "/preview", json={"input": spec, "client_revision": 1})
    assert preview.status_code == 200, preview.text
    assert preview.json()["diagnostics"] == []
    saved = create(client, spec)
    run = wait(client, submit(client, saved)["id"])
    assert run["status"] == "completed", run
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    cycle = plan["summary"]["speculative_cycle"]
    assert cycle["expected_emitted_tokens"] == pytest.approx(2.19)
    assert cycle["output_tokens_s"] > 0
    assert plan["summary"]["gpu_count"] == 2
    assert plan["summary"]["framework_config"]["args"][:2] == ["--tensor-parallel-size", "2"]
    exported = client.get(f'{PREFIX}/runs/{run["id"]}/export').json()
    assert exported["frozen_input"]["design"] == spec
    assert exported["plans"][0]["summary"]["speculative_cycle"] == cycle
    assert client.get(f'{PREFIX}/designs/{saved["id"]}').json()["revision"]["input"] == spec


@pytest.mark.parametrize("layout,ep,enabled", [("vllm-v0.10.2", 4, True), ("sglang-v0.5.3", 2, False)])
def test_real_moe_layout_worker_inspection(api, layout, ep, enabled):
    client, sessions, original, _ = api
    with sessions() as session:
        config = Path(__file__).resolve().parents[2] / (
            "docs/reports/2026-09-13-model-center/evidence/sources/"
            "deepseek-ai--deepseek-moe-16b-base/config.json")
        model = normalize_model("deepseek-ai/deepseek-moe-16b-base",
                                "521d2bc4fb69a3f3ae565310fcc3b65f97af2580",
                                config.read_bytes(), "2026-09-13")
        CatalogRepository(session).add_model(model)
    spec = deepcopy(original)
    spec["model_id"] = model.id
    spec["parallelism"].update(layout=layout, expert_parallel=enabled, mode="manual",
                              manual_candidate={"tp": 2, "pp": 1, "ep": ep, "dp": 2, "boundaries": []})
    spec["resources"].update(gpus_per_node=4, topology_template="switched_sxm_reference-v1")
    spec["load"].update(concurrency=2, request_count=2)
    saved = create(client, spec)
    run = wait(client, submit(client, saved)["id"])
    assert run["status"] == "completed", run
    assert run["summary"]["outcome"] == "feasible", run
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    assert plan["summary"]["gpu_count"] == 4
    assert plan["summary"]["etp"] == 1
    inspection = client.get(f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}/inspection?layer=1&rank=3')
    assert inspection.status_code == 200, inspection.text
    assert inspection.json()["resident"]
    assert any(w["expert_start"] is not None for w in inspection.json()["shards"])
