"""Regression of public source-graph and per-GPU inspection contracts."""
# pytest discovers this imported fixture by name.
# ruff: noqa: F811
import copy

from test_workload_api import PREFIX, api, create, submit, wait  # noqa: F401


def test_inspection_frozen_input_shards_phase_and_nonresident(api):
    client, _, spec, _ = api
    spec = copy.deepcopy(spec)
    spec["resources"]["gpus_per_node"] = 4
    spec["resources"]["topology_template"] = "switched_sxm_reference-v1"
    spec["parallelism"]["mode"] = "manual"
    spec["parallelism"]["manual_candidate"] = {"tp": 2, "pp": 2, "ep": 1, "dp": 1, "boundaries": [12]}
    design = create(client, spec)
    run = wait(client, submit(client, design)["id"])
    assert run["status"] == "completed", run
    assert run["design_input"] == spec
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    base = f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}'
    prefill = client.get(base + "/inspection?layer=0&rank=0&phase=prefill")
    assert prefill.status_code == 200, prefill.text
    a = prefill.json()
    b = client.get(base + "/inspection?layer=0&rank=0&phase=decode").json()
    assert a["result_hash"] == plan["result_hash"]
    assert a["local_tokens"] == 5 and b["local_tokens"] == 1
    assert a["cache"] == b["cache"]  # Same reserved request context.
    assert a["activation_peak_bytes"] > b["activation_peak_bytes"]
    q = next(s for s in a["shards"] if s["module_id"] == "q_proj")
    # Qwen2.5-0.5B H=896, 14 heads, TP2: local Q [448,896], BF16.
    assert q["local_shape"] == [448, 896]
    assert q["bytes"] == (448 * 896 + 448) * 2
    other = client.get(base + "/inspection?layer=0&rank=2").json()
    assert not other["resident"] and other["shards"] == [] and other["allocations"] == []
    assert other["cache"]["total_bytes"] == 0
    assert client.get(base + "/inspection?layer=999&rank=0").status_code == 404
    assert client.get(base + "/inspection?layer=0&rank=0&phase=mixed").status_code == 422


def test_graph_preserves_real_branches_and_source_ids(api):
    client, _, spec, _ = api
    run = wait(client, submit(client, create(client, spec))["id"])
    plan = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()["items"][0]
    graph = client.get(f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}/graph?layer=0').json()
    payload = client.get(f'{PREFIX}/runs/{run["id"]}/export').json()
    source = payload["frozen_input"]["sources"]["structure"]["analysis"]
    template = source["templates"][0]
    by_id = {n["id"]: n["source_node_id"] for n in graph["nodes"]}
    edges = {(by_id[e["source"]], by_id[e["target"]]) for e in graph["edges"]}
    assert edges == {(e["source"], e["target"]) for e in template["edges"]}
    assert all(n["layer_index"] == 0 for n in graph["nodes"])
    assert len(by_id) == len(template["nodes"])
