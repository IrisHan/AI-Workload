"""Network projection and rail efficiency acceptance, isolated from preview data."""
from copy import deepcopy

import pytest
from app.engines.inference.communication import transfer_round
from app.engines.inference.ir import Rank
from app.workload.network import network_graph
from app.workload.schemas import DesignInput
from app.workload.sources import topology_snapshot
from test_workload_api import PREFIX, api, create, submit, wait  # noqa: F401


def mock_sources():
    return {"hardware": {"node": {"nic_gb_s_per_gpu": 50}}, "topology": {
        "template": "switched_sxm_reference-v1", "intra_injection_bytes_s": 1000,
        "inter_injection_bytes_s": 100, "node_network_bytes_s": 100,
        "link_efficiency": .5, "inter_link_efficiency": 1,
        "intra_latency_s": .002, "inter_latency_s": .01, "sharing_domains": [],
    }}


def test_rail_transfer_independent_efficiencies_and_legacy():
    # 100 bytes: two GPU hops .2s each + NIC 1s + startup .012s.
    topology = mock_sources()["topology"]
    placements = [Rank(rank=0, d=0, p=0, e=0, t=0, node=0, gpu=0),
                  Rank(rank=1, d=0, p=1, e=0, t=0, node=1, gpu=0)]
    assert transfer_round([(0, 1, 100)], placements, topology)["seconds"] == pytest.approx(1.412)
    del topology["inter_link_efficiency"]
    assert transfer_round([(0, 1, 100)], placements, topology)["seconds"] == pytest.approx(2.412)
    topology["inter_injection_bytes_s"] = None
    assert transfer_round([(0, 1, 100)], placements, topology)["seconds"] is None


def test_network_128_gpus_dp_fold_and_endpoint_integrity():
    summary = {"candidate": {"ep": 1, "dp": 2}, "placement": [
        {"rank": r, "node": r // 8, "gpu": r % 8, "d": r // 128, "p": r % 128 // 8, "t": r % 8, "e": 0}
        for r in range(256)], "groups": [
        {"id": f"tp:{n}", "kind": "tp", "ranks": list(range(n * 8, (n + 1) * 8))}
        for n in range(32)]}
    before = deepcopy(summary)
    nodes, edges, network = network_graph(summary, mock_sources(), [])
    assert len(nodes) == 128 and len({n["node"] for n in nodes}) == 16
    assert len(edges) == 128  # Complete rings, including the last-to-first edge.
    assert network["replica_count"] == 2
    assert all(e["source"] in {n["id"] for n in nodes} and e["target"] in {n["id"] for n in nodes} for e in edges)
    assert all(e["statistics"]["decode"]["bytes"] is None for e in edges)
    assert summary == before


def test_network_saved_operations_not_multiplied_and_unknown_bandwidth():
    summary = {"candidate": {"ep": 1, "dp": 1}, "placement": [
        {"rank": r, "node": r, "gpu": 0, "d": 0, "p": 0, "t": r, "e": 0} for r in range(2)],
        "groups": [{"id": "tp:0", "kind": "tp", "ranks": [0, 1]}]}
    event = {"kind": "all_reduce", "phase": "decode", "wave_id": 3, "seconds": .02,
             "steps": 2, "ranks": [0, 1], "round": {"transfers": [
                 {"src": 0, "dst": 1, "bytes": 50, "path": ["gpu:0", "nic", "gpu:1"]},
                 {"src": 1, "dst": 0, "bytes": 50, "path": ["gpu:1", "nic", "gpu:0"]}]}}
    sources = mock_sources()
    _, edges, _ = network_graph(summary, sources, [event])
    assert edges[0]["statistics"]["decode"]["bytes"] == 200
    assert edges[0]["statistics"]["decode"]["operation_seconds"] == .02
    assert edges[0]["effective_bandwidth_bytes_s"] == 100
    sources["topology"]["inter_injection_bytes_s"] = None
    _, edges, _ = network_graph(summary, sources, [event], events_complete=False)
    assert edges[0]["bandwidth_bytes_s"] is None


def test_network_api_real_frozen_run(api):  # noqa: F811
    client, _, spec, _ = api
    spec["resources"] = {"node_count": 2, "gpus_per_node": 1, "topology_template": "switched_sxm_reference-v1"}
    spec["parallelism"] = {"mode": "manual", "manual_candidate": {
        "tp": 2, "pp": 1, "ep": 1, "dp": 1, "boundaries": []}}
    design = create(client, spec)
    run = submit(client, design)
    assert wait(client, run["id"])["status"] == "completed"
    plans = client.get(f'{PREFIX}/runs/{run["id"]}/plans').json()
    plan = plans["items"][0]
    base = f'{PREFIX}/runs/{run["id"]}/plans/{plan["id"]}'
    response = client.get(base + "/graph?view=communication")
    assert response.status_code == 200
    graph = response.json()
    assert graph["result_hash"] == plan["result_hash"]
    assert graph["network"]["fabric"] == "rail-optimized"
    assert graph["network"]["inter_bandwidth_bytes_s"] == 50e9
    assert graph["edges"][0]["efficiency"] == 1
    assert graph["edges"][0]["statistics"]["decode"]["operation_count"] > 0
    assert client.get(base).json() == plan
    assert client.get(base.replace(plan["id"], "missing") + "/graph?view=communication").status_code == 404


def test_network_override_validation_and_hardware_fallback():
    for value in (0, -1, 1.1):
        with pytest.raises(ValueError):
            DesignInput.model_validate({"model_id": "mock", "hardware_id": "mock", "resources": {
                "topology_overrides": {"inter_link_efficiency": value}}})
    design = DesignInput.model_validate({"model_id": "mock", "hardware_id": "mock", "resources": {
        "topology_template": "rdma_reference-v1", "node_count": 2,
        "topology_overrides": {"inter_link_efficiency": .8}}})
    topology = topology_snapshot(design, {"form_factor": "PCIe", "node": {"nic_gb_s_per_gpu": 50},
                                         "content_sha256": "mock"})
    assert topology["inter_injection_bytes_s"] == 50e9 and topology["inter_link_efficiency"] == .8


def test_ep_exchange_hub_and_single_gpu():
    summary = {"candidate": {"ep": 4, "dp": 1}, "placement": [
        {"rank": r, "node": r // 2, "gpu": r % 2, "d": 0, "p": 0, "t": 0, "e": r} for r in range(4)],
        "groups": []}
    event = {"kind": "expert_dispatch_combine", "phase": "decode", "wave_id": 1, "seconds": .03,
             "dispatch": {"transfers": [{"src": 0, "dst": 3, "bytes": 100, "path": ["gpu:0", "nic", "gpu:3"]}]},
             "combine": {"transfers": [{"src": 3, "dst": 0, "bytes": 100, "path": ["gpu:3", "nic", "gpu:0"]}]}}
    nodes, edges, _ = network_graph(summary, mock_sources(), [event])
    assert len(nodes) == 5 and len(edges) == 4
    assert edges[0]["statistics"]["decode"]["bytes"] == 200
    assert edges[0]["statistics"]["decode"]["operation_seconds"] == .03
    assert edges[1]["statistics"]["decode"]["bytes"] is None
    assert edges[0]["distance"] == "inter_node"
    summary["candidate"]["ep"] = 1
    summary["placement"] = summary["placement"][:1]
    nodes, edges, _ = network_graph(summary, mock_sources(), [])
    assert len(nodes) == 1 and edges == []
