"""Independent execution goldens. All tiny fixtures carry mock provenance."""
from dataclasses import replace

import pytest
from app.engines.inference.cache import cache_allocation, kv_bytes, request_quotas
from app.engines.inference.communication import collective, ring_time, routing_matrix, transfer_round
from app.engines.inference.compute import attention_pairs, stage_program
from app.engines.inference.ir import Candidate
from app.engines.inference.lifetimes import Allocation, peak_ledger
from app.engines.inference.memory import admission_memory, compose
from app.engines.inference.partition import ranks
from app.engines.inference.roofline import operator_time
from app.workload.domain import FrozenSources
from app.workload.sources import runtime_assumptions
from test_inference_partition import design, mock_ir


def mock_sources(ir=None, spec=None):
    spec = spec or design()
    hardware = {
        "id": "mock-gpu", "memory_bytes": 16 * 1024**3, "memory_bandwidth_bytes_s": 1e12,
        "compute": {"bf16": {"status": "supported", "dense": 100},
                    "fp32": {"status": "supported", "dense": 10}},
        "cost": {"cny_per_day": None, "as_of": None}, "power_watts": None,
    }
    topology = {
        "intra_injection_bytes_s": 1000, "inter_injection_bytes_s": 100,
        "node_network_bytes_s": 100, "link_efficiency": 1,
        "intra_latency_s": 0.001, "inter_latency_s": 0.002,
        "sharing_domains": ["gpu-injection", "node-network"],
    }
    return FrozenSources({}, {}, hardware, ir or mock_ir(), topology, runtime_assumptions(spec))


def test_g05_page_payload_and_metadata_separate():
    layer = mock_ir().layers[0]
    assert 2 * kv_bytes(layer, 10, 2, paged=False) == 320
    assert 2 * kv_bytes(layer, 10, 2) == 512
    allocation = cache_allocation(layer, [10, 10], 2, "bf16")
    assert allocation == {"payload_bytes": 512, "metadata_bytes": 16,
                          "tail_waste_bytes": 192, "total_bytes": 528}
    windowed = replace(layer, window=8)
    assert kv_bytes(windowed, 100, 2) == kv_bytes(windowed, 10000, 2)


def test_g08_simultaneous_lifetimes():
    result = peak_ledger({"weight": 1000, "kv": 200}, [
        Allocation("a", "activation", 300, 0, 1, "mock"),
        Allocation("b", "activation", 500, 1, 2, "mock")])
    assert result["peak_bytes"] == 1700
    assert result["peak_at"] == 1
    assert result["components"]["activation"] == 500


def test_g09_chunk_attention_pairs():
    assert attention_pairs(0, 3) == 6
    assert attention_pairs(3, 2) == 9
    assert attention_pairs(0, 5) == 15
    assert attention_pairs(2, 5, 4) == 19  # 3+4+4+4+4
    assert attention_pairs(1000000, 1000000, 4) == 4000000


def test_g10_ring_and_injection_round():
    assert ring_time(4, 1200, 1000, .001) == pytest.approx(1.806)
    assert ring_time(4, 1200, 1000, .001, "all_gather") == pytest.approx(.903)
    assert ring_time(1, 1200, None, .001) == 0
    c = Candidate(4, 1, 1, 1, ())
    result = collective([0, 1, 2, 3], 1200, ranks(c, 8), mock_sources().topology)
    assert result["seconds"] == pytest.approx(1.806)
    assert result["network_bytes"] == 7200


def test_network_contention_aggregates_node_uplink():
    placements = ranks(Candidate(4, 1, 1, 1, ()), 2)
    topology = mock_sources().topology
    single = transfer_round([(0, 2, 100)], placements, topology)
    shared = transfer_round([(0, 2, 100), (1, 3, 100)], placements, topology)
    assert single["seconds"] == pytest.approx(1.203)
    assert shared["seconds"] == pytest.approx(2.203)
    missing = {**topology, "node_network_bytes_s": None}
    assert transfer_round([(0, 2, 100)], placements, missing)["seconds"] is None


def test_routing_conserves_distinct_topk_and_dp_quota():
    matrix = routing_matrix([3, 2], 4, 2)
    assert sum(map(sum, matrix)) == 10
    assert sum(matrix[0]) == 6
    assert max(matrix[0]) <= 3
    concentrated = routing_matrix([3, 2], 4, 2, concentrated=True)
    assert concentrated == [[3, 3, 0, 0], [2, 2, 0, 0]]
    assert request_quotas(7, Candidate(1, 1, 2, 2, ())) == ((2, 2), (2, 1))


def test_g13_one_byte_capacity_boundary():
    spec = design(overrides=[
        {"path": "runtime_fixed_bytes", "value": 0, "reason": "mock golden"},
        {"path": "safety_fraction", "value": 0, "reason": "mock golden"},
        {"path": "memory_efficiency", "value": 1, "reason": "mock golden"},
    ])
    assumptions = runtime_assumptions(spec)
    assert compose(1000, 200, 500, 0, 1700, assumptions)["capacity_pass"]
    result = compose(1000, 200, 500, 0, 1699, assumptions)
    assert not result["capacity_pass"]
    assert result["deficit_bytes"] == 1
    assert result["free_bytes"] == -1


def test_reserved_memory_components_and_dp_recomputation():
    spec = design()
    sources = mock_sources(spec=spec)
    c = Candidate(2, 2, 1, 1, (1,))
    single = admission_memory(sources.ir, c, spec, sources.hardware, sources.assumptions)
    doubled = admission_memory(sources.ir, replace(c, dp=2), spec, sources.hardware, sources.assumptions)
    assert sum(single["components"].values()) == single["peak_bytes"]
    assert doubled["components"]["kv"] < single["components"]["kv"]
    assert len(single["ranks"]) == 4
    assert single["ranks"][0]["resident_request_quota"] == 8


def test_compute_uses_only_final_prefill_logits_and_actual_pairs():
    spec = design()
    sources = mock_sources(spec=spec)
    c = Candidate(1, 1, 1, 1, ())
    prefix = stage_program(sources, c, spec, 0, 0, [[(0, 3, False)]], details=True)
    suffix = stage_program(sources, c, spec, 0, 0, [[(3, 2, True)]], details=True)
    assert not any(row["id"] == "lm_head" for row in prefix["operators"])
    head = next(row for row in suffix["operators"] if row["id"] == "lm_head")
    assert head["flops"] == 2 * 16 * 32
    attention = next(row for row in suffix["operators"] if row["id"] == "attention")
    assert attention["flops"] == 4 * 4 * 4 * 9
    assert suffix["seconds"] > 0
    assert suffix["scalar_equivalent_ops"] > 0


def test_unknown_scalar_cannot_be_zero_or_tensor_core_substitution():
    sources = mock_sources()
    hardware = {**sources.hardware, "compute": {"bf16": {"status": "supported", "dense": 100}}}
    timing = operator_time(1000, 100, hardware, sources.assumptions, "bf16", scalar=True)
    assert timing["seconds"] is None
    result = stage_program(replace(sources, hardware=hardware), Candidate(1, 1, 1, 1, ()),
                           design(), 0, 0, [[(0, 5, True)]])
    assert result["seconds"] is None
    assert result["flops"] > 0
    assert result["unknown"]


def test_expert_flops_conservation_across_ep_and_tp():
    sources = mock_sources(ir=mock_ir(layers=1, moe=True))
    results = []
    for c, batch in [(Candidate(1, 1, 1, 1, ()), [[(0, 4, True)]]),
                     (Candidate(2, 1, 2, 1, ()), [[(0, 2, True)], [(0, 2, True)]])]:
        result = stage_program(sources, c, design(), 0, 0, batch, details=True)
        results.append(sum(row["flops"] for row in result["operators"]
                           if row["id"].startswith("routed_expert.")))
    assert results == [24576, 24576]  # 2*4tokens*2experts*3matrices*16*32
