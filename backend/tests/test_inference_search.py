"""Search coverage and deployment eligibility must remain explicit."""
from dataclasses import replace

from app.engines.inference.ir import Candidate
from app.engines.inference.search import boundaries, candidate_stream, evaluate, search
from app.workload.schemas import Search
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir
from test_inference_scheduler import offline


def test_candidates_are_in_gpu_order_and_include_non_power_two():
    ir = mock_ir()
    def scaled(weight):
        return replace(weight, shape=tuple(axis * 3 for axis in weight.shape))
    layers = tuple(replace(layer, heads=6, kv_heads=3, hidden=48, head_dim=8, intermediate=96,
                           weights=tuple(scaled(w) for w in layer.weights)) for layer in ir.layers)
    ir = replace(ir, vocab=96, hidden=48, layers=layers, shared=tuple(scaled(w) for w in ir.shared))
    candidates = [c for c, _ in candidate_stream(ir, design())]
    assert [c.gpu_count for c in candidates] == sorted(c.gpu_count for c in candidates)
    assert any(c.tp == 3 for c in candidates)


def test_partition_dp_honestly_reports_partial():
    cuts, exhaustive = boundaries(mock_ir(layers=20), 4)
    assert not exhaustive
    assert (5, 10, 15) in cuts
    assert all(len(cut) == 3 and sorted(set(cut)) == list(cut) for cut in cuts)
    exact, exhaustive = boundaries(mock_ir(layers=4), 2)
    assert exhaustive
    assert exact == [(1,), (2,), (3,)]


def test_search_deterministic_order_and_smallest_found():
    spec = offline(output=1)
    sources = mock_sources()
    options = Search(event_budget=10000)
    first = search(sources, spec, options)
    second = search(sources, spec, options)
    assert first == second
    assert first["outcome"] == "feasible"
    assert first["plans"][0]["gpu_count"] == 1
    assert first["coverage"]["minimum_proven"]
    assert len({p["id"] for p in first["plans"]}) == len(first["plans"])


def test_one_candidate_limit_and_no_solution_is_inconclusive():
    spec = offline()
    sources = mock_sources()
    sources = replace(sources, hardware={**sources.hardware, "memory_bytes": 1})
    result = search(sources, spec, Search(max_candidates=1))
    assert result["status"] == "partial"
    assert result["outcome"] == "inconclusive"
    assert result["coverage"]["reasons"] == {"CAPACITY": 1}
    assert result["rejections"][0]["details"]["deficit"] > 0


def test_capacity_only_unknown_timing_not_recommendation():
    sources = mock_sources()
    sources = replace(sources, hardware={**sources.hardware, "compute": {}})
    plan = evaluate(sources, Candidate(1, 1, 1, 1, ()), offline(), 100)
    assert plan["recommendation_status"] == "capacity_only"
    assert plan["performance"]["metrics"]["ttft_ms"] is None
    result = search(sources, offline(), Search(max_candidates=1))
    assert result["outcome"] == "inconclusive"


def test_dp_recovers_capacity_after_single_replica_oom():
    # Low-capacity mock, no fixed reserve/safety. KV-heavy and tiny activations.
    spec = design(mode="offline", load={
        "prompt_tokens": 1000, "output_tokens": 1, "concurrency": 8, "request_count": 8,
        "request_rate": None, "arrival": None,
    }, runtime={"max_batch_sequences": 1, "prefill_chunk_tokens": 16},
        parallelism={"allowed_tp": [1], "allowed_pp": [1], "dp_scaling": True},
        overrides=[{"path": key, "value": value, "reason": "mock capacity golden"} for key, value in [
            ("runtime_fixed_bytes", 0), ("safety_fraction", 0), ("memory_efficiency", 1)]])
    sources = mock_sources(spec=spec)
    sources = replace(sources, hardware={**sources.hardware, "memory_bytes": 200000})
    result = search(sources, spec, Search(event_budget=100000))
    assert result["coverage"]["reasons"]["CAPACITY"] >= 1
    assert result["outcome"] == "feasible"
    assert result["plans"][0]["candidate"]["dp"] > 1
    assert not result["coverage"]["minimum_proven"]
