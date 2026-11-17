"""Independent interval, window and accounting invariants."""
from dataclasses import replace

import pytest
from app.engines.inference.ir import Candidate
from app.engines.inference.scheduler import add_interval, simulate
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir
from test_inference_scheduler import offline


def test_interval_union_accepts_out_of_order_overlaps_and_zero_width():
    spans = []
    for a, b in [(4, 8), (0, 2), (1, 5), (7, 9), (9, 9), (10, 11), (8, 10)]:
        add_interval(spans, a, b)
    assert spans == [(0, 11)]


@pytest.mark.parametrize("c", [Candidate(2, 1, 1, 1, ()), Candidate(1, 2, 1, 2, (1,)),
                              Candidate(2, 1, 2, 1, ())])
def test_primary_breakdown_token_relation_and_union(c):
    result = simulate(mock_sources(ir=mock_ir(moe=c.ep > 1)), c,
                      offline(count=7, concurrency=4), details=True)
    e, m = result["explanation"], result["metrics"]
    assert sum(e["decode_breakdown_ms"].values()) == pytest.approx(m["iteration_time_ms"])
    assert e["itl_wave_service_ms"] + e["itl_between_wave_wait_ms"] == pytest.approx(m["itl_ms"])
    assert 0 < m["gpu_activity_fraction"] <= 1 + 1e-12
    # Independent sweep: integrate count>0 on all unique endpoints, no add_interval reuse.
    busy = 0
    for rank in range(c.gpu_count):
        spans = [(s["start"], s["start"] + s["rank_activity_seconds"].get(rank, 0))
                 for w in result["waves"] for p in w["stages"] for s in p["segments"]]
        points = sorted({x for span in spans for x in span})
        busy += sum(b - a for a, b in zip(points, points[1:])
                    if any(start <= a and stop >= b for start, stop in spans))
    assert e["gpu_activity_seconds"] == pytest.approx(busy)
    assert m["gpu_activity_fraction"] == pytest.approx(busy / (c.gpu_count * m["window_seconds"]))


def test_warmup_subtracts_completed_request_outputs_before_window():
    source, c = mock_sources(), Candidate(1, 1, 1, 1, ())
    base = simulate(source, c, offline(output=4), details=True)
    first = base["requests"][0]["first"]
    spec = design(mode="online", load={"prompt_tokens": 5, "output_tokens": 4, "concurrency": 1,
                  "request_rate": 1, "duration_seconds": .01, "warmup_seconds": first + 1e-9},
                  runtime={"max_batch_sequences": 1, "prefill_chunk_tokens": 5})
    result = simulate(source, c, spec)
    e, m = result["explanation"], result["metrics"]
    assert e["completed_outputs_before_window"] == 1
    assert e["completed_full_outputs"] == 4
    assert e["unfinished_outputs_in_window"] == 0
    assert m["emitted_tokens"] == 3
    assert e["completed_full_outputs"] + e["unfinished_outputs_in_window"] - e["completed_outputs_before_window"] == m["emitted_tokens"]


@pytest.mark.parametrize("kind", ["partial", "unknown", "cancelled", "o1", "zero_window"])
def test_no_sample_and_unavailable_propagation(kind):
    source, spec = mock_sources(), offline()
    kw = {}
    if kind == "partial":
        kw["event_budget"] = 2
    elif kind == "unknown":
        source = replace(source, hardware={**source.hardware, "compute": {}})
    elif kind == "cancelled":
        kw["cancelled"] = lambda: True
    elif kind == "o1":
        spec = offline(output=1)
    else:
        spec = spec.model_copy(update={"load": spec.load.model_copy(update={"warmup_seconds": 1})})
    result = simulate(source, Candidate(1, 1, 1, 1, ()), spec, **kw)
    assert result["metrics"]["iteration_time_ms"] is None
    assert all(v is None for v in result["explanation"]["decode_breakdown_ms"].values())
    if kind in ("partial", "unknown", "cancelled", "zero_window"):
        assert result["metrics"]["gpu_activity_fraction"] is None
    if kind in ("partial", "unknown", "cancelled"):
        assert result["explanation"]["status"] == "unavailable"


def test_primary_statistics_do_not_depend_on_diagnostic_wave_prefix():
    source, spec, c = mock_sources(), offline(count=1002), Candidate(1, 1, 1, 1, ())
    detailed = simulate(source, c, spec, details=True)
    plain = simulate(source, c, spec)
    assert len(detailed["waves"]) == 2000 < detailed["wave_count"]
    assert detailed["explanation"] == plain["explanation"]
    assert detailed["metrics"] == plain["metrics"]
