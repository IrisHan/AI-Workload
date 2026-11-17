"""Request lifecycle and pipeline scheduling, independently asserted."""
from dataclasses import replace
from itertools import pairwise

from app.engines.inference.ir import Candidate
from app.engines.inference.scheduler import simulate
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir


def offline(count=1, output=3, concurrency=1, chunk=5, continuous=True):
    return design(mode="offline", load={
        "prompt_tokens": 5, "output_tokens": output, "concurrency": concurrency,
        "request_count": count, "request_rate": None, "arrival": None,
    }, runtime={"max_batch_sequences": 1, "prefill_chunk_tokens": chunk,
                "continuous_batching": continuous,
                "profile": "analytical-continuous-v1" if continuous else "analytical-sequential-v1"})


def test_g11_first_token_prefill_then_two_decode_forwards():
    spec = offline()
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), spec, details=True)
    assert result["status"] == "completed"
    assert result["wave_count"] == 3
    assert [wave["phase"] for wave in result["waves"]] == ["prefill", "decode", "decode"]
    assert [wave["contexts"][0][0][0] for wave in result["waves"]] == [0, 5, 6]
    assert result["requests"][0]["context"] == 7
    assert result["metrics"]["emitted_tokens"] == 3
    assert result["metrics"]["completed_requests"] == 1
    assert result["metrics"]["itl_ms"] > 0


def test_o1_has_no_decode_or_itl():
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), offline(output=1, chunk=3), details=True)
    assert result["wave_count"] == 2
    assert all(wave["phase"] == "prefill" for wave in result["waves"])
    assert result["requests"][0]["context"] == 5
    assert result["metrics"]["itl_ms"] is None
    assert result["metrics"]["emitted_tokens"] == 1


def test_pipeline_overlap_and_same_stage_exclusion():
    spec = offline(count=3, concurrency=3, output=2)
    result = simulate(mock_sources(), Candidate(1, 2, 1, 1, (1,)), spec, details=True)
    assert result["status"] == "completed"
    assert result["metrics"]["completed_requests"] == 3
    for p in range(2):
        spans = sorted((w["stages"][p]["start"], w["stages"][p]["stop"]) for w in result["waves"])
        assert all(left[1] <= right[0] for left, right in pairwise(spans))
    first, second = result["waves"][:2]
    assert second["stages"][0]["start"] < first["stages"][1]["stop"]
    assert second["stages"][0]["start"] == first["stages"][0]["stop"]


def test_dp_exact_quota_and_new_simulation():
    spec = offline(count=7, concurrency=7)
    result = simulate(mock_sources(), Candidate(1, 1, 1, 2, ()), spec, details=True)
    assert result["metrics"]["completed_requests"] == 7
    assert len([r for r in result["requests"] if r["d"] == 0]) == 4
    assert len([r for r in result["requests"] if r["d"] == 1]) == 3
    assert result == simulate(mock_sources(), Candidate(1, 1, 1, 2, ()), spec, details=True)


def test_moe_synchronized_wave_keeps_request_identity():
    spec = offline(count=4, concurrency=4)
    sources = mock_sources(ir=mock_ir(moe=True))
    result = simulate(sources, Candidate(2, 1, 2, 1, ()), spec, details=True)
    assert result["metrics"]["completed_requests"] == 4
    assert {r["e"] for r in result["requests"]} == {0, 1}
    assert result["communication_seconds"] > 0
    for wave in result["waves"]:
        assert len(wave["contexts"]) == 2


def test_event_budget_partial_and_cancellation():
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), offline(count=100), event_budget=10)
    assert result["status"] == "partial"
    assert result["reason"] == "event_budget"
    assert result["metrics"]["ttft_ms"] is None
    cancelled = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), offline(), cancelled=lambda: True)
    assert cancelled["status"] == "cancelled"


def test_unknown_scalar_and_censored_overload_no_sla_pass():
    sources = mock_sources()
    hardware = {**sources.hardware, "compute": {}}
    result = simulate(replace(sources, hardware=hardware), Candidate(1, 1, 1, 1, ()), offline())
    assert result["status"] == "unknown"
    assert result["sla"] == "unknown"
    spec = design(load={"prompt_tokens": 5, "output_tokens": 3, "concurrency": 1,
                        "duration_seconds": 0.01, "request_rate": 100000.0},
                  runtime={"max_batch_sequences": 1})
    result = simulate(sources, Candidate(1, 1, 1, 1, ()), spec)
    assert result["metrics"]["censored_count"] > 0
    assert result["metrics"]["overloaded"]
    assert result["sla"] != "pass"


def test_sequential_profile_waits_for_active_cohort():
    spec = offline(count=4, concurrency=2, continuous=False)
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), spec, details=True)
    cohort_one = [r for r in result["requests"] if r["id"] < 2]
    cohort_two = [r for r in result["requests"] if r["id"] >= 2]
    assert min(r["admitted"] for r in cohort_two) >= max(r["finished"] for r in cohort_one)
