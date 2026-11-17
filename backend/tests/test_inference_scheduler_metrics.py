"""Independent workload metric goldens, including measurement boundaries."""
import pytest
from app.engines.inference.ir import Candidate
from app.engines.inference.scheduler import simulate
from test_inference_execution import mock_sources
from test_inference_partition import design
from test_inference_scheduler import offline


def test_mfu_uses_matrix_work_and_actual_gpu_count_not_budget():
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), offline(), details=True)
    metrics = result["metrics"]
    # Two tiny layers; attention + FFN + final head, prefill then two decode forwards.
    expected = 49024 + 11008 + 11136
    assert metrics["model_flops_in_window"] == expected
    assert metrics["mfu"] == pytest.approx(expected / (100e12 * metrics["window_seconds"]))
    decode = [w for w in result["waves"] if w["phase"] == "decode"]
    assert metrics["iteration_time_ms"] == pytest.approx(sum(
        (w["stages"][-1]["stop"] - w["stages"][0]["start"]) * 1000 for w in decode) / 2)
    assert metrics["modeled_busy_fraction"] == pytest.approx(1)
    assert metrics["prefill_busy_fraction"] + metrics["decode_busy_fraction"] == pytest.approx(1)


def test_censored_window_clips_scheduled_work_and_warmup():
    spec = design(load={"prompt_tokens": 5, "output_tokens": 3, "concurrency": 1,
                        "duration_seconds": .000001, "warmup_seconds": .0000005,
                        "request_rate": 1.0}, runtime={"max_batch_sequences": 1})
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), spec)
    metrics = result["metrics"]
    # The entire measured window lies inside the first embedding operator.
    assert result["status"] == "censored"
    assert metrics["model_flops_in_window"] == 0
    assert metrics["mfu"] == 0
    assert metrics["prefill_busy_fraction"] == pytest.approx(1)
    assert metrics["decode_busy_fraction"] == 0
    assert metrics["iteration_time_ms"] is None
    assert all(v is None for v in metrics["percentiles"].values())


@pytest.mark.parametrize("count,available", [(19, []), (20, ["p50_ms"]),
                                          (100, ["p50_ms", "p95_ms"]),
                                          (10000, ["p50_ms", "p95_ms", "p99_ms"])])
def test_nearest_rank_sample_gates(count, available):
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()),
                      offline(count=count, output=1), details=True)
    metrics = result["metrics"]
    assert result["status"] == "completed"
    assert metrics["percentile_sample_count"] == count
    assert metrics["iteration_time_ms"] is None
    assert [k for k, v in metrics["percentiles"].items() if v is not None] == available
    first_ms = result["requests"][0]["finished"] * 1000
    for key, fraction in (("p50_ms", .5), ("p95_ms", .95), ("p99_ms", .99)):
        if key in available:
            assert metrics["percentiles"][key] == pytest.approx(first_ms * count * fraction)


def test_partial_and_zero_window_have_no_complete_metrics():
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), offline(count=10), event_budget=2)
    assert result["status"] == "partial"
    for key in ("mfu", "iteration_time_ms", "prefill_tokens_s_per_gpu", "modeled_busy_fraction"):
        assert result["metrics"][key] is None
    spec = offline().model_copy(update={"load": offline().load.model_copy(update={"warmup_seconds": 1.0})})
    result = simulate(mock_sources(), Candidate(1, 1, 1, 1, ()), spec)
    assert result["metrics"]["window_seconds"] == 0
    assert result["metrics"]["mfu"] is None
