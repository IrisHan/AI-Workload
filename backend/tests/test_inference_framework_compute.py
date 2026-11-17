"""Independent routed work, communication and HBM scenario goldens."""
import pytest
from app.engines.inference.compute import stage_program
from app.engines.inference.ir import Candidate
from app.engines.inference.roofline import operator_time
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir


@pytest.mark.parametrize("candidate,batches", [
    (Candidate(2, 1, 4, 2, (), "vllm-v0.10.2", True), [[(0, 2, True)], [(0, 2, True)]]),
    (Candidate(2, 1, 1, 2, (), "vllm-v0.10.2", True), [[(0, 2, True)], [(0, 2, True)]]),
    (Candidate(4, 1, 2, 1, (), "sglang-v0.5.3", True), [[(0, 4, True)]]),
])
def test_framework_routed_flops_conserved(candidate, batches):
    sources = mock_sources(ir=mock_ir(layers=1, moe=True))
    result = stage_program(sources, candidate, design(), 0, 0, batches, details=True)
    routed = [o for o in result["operators"] if o["id"].startswith("routed_expert.")]
    assert sum(o["flops"] for o in routed) == 24576
    assert sum(s["model_flops"] for s in result["segments"]) == result["flops"]
    assert sum(s["seconds"] for s in result["segments"]) == pytest.approx(result["seconds"])
    exchange = next(e for e in result["communication"] if e["kind"] == "expert_dispatch_combine")
    assert "output" in exchange
    assert exchange["seconds"] == pytest.approx(
        sum(exchange[k]["seconds"] for k in ("dispatch", "combine", "output")))


def test_hbm_override_matches_excel_without_mutating_source():
    inp = design(overrides=[{"path": "hbm_bandwidth_gb_s", "value": 2000, "reason": "mock Excel"}])
    source = mock_sources(spec=inp)
    timing = operator_time(0, 67108864, source.hardware, source.assumptions, "bf16")
    assert timing["hbm_seconds"] * 1000 == pytest.approx(0.04793490285714286)
    assert source.hardware["memory_bandwidth_bytes_s"] == 1e12
