"""Segment accounting and layer-only inspection regression."""
import pytest
from app.engines.inference.compute import stage_program
from app.engines.inference.ir import Candidate
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir


def test_segment_matrix_work_conserved_and_idle_ep_not_counted():
    source = mock_sources(ir=mock_ir(moe=True))
    result = stage_program(source, Candidate(2, 1, 2, 1, ()), design(), 0, 0,
                           [[(0, 3, True)], []], details=True)
    segments, operators = result["segments"], result["operators"]
    assert sum(s["model_flops"] for s in segments) == result["flops"]
    assert sum(s["busy_gpu_seconds"] for s in segments) == pytest.approx(
        sum(o["seconds"] for o in operators))
    assert sum(s["busy_gpu_seconds"] for s in segments) < sum(
        s["seconds"] * 4 for s in segments if s["kind"] == "compute")
    assert all(s["model_flops"] == 0 for s in segments if s["kind"] == "communication")


def test_layer_filter_retains_only_selected_layer_timing():
    result = stage_program(mock_sources(), Candidate(1, 1, 1, 1, ()), design(), 0, 0,
                           [[(0, 5, True)]], details=True, layer_index=1)
    assert {o["layer_index"] for o in result["operators"]} == {1}
    assert not any(o["id"] in {"embedding", "final_norm", "lm_head"} for o in result["operators"])
    # Four attention matrices (768 weights), three FFN matrices (1536 weights),
    # five input tokens plus causal attention (15 pairs, 4 heads, dim 4).
    assert result["flops"] == 2 * 5 * (768 + 1536) + 4 * 4 * 4 * 15
