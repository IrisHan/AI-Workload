"""Critical path and activity goldens independent of UI formatting."""
import pytest
from app.engines.inference.compute import stage_program
from app.engines.inference.ir import Candidate
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir


@pytest.mark.parametrize("candidate,batches", [
    (Candidate(2, 1, 1, 1, ()), [[(0, 4, True)]]),
    (Candidate(2, 1, 4, 2, (), "vllm-v0.10.2", True), [[(0, 1, True)], [(0, 3, True)]]),
    (Candidate(4, 1, 2, 1, (), "sglang-v0.5.3", True), [[(0, 4, True)]]),
])
def test_selected_branch_components_conserve_wall_time(candidate, batches):
    source = mock_sources(ir=mock_ir(layers=1, moe=candidate.moe))
    result = stage_program(source, candidate, design(), 0, 0, batches, details=True)
    for segment in result["segments"]:
        assert sum(segment["components"].values()) == pytest.approx(segment["seconds"])
        assert all(v <= segment["seconds"] + 1e-12 for v in segment["rank_activity_seconds"].values())
    assert sum(sum(s["components"].values()) for s in result["segments"]) == pytest.approx(result["seconds"])
    assert sum(s["components"]["ep"] for s in result["segments"]) > 0 if candidate.moe else True


def test_pp_activity_excludes_empty_group_endpoints():
    # defect-probing: only group 1 has tokens; PP transfers 1 -> 3, never 0 -> 2.
    c = Candidate(1, 2, 2, 1, (1,))
    result = stage_program(mock_sources(ir=mock_ir(moe=True)), c, design(), 0, 0,
                           [[], [(0, 1, True)]])
    pp = [s for s in result["segments"] if s["components"]["pp"]]
    assert len(pp) == 1
    assert set(pp[0]["rank_activity_seconds"]) == {1, 3}
