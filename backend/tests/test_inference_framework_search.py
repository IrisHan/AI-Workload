"""Framework search uses real device budgets and rejects speculative OOM."""
import pytest
from app.engines.inference.ir import Candidate, PlanningError
from app.engines.inference.search import candidate_stream, evaluate, parallel_domains
from test_inference_execution import mock_sources
from test_inference_partition import design, mock_ir


def test_framework_search_default_pp_and_derived_ep():
    ir = mock_ir(layers=28, moe=True)
    inp = design(parallelism={"layout": "vllm-v0.10.2", "expert_parallel": True,
                             "allowed_tp": [2], "allowed_pp": [3]})
    candidates = list(candidate_stream(ir, inp))
    assert len(candidates) == 1
    candidate, exhaustive = candidates[0]
    assert (candidate.tp, candidate.pp, candidate.ep, candidate.dp, candidate.boundaries) == (2, 3, 2, 1, (9, 19))
    assert candidate.gpu_count == 6
    assert exhaustive
    assert parallel_domains(ir, layout="sglang-v0.5.3")["tp"] == [1, 2, 4]


def test_evaluate_rejects_draft_memory_oom():
    # BUG_MAP: target baseline fits, draft memory alone exceeds the 16 GiB mock GPU.
    inp = design(mode="offline", load={"request_count": 1, "request_rate": None, "arrival": None,
                                      "concurrency": 1, "prompt_tokens": 16, "output_tokens": 2},
                 runtime={"max_batch_sequences": 1, "speculative": {
                     "enabled": True, "draft_tokens": 2, "acceptance_rate": .7,
                     "draft_latency_ms": 0, "draft_memory_bytes": 20 * 1024**3}})
    with pytest.raises(PlanningError) as exc:
        evaluate(mock_sources(spec=inp), Candidate(1, 1, 1, 1, ()), inp, 1000)
    assert exc.value.code == "SPEC_CAPACITY"
