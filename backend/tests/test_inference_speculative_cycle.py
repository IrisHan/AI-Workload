"""Fixed-context cycle mathematical goldens, explicit mock provenance."""
from dataclasses import replace

import pytest
from app.engines.inference.ir import Candidate, PlanningError
from app.engines.inference.speculative import speculative_cycle
from test_inference_execution import mock_sources
from test_inference_partition import design


def cycle_input(**spec):
    return design(load={"prompt_tokens": 16, "output_tokens": 8, "concurrency": 2},
                  runtime={"max_batch_sequences": 2, "prefill_chunk_tokens": 8,
                           "max_batched_tokens": 32, "speculative": {
                               "enabled": True, "draft_tokens": 2, "acceptance_rate": .7,
                               "draft_latency_ms": 0, "draft_memory_bytes": 0, **spec}})


@pytest.mark.parametrize("acceptance,accepted,emitted", [(0, 0, 1), (.7, 1.19, 2.19), (1, 2, 3)])
def test_cycle_acceptance_and_all_verify_logits(acceptance, accepted, emitted):
    inp = cycle_input(acceptance_rate=acceptance)
    result = speculative_cycle(mock_sources(spec=inp), Candidate(1, 1, 1, 1, ()), inp)
    assert result["expected_accepted_tokens"] == pytest.approx(accepted)
    assert result["expected_emitted_tokens"] == pytest.approx(emitted)
    assert result["output_tokens"] == pytest.approx(2 * emitted)
    head = next(o for o in result["programs"][0]["operators"] if o["id"] == "lm_head")
    assert head["flops"] == 2 * (2 * 3) * 16 * 32
    assert result["tpot_ms"] == pytest.approx(result["cycle_seconds"] * 1000 / emitted)
    assert result["output_tokens_s"] == pytest.approx(2 * emitted / result["cycle_seconds"])
    assert result["capacity_status"] == "pass"


def test_cycle_missing_draft_cost_is_unknown():
    inp = cycle_input(draft_latency_ms=None, draft_memory_bytes=None)
    result = speculative_cycle(mock_sources(spec=inp), Candidate(1, 1, 1, 1, ()), inp)
    assert result["status"] == "unknown"
    assert result["cycle_seconds"] is None
    assert result["output_tokens_s"] is None
    assert result["memory"][0]["peak_bytes"] is None
    assert result["target_seconds"] > 0


def test_cycle_mixed_prefill_memory_and_latency_monotonic():
    c = Candidate(1, 1, 1, 1, ())
    plain, mixed = cycle_input(), cycle_input(include_prefill=True, draft_latency_ms=2, draft_memory_bytes=100)
    a = speculative_cycle(mock_sources(spec=plain), c, plain)
    b = speculative_cycle(mock_sources(spec=mixed), c, mixed)
    assert b["prefill_tokens"] == 8
    assert b["extra_prefill_request_per_group"]
    assert b["cycle_seconds"] > a["cycle_seconds"] + .002
    assert b["memory"][0]["peak_bytes"] > a["memory"][0]["peak_bytes"] + 100


def test_cycle_budget_and_context_fail_explicitly():
    inp = cycle_input(draft_tokens=20)
    with pytest.raises(PlanningError) as exc:
        speculative_cycle(mock_sources(spec=inp), Candidate(1, 1, 1, 1, ()), inp)
    assert exc.value.code == "SPEC_TOKEN_BUDGET"
    source = mock_sources(spec=inp)
    with pytest.raises(PlanningError) as exc:
        speculative_cycle(replace(source, ir=replace(source.ir, context_limit=32)),
                          Candidate(1, 1, 1, 1, ()), inp)
    assert exc.value.code == "SPEC_CONTEXT_LIMIT"
