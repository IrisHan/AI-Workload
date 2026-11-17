"""Event ledger checks independent of API publication and source adapters."""
from app.engines.inference.diagnostics import event_memory
from app.engines.inference.ir import Candidate
from app.engines.inference.memory import admission_memory
from app.engines.inference.scheduler import simulate
from test_inference_execution import mock_sources
from test_inference_partition import mock_ir
from test_inference_scheduler import offline


def test_event_components_and_final_cache_release():
    design = offline(output=3)
    sources = mock_sources()
    candidate = Candidate(1, 1, 1, 1, ())
    simulation = simulate(sources, candidate, design, details=True)
    result = event_memory(sources, candidate, design, simulation)
    rank = result["ranks"][0]
    assert result["coverage"] == "complete"
    assert not result["capacity_proof"]
    assert rank["peak_bytes"] == sum(rank["component_at_peak"].values())
    assert rank["component_maxima"]["kv"] == 1040  # 2 layers * (512 B payload + 8 B page index)
    assert rank["timeline"][-1]["components"]["kv"] == 0


def test_pipeline_and_moe_events_bounded_by_admission():
    for candidate, moe in [(Candidate(2, 2, 1, 1, (1,)), False), (Candidate(2, 1, 2, 1, ()), True)]:
        design = offline(count=4, concurrency=4, output=3)
        sources = mock_sources(ir=mock_ir(moe=moe))
        simulation = simulate(sources, candidate, design, details=True)
        event = event_memory(sources, candidate, design, simulation)
        admission = admission_memory(sources.ir, candidate, design, sources.hardware, sources.assumptions)
        for rank, bound in zip(event["ranks"], admission["ranks"]):
            assert rank["peak_bytes"] <= bound["peak_bytes"]


def test_budgeted_trace_never_claims_full_coverage():
    design = offline(count=5, concurrency=1)
    sources, candidate = mock_sources(), Candidate(1, 1, 1, 1, ())
    simulation = simulate(sources, candidate, design, details=True)
    result = event_memory(sources, candidate, design, simulation, allocation_budget=1)
    assert result["coverage"] == "prefix"
    assert result["until"] == 0
    assert result["ranks"][0]["component_maxima"]["activation"] == 0
