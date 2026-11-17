"""Frozen acceptance case: integer byte goldens and independently derived timing."""
import json
from pathlib import Path

import pytest
from app.engines.inference.explain import inspection_explanation, memory_explanation, summary_explanation
from app.engines.inference.ir import Candidate
from app.engines.inference.search import evaluate
from app.workload.jobs import restore
from app.workload.schemas import DesignInput

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/reports/2026-10-08-calculation-explainability/evidence"


@pytest.fixture
def acceptance():
    exported = json.loads((EVIDENCE / "original-export.json").read_text())
    frozen = exported["frozen_input"]
    source = restore(frozen)
    spec = DesignInput.model_validate(frozen["design"])
    c = Candidate(**{**exported["plans"][0]["summary"]["candidate"], "boundaries": ()})
    plan = evaluate(source, c, spec, 1000000, details=True)
    return source, spec, c, plan


def test_memory_trace_conserves_exact_bytes_safety_and_source(acceptance):
    source, spec, c, plan = acceptance
    rank = plan["memory"]["ranks"][0]
    assert source.ir.logical_numel == 7615616512
    assert rank["components"] == dict(weight=7615820800, kv=249682944, activation=145752064,
                                      communication=29360128, runtime=1073741824,
                                      fragmentation=22357639, safety=8000000000)
    assert rank["peak_bytes"] == 17136715399
    assert rank["capacity_bytes"] == 80000000000
    assert rank["free_bytes"] == 62863284601
    e = memory_explanation(source, c, spec, rank)
    assert e["traces"]["kv"]["steps"] == [
        {"label": "payload_bytes", "value": 249561088, "unit": "bytes", "formula": "Σ local layers"},
        {"label": "metadata_bytes", "value": 121856, "unit": "bytes", "formula": "Σ local layers"},
        {"label": "tail_waste_bytes", "value": 0, "unit": "bytes", "formula": "Σ local layers"}]
    assert e["capacity_gb"] == 80
    assert e["utilization"] == pytest.approx(17136715399 / 80000000000)
    assert e["traces"]["safety"]["inputs"][0]["source"] == "Hardware"
    assert e["traces"]["safety"]["assumptions"][0]["source"] == "Assumption"


def test_summary_reconciles_old_metrics_and_new_activity(acceptance):
    source, spec, c, plan = acceptance
    m = plan["performance"]["metrics"]
    e = summary_explanation(source, c, spec, plan)
    assert e["traces"]["source_weight"]["result"]["value"] == 15231233024
    assert m["iteration_time_ms"] == pytest.approx(8.224989771797155)
    assert m["itl_ms"] == pytest.approx(46.22984620279624)
    assert m["ttft_ms"] == pytest.approx(5425.982680395114)
    assert m["output_tokens_s"] == 72.8
    assert m["achieved_qps"] == pytest.approx(16 / 30)
    assert m["modeled_busy_fraction"] == pytest.approx(.6265618140609824)
    assert m["gpu_activity_fraction"] == pytest.approx(1)
    p = e["performance"]
    assert p["completed_full_outputs"] == 2048
    assert p["unfinished_outputs_in_window"] == 136
    assert p["itl_samples"] == p["decode_wave_count"] == 2165
    assert sum(t["result"]["value"] for t in p["breakdown"]) == pytest.approx(m["iteration_time_ms"])
    assert p["itl_wave_service_ms"] + p["itl_between_wave_wait_ms"] == pytest.approx(m["itl_ms"])
    assert p["bottleneck"] == "Memory Bound"
    assert next(f for f in e["topology"] if f["key"] == "nic_spec")["value"] == 25
    assert any(f["source"] == "Override" for f in e["topology"])


def test_operator_shape_latency_lifetime_and_non_weighted_mapping(acceptance):
    source, spec, c, _ = acceptance
    result = json.loads((EVIDENCE / "original-inspection.json").read_text())
    e = inspection_explanation(source, c, spec, result)
    gate = e["nodes"]["gate_proj"]
    assert gate["parameters"] == 33947648
    assert gate["weight_bytes"] == 67895296
    assert gate["full_shape"] == [18944, 3584]
    assert gate["local_shape"] == [9472, 3584]
    assert {f["key"]: f["value"] for f in gate["shapes"]} == dict(
        input=[2048, 3584], weight=[9472, 3584], output=[2048, 9472], M=2048, K=3584, N=9472)
    op = gate["operators"][0]
    assert op["flops"] == 2 * 2048 * 3584 * 9472
    compute = 139049566208 / (312e12 * .5)
    memory = 121372672 / (2039e9 * .7)
    assert op["seconds"] == pytest.approx(max(compute, memory) + .000005)
    assert gate["traces"][1]["result"]["value"] == op["seconds"]
    assert [o["id"] for o in e["nodes"]["silu"]["operators"]] == ["ffn_activation"]
    assert e["nodes"]["input"]["operators"] == []
    assert "Shared aggregate" in e["nodes"]["multiply"]["attribution"]
    assert e["steps"][5]["bytes"] == 33292288
    assert max(s["bytes"] for s in e["steps"]) == 145752064
