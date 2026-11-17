"""Frozen inventories survive versioned template signatures."""
import hashlib
import json
from pathlib import Path

import pytest
from app.engines.layer_methods import ATTN_GATED_DELTA, ATTN_GQA_GATE, FFN_DENSE, schedule_layers
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "docs/reports/2026-09-14-qwen-family/evidence/baselines"
MC_SOURCES = ROOT / "docs/reports/2026-09-13-model-center/evidence/sources"
QWEN_SOURCES = ROOT / "docs/reports/2026-09-14-qwen-family/evidence/sources"

CONFIGS = {
    "DeepSeek-V4-Flash-Base": MC_SOURCES / "deepseek-ai--DeepSeek-V4-Flash-Base/config.json",
    "Qwen3.8-27B": MC_SOURCES / "Qwen--Qwen3.8-27B/config.json",
    "Qwen2.5-0.5B": QWEN_SOURCES / "Qwen--Qwen2.5-0.5B/config.json",
}


def load_baseline(name):
    return json.loads((BASELINES / f"{name}.pre-framework.json").read_text())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_pre_framework_totals_and_templates_unchanged(name):
    cfg = json.loads(CONFIGS[name].read_text())
    gold = load_baseline(name)
    result = inspect_model(cfg)
    assert result["totals"]["total"] == gold["analyze"]["parameters"] == gold["structure"]["totals"]["total"]
    for key in ("active", "active_body", "embed", "lm_head", "attn", "ffn_total", "ffn_active", "other", "mtp"):
        assert result["totals"][key] == gold["structure"]["totals"][key]
    if "vision" in gold["structure"]["totals"]:
        assert result["totals"]["vision"] == gold["structure"]["totals"]["vision"]
    want_t = gold["structure"]["templates"]
    normalized = []
    template_ids = {}
    for t, previous in zip(result["templates"], want_t, strict=True):
        if name != "Qwen2.5-0.5B":
            rows = [{k: v for k, v in row.items() if k != "checkpoint_paths"} for row in t["modules"]]
            def signature(version):
                payload = dict(rows=rows, attention=t["attn_kind"], ffn=t["ffn_kind"], formula=version)
                return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            # P1 intentionally bumps component formulas; the old hash must still
            # reproduce from unchanged inventory, and the new hash must use 1.1.
            assert signature("components-1.0.0") == previous["signature"]
            assert signature("components-1.1.0") == t["signature"]
            assert t["id"] == "t-" + t["signature"][:12]
        else:
            assert (t["id"], t["signature"]) == (previous["id"], previous["signature"])
        template_ids[t["id"]] = previous["id"]
        normalized.append((previous["id"], t["label"], t["attn_kind"], t["ffn_kind"], previous["signature"]))
    assert normalized == [
        (t["id"], t["label"], t["attn_kind"], t["ffn_kind"], t["signature"]) for t in want_t
    ]
    want_g = gold["structure"]["groups"]
    assert [(g["id"], template_ids[g["template_id"]], g["count"], g["ranges"]) for g in result["groups"]] == [
        (g["id"], g["template_id"], g["count"], g["ranges"]) for g in want_g
    ]


def test_v4_analyze_breakdown_matches_live_baseline():
    gold = load_baseline("DeepSeek-V4-Flash-Base")
    cfg = json.loads(CONFIGS["DeepSeek-V4-Flash-Base"].read_text())
    result = inspect_model(cfg)
    from app.engines.model_structure import breakdown
    assert breakdown(result) == gold["analyze"]["parameter_breakdown"]


def test_v4_static_memory_matches_live_baseline():
    gold = load_baseline("DeepSeek-V4-Flash-Base")
    cfg = json.loads(CONFIGS["DeepSeek-V4-Flash-Base"].read_text())
    result = inspect_model(cfg)
    mem = memory_components(result, cfg, batch_size=1, context_length=4096, dtype="bf16")
    assert mem["weight_bytes"] == gold["memory"]["weight_bytes"]
    assert mem["kv_cache_bytes"] == gold["memory"]["kv_cache_bytes"]
    assert mem["integer_buffer_bytes"] == 3 * 129280 * 6 * 4
    assert mem["static_total_bytes"] == gold["memory"]["static_total_bytes"] + mem["integer_buffer_bytes"]
    assert [row["total_bytes"] for row in mem["cache_components"]] == gold["memory"]["cache_component_totals"]


def test_interval_schedule_is_three_linear_then_full():
    cfg = {
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
        "layer_types": None,
    }
    del cfg["layer_types"]
    kinds = [item[2] for item in schedule_layers(cfg)]
    assert kinds == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ]
    methods = [item[0] for item in schedule_layers(cfg)]
    assert methods == [
        ATTN_GATED_DELTA, ATTN_GATED_DELTA, ATTN_GATED_DELTA, ATTN_GQA_GATE,
        ATTN_GATED_DELTA, ATTN_GATED_DELTA, ATTN_GATED_DELTA, ATTN_GQA_GATE,
    ]
    assert {item[1] for item in schedule_layers(cfg)} == {FFN_DENSE}
