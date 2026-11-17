"""Frontier families: shared layer methods, not per-repo formulas."""
import json
from pathlib import Path

import pytest
from app.engines.model_memory import memory_components
from app.engines.model_structure import inspect_model
from app.lib.spec_studio.inspect import inspect_config

FRONTIER = Path(__file__).resolve().parents[2] / "docs/reports/2026-09-15-frontier-families/evidence/sources"


def cfg(name):
    return json.loads((FRONTIER / name / "config.json").read_text())


def _group_counts(result):
    counts = {}
    for group in result["groups"]:
        template = next(t for t in result["templates"] if t["id"] == group["template_id"])
        key = (template["attn_kind"], template["ffn_kind"])
        counts[key] = counts.get(key, 0) + group["count"]
    return counts


def test_v4_grouped_path_still_rejected():
    stub = {
        "hidden_size": 4096, "num_hidden_layers": 4, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 11008, "vocab_size": 32000,
        "model_type": "deepseek_v4", "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "compress_ratios": [0, 0, 4, 128], "hc_mult": 4, "index_topk": 512,
    }
    with pytest.raises(ValueError, match="unsupported model_type"):
        inspect_config(stub, name="v4-stub")


def test_minimax_m2_gqa_moe_fused_qk_norm():
    result = inspect_model(cfg("MiniMaxAI--MiniMax-M2.7"))
    assert _group_counts(result) == {("GQA", "MOE"): 62}
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_norm" in names
    assert "router_correction" in names
    assert result["totals"]["total"] < 2 ** 53
    twin = inspect_model(cfg("MiniMaxAI--MiniMax-M2.5"))
    assert twin["totals"]["total"] == result["totals"]["total"]
    mem = memory_components(result, cfg("MiniMaxAI--MiniMax-M2.7"))
    assert mem["weight_bytes"] <= result["totals"]["total"] * 2


def test_minimax_m3_msa_moe_clip():
    result = inspect_model(cfg("MiniMaxAI--MiniMax-M3"))
    counts = _group_counts(result)
    assert counts[("GQA", "DENSE")] == 3
    assert counts[("MSA", "MOE")] == 57
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "indexer.q_proj" in names
    assert "indexer.weights_proj" not in names
    assert result["totals"]["vision"]
    assert result["morphology"]["vision"] == "CLIP"
    assert "MSA" in result["morphology"]["attn"]
    assert "GQA" in result["morphology"]["attn"]
    mx = inspect_model(cfg("MiniMaxAI--MiniMax-M3-MXFP8"))
    assert mx["totals"]["total"] == result["totals"]["total"]
    assert mx["morphology"]["quant"] == "fp8"


def test_gpt_oss_swa_gqa_moe_sinks_mxfp4():
    large = inspect_model(cfg("openai--gpt-oss-120b"))
    small = inspect_model(cfg("openai--gpt-oss-20b"))
    assert _group_counts(large) == {("SWA", "MOE"): 18, ("GQA", "MOE"): 18}
    assert _group_counts(small) == {("SWA", "MOE"): 12, ("GQA", "MOE"): 12}
    names = {row["id"] for t in large["templates"] for row in t["modules"]}
    assert "attn_sink" in names
    assert "routed_expert.gate_up_bias" in names
    assert large["morphology"]["quant"] == "mxfp4-group"
    assert small["totals"]["total"] < large["totals"]["total"]


def test_glm_indexshare_not_per_layer_indexer():
    share = inspect_model(cfg("zai-org--GLM-5.2"))
    full = inspect_model(cfg("zai-org--GLM-5.1"))
    share_idx = sum(
        group["count"]
        for group in share["groups"]
        for item in share["templates"]
        if item["id"] == group["template_id"]
        and any(row["id"] == "indexer.weights_proj" for row in item["modules"])
    )
    full_idx = sum(
        group["count"]
        for group in full["groups"]
        for item in full["templates"]
        if item["id"] == group["template_id"]
        and any(row["id"] == "indexer.weights_proj" for row in item["modules"])
    )
    assert share_idx < full_idx
    assert any(t["attn_kind"] == "DSA" for t in share["templates"])
    fp8 = inspect_model(cfg("zai-org--GLM-5.2-FP8"))
    assert fp8["totals"]["total"] == share["totals"]["total"]
    glm53 = inspect_model(cfg("zai-org--GLM-5.3"))
    assert glm53["totals"]["total"] == share["totals"]["total"]


def test_glm_flash_kda_dsa_mhc_vision():
    result = inspect_model(cfg("zai-org--GLM-5.3-Flash"))
    counts = _group_counts(result)
    assert counts[("KDA", "DENSE")] == 3
    assert counts[("KDA", "MOE")] == 31
    assert counts[("DSA", "MOE")] == 11
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "q_conv1d" in names
    assert "indexer.weights_proj" in names
    assert "indexer.index_kpool_compress_ape" in names
    assert "hc_attn.fn" in names
    assert result["totals"]["vision"]
    assert result["morphology"]["vision"] == "QwenViT"


def test_ernie_moe_aliases():
    result = inspect_model(cfg("baidu--ERNIE-4.5-21B-A3B-Thinking"))
    assert _group_counts(result) == {("GQA", "DENSE"): 1, ("GQA", "MOE"): 27}
    names = {row["id"] for t in result["templates"] for row in t["modules"]}
    assert "routed_expert.gate_proj" in names
    assert "shared_expert.gate_proj" in names
    assert "router_correction" in names
    assert result["totals"]["total"] == 21825437888


def test_qianfan_qwen3_internvit():
    result = inspect_model(cfg("baidu--Qianfan-OCR"))
    assert all(t["attn_kind"] == "GQA" and t["ffn_kind"] == "DENSE" for t in result["templates"])
    vision_ids = {row["id"] for b in result["branches"] for row in b["modules"]}
    assert "vision.class_embedding" in vision_ids
    assert "vision.mlp1.fc1" in vision_ids
    assert result["morphology"]["vision"] == "InternViT"


def test_no_per_repo_inspect_functions():
    import app.engines.composite_model as cm
    import app.engines.model_structure as ms
    for module in (ms, cm):
        for name in dir(module):
            assert "minimax" not in name.lower()
            assert "gptoss" not in name.lower()
            assert not name.lower().startswith("inspect_glm")
