"""MC-07/08/09/10/11/12: method authorization and pinned graph/shape contracts."""
import json
from pathlib import Path

import pytest
from app.engines.layer_methods import ffn_method_for_index
from app.engines.model_structure import inspect_model

ROOT = Path(__file__).resolve().parents[2] / "docs/reports"


def load(group, repo):
    return json.loads((ROOT / group / "evidence/sources" / repo / "config.json").read_text())


def test_grouped_rejects_unimplemented_mhc_and_linear():
    base = dict(model_type="custom_model", hidden_size=16, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                intermediate_size=32, vocab_size=64, max_position_embeddings=128, hidden_act="silu")
    assert inspect_model(base)["status"] == "supported"
    with pytest.raises(ValueError, match="unsupported"):
        inspect_model(base | {"layer_types": ["linear_attention"] * 2})
    mla = dict(kv_lora_rank=8, q_lora_rank=8, qk_nope_head_dim=4, qk_rope_head_dim=2, v_head_dim=4)
    with pytest.raises(ValueError):
        inspect_model(base | mla | {"hc_mult": 4})
    with pytest.raises(ValueError):
        inspect_model(base | {"model_type": "qwen3", "hc_mult": 4})


def test_ffn_frequency_zero_based_distinct_from_sparse_step():
    cfg = dict(n_routed_experts=4, intermediate_size=32, first_k_dense_replace=1,
               routed_expert_hidden_size=8, moe_layer_freq=2)
    assert [ffn_method_for_index(i, cfg) for i in range(4)] == ["DENSE", "DENSE", "LATENT_MOE", "DENSE"]
    cfg["decoder_sparse_step"] = 2
    assert [ffn_method_for_index(i, cfg) for i in range(4)] == ["DENSE", "LATENT_MOE", "DENSE", "LATENT_MOE"]


def test_glm_downsample_matches_pinned_header_and_extras_connected():
    cfg = load("2026-09-15-frontier-families", "zai-org--GLM-5.3-Flash")
    analysis = inspect_model(cfg)
    vision = next(b for b in analysis["branches"] if b["role"] == "vision")
    row = next(r for r in vision["modules"] if r["id"] == "vision.downsample")
    assert row["shape"] == [4096, 1024, 2, 2]
    assert row["params"] == 16777216
    bias = next(r for r in vision["modules"] if r["id"] == "vision.downsample.bias")
    assert bias["params"] == 4096
    assert analysis["totals"]["total"] == 313873723203 + 16781312
    for t in analysis["templates"]:
        edges = {(e["source"], e["target"]) for e in t["edges"]}
        connected = {n for edge in edges for n in edge}
        extras = {r["id"] for r in t["modules"] if r["id"].startswith(("hc_", "indexer."))}
        assert extras <= connected
        if t["attn_kind"] == "DSA":
            assert ("indexer_top_k", "scale_mask") in edges
            assert ("indexer_key_pool", "indexer_scores") in edges
        assert ("hc_attn_pre", "attn_norm.weight") in edges
        assert ("hc_ffn_post", "output_add") in edges


def test_kimi_latent_sum_precedes_norm_and_shared_merge():
    analysis = inspect_model(load("2026-09-15-kimi-family", "moonshotai--Kimi-K3"))
    for t in analysis["templates"]:
        edges = {(e["source"], e["target"]) for e in t["edges"]}
        connected = {n for edge in edges for n in edge}
        extras = {r["id"] for r in t["modules"] if "_res_" in r["id"]}
        assert extras <= connected
        assert ("mlp_res_weighted_sum", "ffn_norm.weight") in edges
        if t["ffn_kind"] == "MOE":
            assert {
                ("routed_expert.down_proj", "weighted_expert_sum"),
                ("weighted_expert_sum", "routed_expert_norm.weight"),
                ("routed_expert_norm.weight", "routed_expert_up_proj"),
                ("routed_expert_up_proj", "expert_output_sum"),
                ("shared_expert.down_proj", "expert_output_sum"),
            } <= edges
            assert ("routed_expert_up_proj", "weighted_expert_sum") not in edges
