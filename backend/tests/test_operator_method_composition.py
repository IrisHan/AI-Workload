"""Independent small-shape golds and method-composition rejection contracts."""
import json
from pathlib import Path

import pytest
from app.engines.components import Ledger
from app.engines.layer_methods import (
    declared_layer_types,
    fill_hybrid_layer,
    layer_links,
)
from app.engines.model_structure import inspect_model
from app.engines.operator_methods import (
    compressed_attention,
    dsa_projections,
    feed_forward,
    gated_delta_projections,
    kda_projections,
    mla_projections,
    msa_projections,
    qk_norm,
    qkv_projections,
)
from app.lib.spec_studio.inventory import inventory_metrics


def metrics(ledger):
    item = inventory_metrics(ledger.rows)
    return item["params_total"], item["params_active"], item["matrix_params"]


@pytest.mark.parametrize("kv,expected", [(1, 640), (2, 768), (4, 1024)])
def test_mqa_gqa_mha_share_projection_method(kv, expected):
    ledger = Ledger("independent shape gold")
    qkv_projections(ledger, hidden=16, heads=4, kv_heads=kv, head_dim=4)
    assert metrics(ledger) == (expected, expected, expected)
    qk_norm(ledger, query_dim=16, key_dim=4 * kv)
    assert metrics(ledger) == (expected + 16 + 4 * kv, expected + 16 + 4 * kv, expected)


@pytest.mark.parametrize("rank,expected", [(4, 386), (None, 510)])
def test_mla_qkv_low_rank_and_direct_q(rank, expected):
    ledger = Ledger("independent shape gold")
    mla_projections(ledger, hidden=16, heads=4, q_rank=rank, kv_rank=6,
                    nope_dim=2, rope_dim=2, value_dim=3)
    assert metrics(ledger)[0] == expected
    rows = {r["id"]: r for r in ledger.rows}
    assert rows["kv_a_proj"]["shape"] == [8, 16]
    assert rows["kv_b_proj"]["shape"] == [20, 6]


def test_dsa_msa_are_different_methods():
    dsa, msa = Ledger("gold"), Ledger("gold")
    dsa_projections(dsa, hidden=16, q_rank=4, index_heads=2, index_dim=3)
    msa_projections(msa, hidden=16, index_heads=2, index_dim=3)
    assert metrics(dsa) == (110, 110, 104)
    assert metrics(msa) == (150, 150, 144)
    assert "indexer.weights_proj" not in {r["id"] for r in msa.rows}
    for invalid in (None, 0, True, "4"):
        with pytest.raises(ValueError, match="q_lora_rank"):
            dsa_projections(Ledger("gold"), hidden=16, q_rank=invalid,
                            index_heads=2, index_dim=3)


def test_swiglu_expert_multiplicity_and_gelu_are_distinct():
    ledger = Ledger("gold")
    feed_forward(ledger, 16, 8, count=4, active_count=2, bias=True)
    assert metrics(ledger) == (1664, 832, 768)
    ungated = Ledger("gold")
    feed_forward(ungated, 16, 8, gated=False, bias=True)
    assert metrics(ungated) == (280, 280, 256)


def test_recurrent_methods_keep_independent_key_value_dimensions():
    delta = Ledger("gold")
    gated_delta_projections(delta, hidden=16, key_heads=2, key_dim=3,
                            value_heads=4, value_dim=5, kernel=3)
    assert metrics(delta) == (1389, 1389, 1280)
    for full, total, matrix in ((False, 613, 548), (True, 643, 578)):
        kda = Ledger("gold")
        kda_projections(kda, hidden=16, heads=2, head_dim=3, kernel=3, full_rank_gate=full)
        assert metrics(kda) == (total, total, matrix)


@pytest.mark.parametrize("ratio,total", [(0, 300), (4, 786), (128, 944)])
def test_compressed_shared_kv_and_compressors(ratio, total):
    ledger = Ledger("gold", "fp8")
    compressed_attention(ledger, hidden=16, heads=4, head_dim=4, q_rank=4,
                         output_groups=2, output_rank=2, ratio=ratio,
                         index_heads=2, index_dim=2)
    assert metrics(ledger)[0] == total
    rows = {row["id"]: row for row in ledger.rows}
    assert rows["kv_proj"]["params"] == 64  # K=V is stored once.
    assert ("indexer.q_proj" in rows) == (ratio == 4)


@pytest.mark.parametrize("kda,full", [
    ([0, 0, 2], [1, 3]), (["0", 2], [1, 3]), ([False, 2], [1, 3]),
    ([0.0, 2], [1, 3]), ([0, 2], [2, 3]), ([0, 2], [1, 4]), ([0, 2], None),
])
def test_kda_partition_rejects_coercion_duplicates_overlap_and_gaps(kda, full):
    with pytest.raises(ValueError):
        declared_layer_types(dict(num_hidden_layers=4, linear_attn_config=dict(
            kda_layers=kda, full_attn_layers=full)))


def test_kda_partition_and_explicit_schedule_agree():
    cfg = dict(num_hidden_layers=4,
               linear_attn_config=dict(kda_layers=[0, 2], full_attn_layers=[1, 3]))
    assert declared_layer_types(cfg) == ["linear_attention", "full_attention"] * 2
    cfg["layer_types"] = ["linear_attention", "deepseek_sparse_attention"] * 2
    assert declared_layer_types(cfg) == cfg["layer_types"]
    cfg["layer_types"] = ["full_attention"] * 4
    with pytest.raises(ValueError, match="conflicts"):
        declared_layer_types(cfg)


@pytest.mark.parametrize("routing,has_gate", [("noaux_tc", False), ("greedy", True)])
def test_moe_graph_gate_matches_weight_inventory(routing, has_gate):
    cfg = dict(hidden_size=16, head_dim=4, num_attention_heads=4, num_key_value_heads=2,
               moe_intermediate_size=8, n_routed_experts=4, num_experts_per_tok=2,
               n_shared_experts=1, topk_method=routing)
    ledger = fill_hybrid_layer(cfg, "GQA_GATE", "MOE")
    names = {row["id"] for row in ledger.rows}
    nodes = {node for edge in layer_links("GQA_GATE", "MOE", cfg) for node in edge}
    assert ("shared_expert_gate" in names) is has_gate
    assert ("shared_expert_gate" in nodes) is has_gate


def test_missing_dsa_rank_is_value_error_at_model_boundary():
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / "docs/reports/2026-09-15-model-center-acceptance/evidence/catalog.json").read_text())
    cfg = next(m["raw_config"] for m in catalog if m["repo_id"] == "deepseek-ai/DeepSeek-V3.2-Exp-Base")
    cfg.pop("q_lora_rank")
    with pytest.raises(ValueError, match="q_lora_rank"):
        inspect_model(cfg)
