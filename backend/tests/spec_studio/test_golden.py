import pytest
from app.lib.spec_studio.inspect import inspect_config
from app.lib.spec_studio.ir import AttentionKind, FfnKind, NormKind


def test_llama_gqa_dense(llama_cfg):
    r = inspect_config(llama_cfg, name="llama-3.2-1b")
    arch = r.arch
    assert arch.attention_kind == AttentionKind.GQA
    assert arch.ffn_kind == FfnKind.DENSE
    assert arch.backbone.tied_embeddings is True
    assert all(layer.ffn_kind == FfnKind.DENSE for layer in arch.layers)
    ids = {c.id for c in arch.capabilities}
    assert "gqa" in ids
    assert "dense-ffn" in ids
    assert "rmsnorm" in ids
    assert "moe" not in ids
    total = arch.totals.total
    # Meta card: 1.23B
    assert abs(total - 1.23e9) / 1.23e9 < 0.01


def test_qwen_gqa_no_swa_when_disabled(qwen_cfg):
    r = inspect_config(qwen_cfg, name="qwen2.5-1.5b")
    arch = r.arch
    assert arch.attention_kind == AttentionKind.GQA
    assert arch.backbone.sliding_window is None
    assert "swa" not in {c.id for c in arch.capabilities}
    assert arch.backbone.intermediate_size == 8960
    assert abs(arch.totals.total - 1.54e9) / 1.54e9 < 0.01


def test_deepseek_v3_mla_moe(v3_cfg):
    r = inspect_config(v3_cfg, name="deepseek-v3")
    arch = r.arch
    assert arch.attention_kind == AttentionKind.MLA
    assert arch.ffn_kind == FfnKind.MOE
    assert arch.mla is not None
    assert arch.moe is not None
    assert arch.moe.n_routed == 256
    assert arch.moe.top_k == 8
    assert arch.moe.dense_prefix == 3
    assert sum(1 for layer in arch.layers if layer.ffn_kind == FfnKind.DENSE) == 3
    assert sum(1 for layer in arch.layers if layer.ffn_kind == FfnKind.MOE) == 58
    ids = {c.id for c in arch.capabilities}
    assert "mla" in ids
    assert "moe" in ids
    assert "yarn" in ids
    assert "mtp" in ids
    # Card ~671B / ~37B; MTP excluded by default
    assert abs(arch.totals.total - 671e9) / 671e9 < 0.03
    assert abs(arch.totals.active - 37e9) / 37e9 < 0.08
    assert arch.totals.mtp == 0
    assert arch.serving.nparams_total == arch.totals.total
    assert arch.serving.flop_gemm_decode > 0
    assert arch.serving.kv_layout.kind == "MLA"


def test_v3_mtp_toggle(v3_cfg):
    off = inspect_config(v3_cfg, include_mtp=False).arch
    on = inspect_config(v3_cfg, include_mtp=True).arch
    assert on.totals.total > off.totals.total
    assert on.totals.mtp > 0


def test_mixtral_is_moe_not_dense(mixtral_cfg):
    r = inspect_config(mixtral_cfg, name="mixtral-8x7b")
    arch = r.arch
    assert arch.attention_kind == AttentionKind.GQA
    assert arch.ffn_kind == FfnKind.MOE
    assert arch.moe is not None
    assert arch.moe.n_routed == 8
    assert arch.moe.top_k == 2
    assert all(layer.ffn_kind == FfnKind.MOE for layer in arch.layers)
    assert abs(arch.totals.total - 46.7e9) / 46.7e9 < 0.02


def test_v4_flags_unsupported():
    cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "model_type": "deepseek_v4",
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "compress_ratios": [0, 0, 4, 128],
        "hc_mult": 4,
        "index_topk": 512,
    }
    with pytest.raises(ValueError, match="unsupported model_type"):
        inspect_config(cfg, name="v4-stub")


def test_graph_has_embed_layers_head(llama_cfg):
    r = inspect_config(llama_cfg, name="llama-3.2-1b")
    kinds = [n.kind for n in r.graph.nodes if n.parent is None]
    assert kinds[0] == "embed"
    assert kinds[-1] == "head"
    assert sum(1 for n in r.graph.nodes if n.kind == "layer") == 16


def test_calcs_have_required_ids(llama_cfg):
    arch = inspect_config(llama_cfg).arch
    ids = {c.calc_id for c in arch.calcs}
    assert "param.total" in ids
    assert "param.active" in ids
    assert "flop.decode_gemm" in ids
    assert "mem.weight_bytes" in ids


QWEN1_TOTAL = 72285954048
QWEN1_DISK = 41277997056
V3_FAKE_INDEX = 1369062772000
V3_HIST = {"BF16": 3918786560, "F8_E4M3": 680571043840, "F32": 41555600}
V3_HIST_BYTES = 688574839360


def test_qwen1_swiglu_half_intermediate(qwen1_72b_cfg):
    r = inspect_config(
        qwen1_72b_cfg,
        name="qwen-72b-chat",
        safetensors_index={"metadata": {"total_size": QWEN1_DISK}},
    )
    arch = r.arch
    assert arch.attention_kind == AttentionKind.MHA
    assert arch.norm_kind == NormKind.RMSNORM
    assert arch.backbone.intermediate_size == 24576
    assert arch.backbone.activation.lower() in {"silu", "swiglu"}
    assert arch.backbone.head_dim == 128
    assert arch.totals.total == QWEN1_TOTAL
    assert arch.serving.weight_bytes == QWEN1_DISK
    ids = {c.id: c for c in arch.capabilities}
    assert "quant" in ids
    assert "gptq" in ids["quant"].label
    assert "layernorm" not in ids
    unread = {f.key: f.reason for f in arch.unread_fields}
    assert "kv_channels" not in unread or unread.get("kv_channels") == "synonym"
    assert "no_bias" not in unread
    assert not any("328" in w for w in arch.warnings)
    assert not arch.errors


def test_v3_fake_index_is_info_not_error(v3_cfg):
    r = inspect_config(
        v3_cfg,
        name="deepseek-v3",
        safetensors_index={"metadata": {"total_size": V3_FAKE_INDEX}},
        hub_safetensors=V3_HIST,
        hub_used_storage=688727648088,
    )
    arch = r.arch
    assert arch.totals.total == 671026419200
    assert arch.serving.weight_bytes == V3_HIST_BYTES
    assert not any("51.0%" in w for w in arch.warnings)
    assert any("BF16-inflated" in msg for msg in arch.infos)
    assert not arch.errors


def test_v3_nparams_unchanged_without_hub(v3_cfg):
    off = inspect_config(v3_cfg).arch
    fake = inspect_config(v3_cfg, safetensors_index={"metadata": {"total_size": V3_FAKE_INDEX}}).arch
    assert off.totals.total == fake.totals.total
    assert any("MTP" in msg for msg in fake.infos)
    assert not any("51.0%" in w for w in fake.warnings)
