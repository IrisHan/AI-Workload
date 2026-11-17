"""Per-layer attention and FFN methods. Callers assemble; repos do not own formulas."""
from app.engines.components import Ledger
from app.engines.operator_methods import (
    dsa_projections,
    gated_delta_projections,
    gated_qkv_projections,
    kda_projections,
    mla_projections,
)
from app.engines.vision_components import QWEN_SOURCE
from app.lib.spec_studio.validation import shared_expert_multiplicity

ATTN_GQA_GATE = "GQA_GATE"
ATTN_GATED_DELTA = "GATED_DELTA"
ATTN_MLA = "MLA"
ATTN_KDA = "KDA"
FFN_DENSE = "DENSE"
FFN_MOE = "MOE"
FFN_LATENT_MOE = "LATENT_MOE"
KIMI_SOURCE = (
    "https://huggingface.co/moonshotai/Kimi-K3/blob/"
    "f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py"
)
INDEXER_KEYS = (
    "indexer_budget", "indexer_head_dim", "indexer_n_heads",
    "indexer_kv_heads", "indexer_compress_ratio",
)
NGRAM_KEYS = (
    "ngram_size", "ngram_vocab_size_base", "heads_per_ngram",
    "ple_embed_dim", "ple_conv_kernel_size", "ple_layer_ids",
)


def hybrid_language_config(cfg):
    inner = cfg.get("text_config")
    return inner if isinstance(inner, dict) else cfg


def is_hybrid_language(cfg):
    inner = hybrid_language_config(cfg)
    if inner.get("linear_num_key_heads") is None:
        return False
    return isinstance(inner.get("layer_types"), list) or type(inner.get("full_attention_interval")) is int


def linear_attn_config(cfg):
    inner = hybrid_language_config(cfg)
    spec = inner.get("linear_attn_config")
    return spec if isinstance(spec, dict) else None


def is_kda_language(cfg):
    spec = linear_attn_config(cfg)
    return isinstance(spec, dict) and isinstance(spec.get("kda_layers"), list)


def unimplemented_hybrid_modules(cfg):
    inner = hybrid_language_config(cfg)
    reasons = []
    if any(inner.get(key) for key in INDEXER_KEYS):
        reasons.append(
            "attn: QSA indexer on full_attention "
            "(indexer_budget/compress_ratio/head_dim/kv_heads/n_heads); "
            "cannot reuse DeepSeek DSA constants"
        )
    if any(inner.get(key) for key in NGRAM_KEYS):
        reasons.append("embed: n-gram/PLE branch (ngram_size/ngram_vocab_size_base/ple_*)")
    if inner.get("hc_count") or inner.get("hc_lowrank"):
        reasons.append("residual: hc_count/hc_lowrank (not V4 hc_mult)")
    if inner.get("hc_mult"):
        reasons.append("residual: hc_mult is not implemented by the Gated DeltaNet adapter")
    if reasons:
        raise ValueError(
            "unsupported layer variants: " + "; ".join(reasons) + "; no reliable parameter total"
        )


def unimplemented_linear_modules(cfg):
    inner = hybrid_language_config(cfg)
    if is_kda_language(cfg):
        return
    reasons = []
    if isinstance(inner.get("linear_attn_config"), dict):
        reasons.append("attn: linear_attn_config without kda_layers/full_attn_layers partition")
    if inner.get("routed_expert_hidden_size"):
        reasons.append("ffn: LatentMoE routed_expert_hidden_size")
    if inner.get("attn_res_block_size"):
        reasons.append("residual: attn_res_block_size")
    if reasons:
        raise ValueError(
            "unsupported layer variants: " + "; ".join(reasons) + "; no reliable parameter total"
        )


def attention_method_for_type(layer_type, cfg):
    if layer_type in ("full_attention", "deepseek_sparse_attention"):
        if any(cfg.get(key) for key in INDEXER_KEYS):
            raise ValueError("QSA indexer on full_attention is not implemented")
        if cfg.get("kv_lora_rank") is not None:
            return ATTN_MLA
        return ATTN_GQA_GATE
    if layer_type == "linear_attention":
        if linear_attn_config(cfg):
            return ATTN_KDA
        return ATTN_GATED_DELTA
    raise ValueError("unsupported hybrid attention type")


def kda_partition_types(spec, layers):
    """Validate a complete zero- or one-based partition without coercing indices."""
    raw_kda, raw_full = spec.get("kda_layers"), spec.get("full_attn_layers")
    if not isinstance(raw_kda, list) or not isinstance(raw_full, list):
        raise ValueError("kda_layers and full_attn_layers must be lists")
    combined = raw_kda + raw_full
    if any(type(value) is not int for value in combined):
        raise ValueError("KDA layer indices must be strict integers")
    zero_based = 0 in combined
    start = 0 if zero_based else 1
    if len(combined) != layers or set(combined) != set(range(start, start + layers)):
        raise ValueError("kda_layers and full_attn_layers must partition num_hidden_layers")
    kda = set(raw_kda)
    return ["linear_attention" if index in kda else "full_attention"
            for index in range(start, start + layers)]


def declared_layer_types(cfg):
    types = cfg.get("layer_types")
    layers = cfg["num_hidden_layers"]
    spec = cfg.get("linear_attn_config")
    partition = kda_partition_types(spec, layers) if isinstance(spec, dict) and "kda_layers" in spec else None
    if isinstance(types, list):
        if len(types) != layers:
            raise ValueError("hybrid layer_types length mismatch")
        normalized = ["full_attention" if item == "deepseek_sparse_attention" else item for item in types]
        if partition is not None and normalized != partition:
            raise ValueError("layer_types conflicts with KDA/full attention partition")
        return list(types)
    interval = cfg.get("full_attention_interval")
    if type(interval) is int and interval > 0:
        return [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layers)
        ]
    if partition is not None:
        return partition
    raise ValueError("hybrid layer_types length mismatch")


def has_routed_experts(cfg):
    n = (
        cfg.get("num_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_local_experts")
        or cfg.get("moe_num_experts")
    )
    return type(n) is int and n > 1


def shared_expert_count(cfg):
    expert_dim = cfg.get("moe_intermediate_size")
    counted = cfg.get("n_shared_experts", cfg.get("num_shared_experts", cfg.get("moe_num_shared_experts", 0)))
    has_width = any(key in cfg for key in ("shared_expert_intermediate_size", "shared_intermediate_size"))
    width = None
    if has_width:
        width = cfg.get("shared_expert_intermediate_size", cfg.get("shared_intermediate_size"))
    return shared_expert_multiplicity(width, expert_dim, counted)


def routed_expert_count(cfg):
    return (
        cfg.get("num_experts")
        or cfg.get("n_routed_experts")
        or cfg.get("num_local_experts")
        or cfg.get("moe_num_experts")
    )


def top_k_experts(cfg):
    return (
        cfg.get("num_experts_per_tok")
        or cfg.get("num_experts_per_token")
        or cfg.get("moe_k")
        or cfg.get("moe_topk")
        or cfg.get("moe_top_k")
    )


def ffn_method_for_index(index, cfg):
    if not has_routed_experts(cfg):
        return FFN_DENSE
    mlp_types = cfg.get("mlp_layer_types")
    if isinstance(mlp_types, list) and index < len(mlp_types) and str(mlp_types[index]) == "dense":
        if type(cfg.get("intermediate_size")) is not int:
            raise ValueError("Dense FFN layer requires intermediate_size")
        return FFN_DENSE
    prefix = cfg.get("first_k_dense_replace") or cfg.get("moe_layer_start_index") or 0
    if type(prefix) is int and index < prefix:
        if type(cfg.get("intermediate_size")) is not int:
            raise ValueError("Dense FFN layer requires intermediate_size")
        return FFN_DENSE
    mlp_only = cfg.get("mlp_only_layers") or []
    if not isinstance(mlp_only, list):
        raise ValueError("mlp_only_layers must be a list")
    if index in mlp_only:
        if type(cfg.get("intermediate_size")) is not int:
            raise ValueError("Dense FFN layer requires intermediate_size")
        return FFN_DENSE
    step = cfg.get("decoder_sparse_step")
    schedule_index = index + 1
    if step is None:
        step = cfg.get("moe_layer_freq", 1)
        schedule_index = index
    if isinstance(step, list):
        if index < len(step) and not step[index]:
            if type(cfg.get("intermediate_size")) is not int:
                raise ValueError("Dense FFN layer requires intermediate_size")
            return FFN_DENSE
        if cfg.get("routed_expert_hidden_size"):
            return FFN_LATENT_MOE
        return FFN_MOE
    if type(step) is int and step > 1 and schedule_index % step != 0:
        if type(cfg.get("intermediate_size")) is not int:
            raise ValueError("Dense FFN layer requires intermediate_size")
        return FFN_DENSE
    if cfg.get("routed_expert_hidden_size"):
        return FFN_LATENT_MOE
    return FFN_MOE


def schedule_layers(cfg):
    types = declared_layer_types(cfg)
    return [
        (attention_method_for_type(kind, cfg), ffn_method_for_index(index, cfg), kind)
        for index, kind in enumerate(types)
    ]


def apply_gqa_gate(ledger, cfg):
    gated_qkv_projections(ledger, hidden=cfg["hidden_size"], heads=cfg["num_attention_heads"],
                          kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"])


def apply_gated_delta(ledger, cfg):
    gated_delta_projections(
        ledger, hidden=cfg["hidden_size"], key_heads=cfg["linear_num_key_heads"],
        key_dim=cfg["linear_key_head_dim"], value_heads=cfg["linear_num_value_heads"],
        value_dim=cfg["linear_value_head_dim"], kernel=cfg["linear_conv_kernel_dim"],
    )


def apply_dense_swiglu(ledger, cfg):
    ledger.ffn(cfg["hidden_size"], cfg["intermediate_size"])


def apply_mla(ledger, cfg):
    d, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    v = cfg["v_head_dim"]
    mla_projections(ledger, hidden=d, heads=heads, q_rank=cfg.get("q_lora_rank"),
                    kv_rank=cfg["kv_lora_rank"], nope_dim=cfg["qk_nope_head_dim"],
                    rope_dim=cfg["qk_rope_head_dim"], value_dim=v)
    if cfg.get("mla_use_output_gate"):
        ledger.linear("g_proj", d, heads * v)
    ledger.linear("o_proj", heads * v, d)


def apply_kda(ledger, cfg):
    spec = cfg["linear_attn_config"]
    kda_projections(ledger, hidden=cfg["hidden_size"], heads=spec["num_heads"],
                    head_dim=spec["head_dim"], kernel=spec["short_conv_kernel_size"],
                    full_rank_gate=spec.get("use_full_rank_gate", False))


def apply_dsa_indexer(ledger, cfg):
    d = cfg["hidden_size"]
    ic, ih = cfg["index_head_dim"], cfg["index_n_heads"]
    dsa_projections(ledger, hidden=d, q_rank=cfg.get("q_lora_rank"),
                    index_heads=ih, index_dim=ic)
    kpool = int(cfg.get("index_kpool") or 1)
    if kpool > 1:
        ledger.tensor("indexer.index_kpool_compress_ape", [kpool, ic], "attention")
        ledger.linear("indexer.index_kpool_compress_gate", d, ic)


def apply_mhc(ledger, cfg):
    from app.engines.compressed_model import hyper_connection

    d, n = cfg["hidden_size"], cfg["hc_mult"]
    hyper_connection(ledger, "hc_attn", d, n)
    hyper_connection(ledger, "hc_ffn", d, n)


def apply_attention(ledger, attn_id, cfg):
    if attn_id == ATTN_GQA_GATE:
        apply_gqa_gate(ledger, cfg)
        return
    if attn_id == ATTN_GATED_DELTA:
        apply_gated_delta(ledger, cfg)
        return
    if attn_id == ATTN_MLA:
        apply_mla(ledger, cfg)
        return
    if attn_id == ATTN_KDA:
        apply_kda(ledger, cfg)
        return
    raise ValueError(f"unsupported attention method: {attn_id}")


def apply_moe_swiglu(ledger, cfg):
    d = cfg["hidden_size"]
    expert_dim = cfg["moe_intermediate_size"]
    n_routed = routed_expert_count(cfg)
    top_k = top_k_experts(cfg)
    ledger.linear("router", d, n_routed, "ffn")
    if cfg.get("topk_method") in ("noaux_tc", "noaux-tc") or cfg.get("use_routing_bias"):
        ledger.tensor("router_correction", [n_routed], "ffn", dtype="fp32")
    ledger.ffn(d, expert_dim, prefix="routed_expert.", count=n_routed, active_count=top_k)
    n_shared = shared_expert_count(cfg)
    if n_shared:
        ledger.ffn(d, expert_dim * n_shared, prefix="shared_expert.")
        if has_shared_expert_gate(cfg):
            ledger.linear("shared_expert_gate", d, 1, "ffn")


def has_shared_expert_gate(cfg):
    """One predicate for inventory and graph; noaux shared experts have no gate."""
    return bool(shared_expert_count(cfg)) and cfg.get("topk_method") not in ("noaux_tc", "noaux-tc")


def apply_latent_moe(ledger, cfg):
    d = cfg["hidden_size"]
    latent = cfg["routed_expert_hidden_size"]
    expert_dim = cfg["moe_intermediate_size"]
    n_routed = routed_expert_count(cfg)
    top_k = top_k_experts(cfg)
    ledger.linear("router", d, n_routed, "ffn")
    if cfg.get("topk_method") in ("noaux_tc", "noaux-tc"):
        ledger.tensor("router_correction", [n_routed], "ffn", dtype="fp32")
    ledger.linear("routed_expert_down_proj", d, latent, "ffn")
    ledger.ffn(latent, expert_dim, prefix="routed_expert.", count=n_routed, active_count=top_k)
    if cfg.get("latent_moe_use_norm"):
        ledger.norm("routed_expert_norm", latent, category="ffn")
    ledger.linear("routed_expert_up_proj", latent, d, "ffn")
    n_shared = shared_expert_count(cfg)
    if n_shared:
        ledger.ffn(d, expert_dim * n_shared, prefix="shared_expert.")


def apply_ffn(ledger, ffn_id, cfg):
    if ffn_id == FFN_DENSE:
        apply_dense_swiglu(ledger, cfg)
        return
    if ffn_id == FFN_MOE:
        apply_moe_swiglu(ledger, cfg)
        return
    if ffn_id == FFN_LATENT_MOE:
        apply_latent_moe(ledger, cfg)
        return
    raise ValueError(f"unsupported FFN method: {ffn_id}")


def gqa_gate_links():
    links = []

    def chain(*names):
        links.extend(zip(names, names[1:]))

    chain("attn_norm.weight", "q_gate_proj", "q_norm.weight", "rope_q", "qk_matmul")
    chain("attn_norm.weight", "k_proj", "k_norm.weight", "rope_k", "qk_matmul")
    chain("attn_norm.weight", "v_proj", "av_matmul")
    chain("qk_matmul", "scale_mask", "softmax", "av_matmul", "output_gate", "o_proj", "residual_add")
    chain("q_gate_proj", "output_gate")
    return links


def gated_delta_links(cfg):
    links = []

    def chain(*names):
        links.extend(zip(names, names[1:]))

    chain("attn_norm.weight", "in_proj_qkv", "depthwise_conv", "l2_normalize", "gated_delta_recurrence")
    chain("attn_norm.weight", "in_proj_a", "decay_gate", "gated_delta_recurrence")
    chain("dt_bias", "decay_gate")
    chain("A_log", "decay_gate")
    chain("attn_norm.weight", "in_proj_b", "beta_gate", "gated_delta_recurrence")
    chain("gated_delta_recurrence", "gated_norm.weight", "output_gate", "out_proj", "residual_add")
    chain("attn_norm.weight", "in_proj_z", "output_gate")
    return links


def mla_links(cfg):
    links = []

    def chain(*names):
        links.extend(zip(names, names[1:]))

    if cfg.get("q_lora_rank"):
        chain("attn_norm.weight", "q_a_proj", "q_a_norm.weight", "q_b_proj", "qk_matmul")
    else:
        chain("attn_norm.weight", "q_proj", "qk_matmul")
    chain("attn_norm.weight", "kv_a_proj", "kv_a_norm.weight", "kv_b_proj", "qk_matmul")
    chain("kv_b_proj", "av_matmul")
    chain("qk_matmul", "scale_mask", "softmax", "av_matmul")
    if cfg.get("mla_use_output_gate"):
        chain("attn_norm.weight", "g_proj", "output_gate")
        chain("av_matmul", "output_gate", "o_proj", "residual_add")
    else:
        chain("av_matmul", "o_proj", "residual_add")
    return links


def kda_links(cfg):
    links = []

    def chain(*names):
        links.extend(zip(names, names[1:]))

    chain("attn_norm.weight", "q_proj", "q_conv1d", "kda_recurrence")
    chain("attn_norm.weight", "k_proj", "k_conv1d", "kda_recurrence")
    chain("attn_norm.weight", "v_proj", "v_conv1d", "kda_recurrence")
    chain("attn_norm.weight", "f_a_proj", "f_b_proj", "decay_gate", "kda_recurrence")
    chain("A_log", "decay_gate")
    chain("dt_bias", "decay_gate")
    chain("attn_norm.weight", "b_proj", "beta_gate", "kda_recurrence")
    if (cfg.get("linear_attn_config") or {}).get("use_full_rank_gate"):
        chain("attn_norm.weight", "g_proj", "output_gate")
    else:
        chain("attn_norm.weight", "g_a_proj", "g_b_proj", "output_gate")
    chain("kda_recurrence", "o_norm.weight", "output_gate", "o_proj", "residual_add")
    return links


def attention_links(attn_id, cfg):
    if attn_id == ATTN_GQA_GATE:
        return gqa_gate_links()
    if attn_id == ATTN_GATED_DELTA:
        return gated_delta_links(cfg)
    if attn_id == ATTN_MLA:
        return mla_links(cfg)
    if attn_id == ATTN_KDA:
        return kda_links(cfg)
    raise ValueError(f"unsupported attention method: {attn_id}")


def dense_ffn_links():
    links = [
        ("residual_add", "ffn_norm.weight"),
        ("ffn_norm.weight", "gate_proj"),
        ("gate_proj", "silu"),
        ("silu", "multiply"),
        ("multiply", "down_proj"),
        ("down_proj", "output_add"),
        ("ffn_norm.weight", "up_proj"),
        ("up_proj", "multiply"),
        ("residual_add", "output_add"),
    ]
    return links


def moe_ffn_links(cfg):
    links = [
        ("residual_add", "ffn_norm.weight"),
        ("ffn_norm.weight", "router"),
        ("router", "expert_selection"),
        ("ffn_norm.weight", "routed_expert.gate_proj"),
        ("routed_expert.gate_proj", "routed_expert.silu"),
        ("routed_expert.silu", "routed_expert.multiply"),
        ("ffn_norm.weight", "routed_expert.up_proj"),
        ("routed_expert.up_proj", "routed_expert.multiply"),
        ("routed_expert.multiply", "routed_expert.down_proj"),
        ("routed_expert.down_proj", "weighted_expert_sum"),
        ("expert_selection", "routed_expert.gate_proj"),
        ("expert_selection", "weighted_expert_sum"),
        ("weighted_expert_sum", "output_add"),
        ("residual_add", "output_add"),
    ]
    if shared_expert_count(cfg):
        links.extend([
            ("ffn_norm.weight", "shared_expert.gate_proj"),
            ("shared_expert.gate_proj", "shared_expert.silu"),
            ("shared_expert.silu", "shared_expert.multiply"),
            ("ffn_norm.weight", "shared_expert.up_proj"),
            ("shared_expert.up_proj", "shared_expert.multiply"),
            ("shared_expert.multiply", "shared_expert.down_proj"),
            ("shared_expert.down_proj", "weighted_expert_sum"),
        ])
        if has_shared_expert_gate(cfg):
            links.extend([
            ("ffn_norm.weight", "shared_expert_gate"),
            ("shared_expert_gate", "weighted_expert_sum"),
            ])
    return links


def latent_moe_ffn_links(cfg):
    links = [
        ("residual_add", "ffn_norm.weight"),
        ("ffn_norm.weight", "router"),
        ("router", "expert_selection"),
        ("ffn_norm.weight", "routed_expert_down_proj"),
        ("routed_expert_down_proj", "routed_expert.gate_proj"),
        ("routed_expert.gate_proj", "routed_expert.silu"),
        ("routed_expert.silu", "routed_expert.multiply"),
        ("routed_expert_down_proj", "routed_expert.up_proj"),
        ("routed_expert.up_proj", "routed_expert.multiply"),
        ("routed_expert.multiply", "routed_expert.down_proj"),
        ("expert_selection", "routed_expert.gate_proj"),
        ("expert_selection", "weighted_expert_sum"),
        ("routed_expert.down_proj", "weighted_expert_sum"),
        ("routed_expert_up_proj", "expert_output_sum"),
        ("expert_output_sum", "output_add"),
        ("residual_add", "output_add"),
    ]
    if cfg.get("latent_moe_use_norm"):
        links.extend([
            ("weighted_expert_sum", "routed_expert_norm.weight"),
            ("routed_expert_norm.weight", "routed_expert_up_proj"),
        ])
    else:
        links.extend([
            ("weighted_expert_sum", "routed_expert_up_proj"),
        ])
    if shared_expert_count(cfg):
        links.extend([
            ("ffn_norm.weight", "shared_expert.gate_proj"),
            ("shared_expert.gate_proj", "shared_expert.silu"),
            ("shared_expert.silu", "shared_expert.multiply"),
            ("ffn_norm.weight", "shared_expert.up_proj"),
            ("shared_expert.up_proj", "shared_expert.multiply"),
            ("shared_expert.multiply", "shared_expert.down_proj"),
            ("shared_expert.down_proj", "expert_output_sum"),
        ])
    return links


def layer_links(attn_id, ffn_id, cfg, *, index=None):
    prefix = [("input", "attn_norm.weight"), ("input", "residual_add")]
    attn = attention_links(attn_id, cfg)
    if ffn_id == FFN_DENSE:
        links = prefix + attn + dense_ffn_links()
    elif ffn_id == FFN_MOE:
        links = prefix + attn + moe_ffn_links(cfg)
    elif ffn_id == FFN_LATENT_MOE:
        links = prefix + attn + latent_moe_ffn_links(cfg)
    else:
        raise ValueError(f"unsupported FFN method: {ffn_id}")
    if attn_id == ATTN_MLA and cfg.get("index_n_heads"):
        kinds = cfg.get("indexer_types") or []
        shared = index is not None and index < len(kinds) and kinds[index] == "shared"
        if shared:
            links.append(("shared_index_selection", "scale_mask"))
        else:
            links.extend([
                ("q_a_norm.weight", "indexer.q_proj"),
                ("attn_norm.weight", "indexer.k_proj"),
                ("indexer.k_proj", "indexer.k_norm.weight"),
                ("indexer.k_norm.weight", "indexer.k_norm.bias"),
                ("indexer.k_norm.bias", "indexer_scores"),
                ("indexer.q_proj", "indexer_scores"),
                ("attn_norm.weight", "indexer.weights_proj"),
                ("indexer.weights_proj", "indexer_scores"),
                ("indexer_scores", "indexer_top_k"),
                ("indexer_top_k", "scale_mask"),
            ])
            if int(cfg.get("index_kpool") or 1) > 1:
                links.remove(("indexer.k_norm.bias", "indexer_scores"))
                links.extend([
                    ("indexer.k_norm.bias", "indexer_key_pool"),
                    ("indexer.index_kpool_compress_ape", "indexer_key_pool"),
                    ("attn_norm.weight", "indexer.index_kpool_compress_gate"),
                    ("indexer.index_kpool_compress_gate", "indexer_key_pool"),
                    ("indexer_key_pool", "indexer_scores"),
                ])
    if cfg.get("hc_mult"):
        for name, incoming, norm, output in (
            ("hc_attn", "input", "attn_norm.weight", "residual_add"),
            ("hc_ffn", "residual_add", "ffn_norm.weight", "output_add"),
        ):
            links.remove((incoming, norm))
            links.remove((incoming, output))
            links = [(a, name + "_post" if b == output and a != incoming else b) for a, b in links]
            links.extend([
                (incoming, name + ".fn"), (name + ".fn", name + "_mix"),
                (name + ".base", name + "_mix"), (name + ".scale", name + "_mix"),
                (name + "_mix", name + "_sinkhorn"), (name + "_sinkhorn", name + "_pre"),
                (incoming, name + "_pre"), (name + "_pre", norm),
                (name + "_sinkhorn", name + "_post"), (incoming, name + "_post"),
                (name + "_post", output),
            ])
    if cfg.get("attn_res_block_size"):
        for stem, incoming, norm in (
            ("self_attention_res", "input", "attn_norm.weight"),
            ("mlp_res", "residual_add", "ffn_norm.weight"),
        ):
            links.remove((incoming, norm))
            links.extend([
                (incoming, stem + "_values"), ("block_residual_history", stem + "_values"),
                (stem + "_values", stem + "_norm.weight"),
                (stem + "_norm.weight", stem + "_proj"),
                (stem + "_proj", stem + "_softmax"),
                (stem + "_softmax", stem + "_weighted_sum"),
                (stem + "_values", stem + "_weighted_sum"),
                (stem + "_weighted_sum", norm),
            ])
    if cfg.get("topk_method") in ("noaux_tc", "noaux-tc") or cfg.get("use_routing_bias"):
        if ffn_id != FFN_DENSE:
            links.append(("router_correction", "expert_selection"))
    return links


def hybrid_cache(cfg, types, options):
    elem = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[options.get("kv_dtype", "bf16")]
    linear_layers = types.count("linear_attention")
    full_layers = sum(1 for kind in types if kind in ("full_attention", "deepseek_sparse_attention"))
    kv = cfg["num_key_value_heads"]
    c = cfg["head_dim"]
    fixed = linear_layers * (
        (2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
         + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]) * cfg["linear_conv_kernel_dim"] * elem
        + cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"] * cfg["linear_value_head_dim"] * 4)
    slope = full_layers * 2 * kv * c * elem
    return dict(
        kind="hybrid-recurrent", bytes_per_token=None, fixed_state_bytes=fixed,
        linear_bytes_per_token=slope, dtype_assumed=options.get("kv_dtype", "bf16"),
        formula="B * (recurrent FP32 state + convolution state + S * full-attention KV)",
        notes="Recurrent state does not grow with sequence length. State uses FP32; convolution/KV use selected dtype.",
        reference_context=4096, reference_bytes=fixed + 4096 * slope,
    )


def kda_cache(cfg, types, options):
    elem = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[options.get("kv_dtype", "bf16")]
    spec = cfg["linear_attn_config"]
    heads, c, kernel = spec["num_heads"], spec["head_dim"], spec["short_conv_kernel_size"]
    width = heads * c
    linear_layers = types.count("linear_attention")
    fixed = linear_layers * (3 * width * (kernel - 1) * elem + heads * c * c * 4)
    full_layers = sum(1 for kind in types if kind in ("full_attention", "deepseek_sparse_attention"))
    slope = full_layers * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]) * elem
    return dict(
        kind="hybrid-recurrent", bytes_per_token=None, fixed_state_bytes=fixed,
        linear_bytes_per_token=slope, dtype_assumed=options.get("kv_dtype", "bf16"),
        formula="B * (KDA conv/recurrent FP32 state + S * MLA latent KV)",
        notes="KDA convolution uses selected dtype; recurrent state is FP32. MLA layers keep latent KV.",
        reference_context=4096, reference_bytes=fixed + 4096 * slope,
    )


def fill_hybrid_layer(cfg, attn_id, ffn_id, *, source=QWEN_SOURCE):
    ledger = Ledger(source)
    d = cfg["hidden_size"]
    ledger.norm("attn_norm", d)
    ledger.norm("ffn_norm", d)
    apply_attention(ledger, attn_id, cfg)
    apply_ffn(ledger, ffn_id, cfg)
    return ledger


def fill_linear_layer(cfg, attn_id, ffn_id, index=None):
    ledger = fill_hybrid_layer(cfg, attn_id, ffn_id, source=KIMI_SOURCE)
    if attn_id == ATTN_MLA and cfg.get("index_n_heads"):
        kinds = cfg.get("indexer_types")
        kind = "full"
        if isinstance(kinds, list) and index is not None and index < len(kinds):
            kind = str(kinds[index])
        if kind != "shared":
            apply_dsa_indexer(ledger, cfg)
    if cfg.get("hc_mult"):
        apply_mhc(ledger, cfg)
    if cfg.get("attn_res_block_size"):
        d = cfg["hidden_size"]
        extra = Ledger(KIMI_SOURCE)
        extra.norm("self_attention_res_norm", d)
        extra.linear("self_attention_res_proj", d, 1, "residual")
        extra.norm("mlp_res_norm", d)
        extra.linear("mlp_res_proj", d, 1, "residual")
        ledger.rows.extend(extra.rows)
    return ledger
