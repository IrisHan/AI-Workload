from __future__ import annotations

from typing import Any

from .ir import (
    AttentionKind,
    Backbone,
    Capability,
    Evidence,
    FfnKind,
    Identity,
    LayerSpec,
    MLAConfig,
    MoEConfig,
    NormKind,
    NormMode,
    PositionalKind,
    ResidualKind,
    UnreadField,
)

QK_NORM_MODEL_TYPES = {
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "olmo2",
    "gemma3",
    "gemma3_text",
}
POST_NORM_MODEL_TYPES = {"olmo2"}
SANDWICH_NORM_MODEL_TYPES = {"gemma2", "gemma3", "gemma3_text"}

METADATA_KEYS = {
    "architectures",
    "auto_map",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "transformers_version",
    "torch_dtype",
    "dtype",
    "use_cache",
    "initializer_range",
    "attention_dropout",
    "output_router_logits",
    "router_aux_loss_coef",
    "pretraining_tp",
    "_name_or_path",
    "model_type",
    "quantization_config",
}

HIDDEN_KEYS = ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim")
LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
HEAD_KEYS = ("num_attention_heads", "n_head", "num_heads", "n_heads")
KV_HEAD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_kv_heads")
HEAD_DIM_KEYS = ("head_dim", "kv_channels")
VOCAB_KEYS = ("vocab_size", "padded_vocab_size")
CTX_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "n_ctx",
    "max_seq_len",
    "max_sequence_length",
    "seq_length",
    "model_max_length",
)
INTER_KEYS = (
    "intermediate_size",
    "n_inner",
    "ffn_hidden_size",
    "ffn_dim",
    "block_ff_dim",
)
ACT_KEYS = ("hidden_act", "hidden_activation", "activation_function")
EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
TOPK_KEYS = (
    "num_experts_per_tok",
    "experts_per_token",
    "num_experts_per_token",
    "moe_topk",
    "moe_top_k",
    "moe_k",
    "num_selected_experts",
)

DECORATIVE_KEYS = (
    "attn_dropout_prob",
    "emb_dropout_prob",
    "bf16",
    "fp16",
    "fp32",
    "onnx_safe",
    "tokenizer_class",
    "scale_attn_weights",
    "use_flash_attn",
    "use_cache_quantization",
    "use_cache_kernel",
    "softmax_in_fp32",
)

SYNONYM_GROUPS = (
    HIDDEN_KEYS,
    LAYER_KEYS,
    HEAD_KEYS,
    KV_HEAD_KEYS,
    HEAD_DIM_KEYS,
    VOCAB_KEYS,
    CTX_KEYS,
    INTER_KEYS,
    ACT_KEYS,
    EXPERT_KEYS,
    TOPK_KEYS,
    ("rope_theta", "rotary_emb_base", "rope_embedding_base"),
    ("attention_bias", "no_bias"),
    ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"),
)


def sparse_attention_spec(cfg):
    spec = cfg.get("sparse_attention_config")
    return spec if isinstance(spec, dict) else {}


def layer_extras(cfg, index, *, qk_norm=False, ffn_kind=None):
    extras = []
    if qk_norm:
        extras.append("QK_NORM_FUSED" if cfg.get("qk_norm_type") == "per_layer" else "QK_NORM")
    sparse = sparse_attention_spec(cfg)
    freq = sparse.get("sparse_attention_freq")
    if isinstance(freq, list) and index < len(freq) and freq[index]:
        extras.append("MSA")
    indexer_types = cfg.get("indexer_types")
    if cfg.get("index_n_heads") and cfg.get("kv_lora_rank") is not None:
        kind = "full"
        if isinstance(indexer_types, list) and index < len(indexer_types):
            kind = str(indexer_types[index])
        if kind != "shared":
            extras.append("DSA_INDEXER")
            if int(cfg.get("index_kpool") or 1) > 1:
                extras.append("INDEX_KPOOL")
    if cfg.get("attention_bias") and (
        cfg.get("num_local_experts") or cfg.get("n_routed_experts") or cfg.get("num_experts")
    ):
        extras.append("ATTN_SINKS")
        if ffn_kind == FfnKind.MOE:
            extras.extend(("ROUTER_BIAS", "EXPERT_BIAS"))
    if (cfg.get("use_routing_bias") or cfg.get("moe_use_aux_free")) and ffn_kind == FfnKind.MOE:
        extras.append("ROUTER_CORRECTION")
    return tuple(extras)


def _is_qwen1(model_type: str, arches: list[str]) -> bool:
    if str(model_type) == "qwen":
        return True
    return any("QWenLMHeadModel" in str(a) for a in arches)


class MatchContext:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.consumed: set[str] = set()
        self.capabilities: list[Capability] = []
        self.warnings: list[str] = []

    def get(self, *keys: str, default: Any = None) -> Any:
        for k in keys:
            if self.cfg.get(k) is not None:
                self.consumed.add(k)
                return self.cfg[k]
        return default

    def has(self, key: str) -> bool:
        return self.cfg.get(key) is not None

    def mark(self, *keys: str) -> None:
        self.consumed.update(keys)

    def add_cap(self, cap: Capability) -> None:
        self.capabilities.append(cap)


def _unwrap(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    arches = list(cfg.get("architectures") or [])
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        inner = {**cfg["text_config"]}
        if arches and "architectures" not in inner:
            inner["architectures"] = arches
        if cfg.get("model_type") and "model_type" not in inner:
            inner["model_type"] = cfg["model_type"]
        return inner, arches
    return cfg, arches


def match_config(raw: dict[str, Any], *, name: str = "model", compact: bool = False) -> dict[str, Any]:
    from .validation import has_mla_fields, validate_config, validate_match

    validate_config(raw)
    cfg, arches = _unwrap(raw)
    ctx = MatchContext(cfg)
    ctx.mark("architectures", "text_config")

    model_type = ctx.get("model_type", default="unknown")
    is_qwen1 = _is_qwen1(str(model_type), arches)
    hidden = ctx.get(*HIDDEN_KEYS)
    layers = ctx.get(*LAYER_KEYS)
    heads = ctx.get(*HEAD_KEYS)
    if hidden is None or layers is None:
        raise ValueError(
            f"config '{name}' has no recognizable hidden_size / num_hidden_layers (model_type={model_type!r})"
        )
    if heads is None:
        heads = max(hidden // 128, 1)
        ctx.warnings.append("num_attention_heads missing; guessed hidden_size/128")

    kv_heads = ctx.get(*KV_HEAD_KEYS, default=heads)
    if ctx.get("multi_query"):
        kv_heads = 1
        ctx.mark("multi_query")

    head_dim = ctx.get(*HEAD_DIM_KEYS, default=None)
    vocab = ctx.get(*VOCAB_KEYS, default=0) or 0
    ctx_len = ctx.get(*CTX_KEYS, default=0) or 0
    inter = ctx.get(*INTER_KEYS, default=4 * hidden)
    dense_width = ctx.get("dense_intermediate_size", default=None)
    act = ctx.get(*ACT_KEYS, default=None)
    ctx.mark("block_use_swiglu")
    if is_qwen1:
        # QWenMLP: w1/w2/c_proj use intermediate_size // 2; SwiGLU even without hidden_act.
        if inter:
            inter = int(inter) // 2
        if act is None:
            act = "silu"
    elif act is None:
        act = "silu"
    if act == "swigluoai":
        act = "silu"

    attn_bias = bool(ctx.get("attention_bias", default=model_type == "gpt2"))
    qkv_bias = True if model_type == "qwen2" else attn_bias
    o_bias = False if model_type == "qwen2" else attn_bias
    mlp_bias = bool(ctx.get("mlp_bias", default=False))
    if ctx.cfg.get("qkv_bias") is not None:
        qkv_bias = bool(ctx.get("qkv_bias"))
    if ctx.get("no_bias"):
        attn_bias = False
        mlp_bias = False
    tied = bool(
        ctx.get(
            "tie_word_embeddings",
            default=model_type in {"gpt2", "gemma", "gemma2", "gemma3", "gemma3_text"},
        )
    )
    mtp = 0
    if has_mla_fields(cfg):
        mtp = int(ctx.get("num_nextn_predict_layers", default=0) or 0)
    else:
        ctx.mark("num_nextn_predict_layers")

    mla, attn_kind = _match_attention(ctx, heads, kv_heads, hidden)
    if head_dim is None:
        if mla is not None:
            head_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
        else:
            head_dim = hidden // heads if heads else 0

    moe = _match_moe(ctx, inter)
    ffn_kind = FfnKind.MOE if moe else FfnKind.DENSE
    positional = _match_positional(ctx, model_type)
    _match_qwen_pos_extras(ctx)
    norm_kind, norm_mode = _match_norm(ctx, model_type, rms_despite_ln_eps=is_qwen1)
    residual = _match_residual(ctx)
    sliding = _match_window(ctx)
    _match_scoring(ctx)
    _match_quant(ctx)
    _match_unsupported(ctx)
    ctx.mark(*DECORATIVE_KEYS)

    if mtp:
        ctx.add_cap(
            Capability(
                id="mtp",
                label=f"MTP ×{mtp}",
                kind="head",
                evidence=[Evidence(signal="mtp", config_key="num_nextn_predict_layers", value=mtp)],
            )
        )

    qk_norm = model_type in QK_NORM_MODEL_TYPES or bool(ctx.get("qk_norm", default=False))
    if ctx.cfg.get("use_qk_norm") is not None:
        qk_norm = bool(ctx.get("use_qk_norm"))
    ctx.mark("qk_norm_type")
    if qk_norm:
        ctx.add_cap(
            Capability(
                id="qk-norm",
                label="QK-Norm",
                kind="attention",
                evidence=[Evidence(signal="qk_norm", config_key="model_type", value=model_type)],
            )
        )

    use_swa = bool(sliding) and bool(ctx.cfg.get("use_sliding_window", True))
    layer_types = ctx.get("layer_types")
    backbone = Backbone(
        vocab_size=int(vocab),
        num_layers=int(layers),
        hidden_size=int(hidden),
        num_heads=int(heads),
        num_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        intermediate_size=int(dense_width if dense_width is not None else inter),
        context_length=int(ctx_len),
        activation=str(act),
        attention_bias=attn_bias,
        qkv_bias=qkv_bias,
        o_bias=o_bias,
        mlp_bias=mlp_bias,
        tied_embeddings=tied,
        mtp_layers=mtp,
        sliding_window=int(sliding) if use_swa and sliding else None,
    )

    moe_frequency = cfg.get("moe_layer_freq", 1)
    if "decoder_sparse_step" in cfg:
        ctx.mark("decoder_sparse_step")
        moe_frequency = cfg["decoder_sparse_step"]
    mlp_only = cfg.get("mlp_only_layers") or []
    ctx.mark("mlp_only_layers")
    ctx.mark(
        "sparse_attention_config",
        "indexer_types",
        "mlp_layer_types",
        "index_head_dim",
        "index_n_heads",
        "index_topk",
        "index_kpool",
        "index_kpool_always_select_tail",
        "index_kpool_compress",
        "index_topk_freq",
        "index_skip_topk_offset",
        "index_topk_pattern",
        "indexer_rope_interleave",
        "index_share_for_mtp_iteration",
        "use_routing_bias",
        "use_gemma_norm",
        "attention_output_gate",
        "rotary_dim",
        "partial_rotary_factor",
        "swiglu_limit",
        "swiglu_alpha",
        "attn_type_list",
        "dense_intermediate_size",
        "moe_layer_interval",
        "moe_layer_start_index",
        "moe_capacity",
        "moe_gate",
        "moe_use_aux_free",
        "use_bias",
        "mtp_transformer_layers",
        "num_mtp_modules",
        "use_mtp",
        "initial_context_length",
        "experts_per_token",
        "qk_head_dim",
        "rope_interleave",
        "rope_parameters",
        "moe_router_dtype",
        "extra_mtp_params",
        "extra_mtp_num_heads",
        "extra_mtp_hidden_size",
        "extra_mtp_num_layers",
        "linear_attn_config",
        "hc_mult",
        "hc_eps",
        "hc_sinkhorn_iters",
    )

    unread = _unread(ctx, raw_keys=set(cfg.keys()))
    identity = Identity(
        name=name,
        model_type=str(model_type),
        architectures=arches or list(ctx.get("architectures", default=[]) or []),
    )

    result = {
        "identity": identity,
        "backbone": backbone,
        "capabilities": ctx.capabilities,
        "attention_kind": attn_kind,
        "ffn_kind": ffn_kind,
        "norm_kind": norm_kind,
        "norm_mode": norm_mode,
        "positional": positional,
        "residual": residual,
        "mla": mla,
        "moe": moe,
        "layers": [],
        "warnings": ctx.warnings,
        "unread_fields": unread,
        "qk_norm": qk_norm,
        "cfg": cfg,
    }
    validate_match(result)
    result["layers"] = _build_layers(
        layers=layers,
        attn_kind=attn_kind,
        moe=moe,
        sliding=sliding if use_swa else None,
        layer_types=layer_types,
        moe_frequency=moe_frequency,
        mlp_only=mlp_only,
        compact=compact,
        cfg=cfg,
        qk_norm=qk_norm,
    )
    return result


def _match_attention(
    ctx: MatchContext, heads: int, kv_heads: int, hidden: int
) -> tuple[MLAConfig | None, AttentionKind]:
    mla_keys = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    if all(ctx.has(k) for k in mla_keys):
        mla = MLAConfig(
            q_lora_rank=ctx.get("q_lora_rank"),
            kv_lora_rank=int(ctx.get("kv_lora_rank")),
            qk_nope_head_dim=int(ctx.get("qk_nope_head_dim")),
            qk_rope_head_dim=int(ctx.get("qk_rope_head_dim")),
            v_head_dim=int(ctx.get("v_head_dim")),
        )
        ctx.add_cap(
            Capability(
                id="mla",
                label="MLA",
                kind="attention",
                evidence=[Evidence(signal="mla", config_key=k, value=ctx.cfg[k]) for k in mla_keys],
            )
        )
        ctx.mark("mla_use_output_gate", "mla_use_nope")
        return mla, AttentionKind.MLA

    if kv_heads == 1:
        kind = AttentionKind.MQA
        cap_id, label = "mqa", "MQA"
    elif kv_heads < heads:
        kind = AttentionKind.GQA
        cap_id, label = "gqa", "GQA"
    else:
        kind = AttentionKind.MHA
        cap_id, label = "mha", "MHA"
    ctx.add_cap(
        Capability(
            id=cap_id,
            label=label,
            kind="attention",
            evidence=[
                Evidence(signal="heads", config_key="num_attention_heads", value=heads),
                Evidence(signal="kv_heads", config_key="num_key_value_heads", value=kv_heads),
            ],
        )
    )
    return None, kind


def _match_moe(ctx: MatchContext, inter: int) -> MoEConfig | None:
    from .validation import shared_expert_multiplicity

    n_experts = ctx.get(*EXPERT_KEYS)
    if not n_experts or n_experts <= 1:
        ctx.add_cap(
            Capability(
                id="dense-ffn",
                label="Dense FFN",
                kind="ffn",
                evidence=[Evidence(signal="dense", config_key="intermediate_size", value=inter)],
            )
        )
        return None
    counted_shared = ctx.get("n_shared_experts", "num_shared_experts", "moe_num_shared_experts", default=0) or 0
    shared_width = None
    for key in ("shared_expert_intermediate_size", "shared_intermediate_size"):
        if key in ctx.cfg:
            ctx.mark(key)
            if ctx.cfg[key] is not None:
                shared_width = ctx.cfg[key]
    expert_dim = ctx.get("moe_intermediate_size", default=inter)
    shared = shared_expert_multiplicity(shared_width, expert_dim, counted_shared)
    moe = MoEConfig(
        n_routed=int(n_experts),
        top_k=int(ctx.get(*TOPK_KEYS, default=1)),
        n_shared=int(shared),
        expert_dim=int(expert_dim),
        dense_prefix=int(ctx.get("first_k_dense_replace", "moe_layer_start_index", default=0) or 0),
        scoring_func=ctx.get("scoring_func"),
        topk_method=ctx.get("topk_method"),
        n_group=ctx.get("n_group"),
        topk_group=ctx.get("topk_group"),
    )
    ctx.mark("moe_layer_freq", "norm_topk_prob", "routed_scaling_factor", "ep_size",
             "moe_renormalize", "moe_router_activation_func", "num_expert_group",
             "use_grouped_topk", "aux_loss_alpha", "seq_aux", "moe_layer_interval")
    ctx.add_cap(
        Capability(
            id="moe",
            label=f"MoE {moe.n_routed}e/{moe.top_k}k",
            kind="ffn",
            evidence=[
                Evidence(signal="experts", config_key="n_routed_experts", value=moe.n_routed),
                Evidence(signal="topk", config_key="num_experts_per_tok", value=moe.top_k),
            ],
        )
    )
    if moe.n_shared:
        ctx.add_cap(
            Capability(
                id="shared-expert",
                label=f"{moe.n_shared} shared expert",
                kind="ffn",
                evidence=[Evidence(signal="shared", config_key="n_shared_experts", value=moe.n_shared)],
            )
        )
    return moe


def _match_positional(ctx: MatchContext, model_type: str) -> PositionalKind:
    theta = ctx.get("rope_theta", "rotary_emb_base", "rope_embedding_base")
    scaling = ctx.get("rope_scaling")
    ctx.mark("use_mrope")
    if ctx.get("alibi"):
        kind = PositionalKind.ALIBI
        cap = "alibi"
    elif (
        scaling
        and isinstance(scaling, dict)
        and str(scaling.get("type") or scaling.get("rope_type") or "").lower() == "yarn"
    ):
        kind = PositionalKind.YARN
        cap = "yarn"
    elif (
        scaling
        and isinstance(scaling, dict)
        and str(scaling.get("rope_type") or scaling.get("type") or "").lower() in {"llama3"}
    ):
        kind = PositionalKind.LLAMA3
        cap = "rope-llama3"
    elif theta is not None or ctx.get("rotary"):
        kind = PositionalKind.ROPE
        cap = "rope"
    elif model_type == "gpt2":
        kind = PositionalKind.LEARNED
        cap = "learned-pe"
    else:
        kind = PositionalKind.NOPE
        cap = "nope"
    ctx.add_cap(
        Capability(
            id=cap,
            label=kind.value,
            kind="position",
            evidence=[Evidence(signal="positional", config_key="rope_theta", value=theta)],
        )
    )
    return kind


def _match_qwen_pos_extras(ctx: MatchContext) -> None:
    rotary_pct = ctx.get("rotary_pct")
    if rotary_pct is not None and float(rotary_pct) < 1.0:
        ctx.warnings.append(f"rotary_pct={rotary_pct}: RoPE applied to a subset of head_dim")
    if ctx.get("use_dynamic_ntk"):
        ctx.add_cap(
            Capability(
                id="dynamic-ntk",
                label="dynamic NTK",
                kind="position",
                evidence=[Evidence(signal="ntk", config_key="use_dynamic_ntk", value=True)],
            )
        )
    if ctx.get("use_logn_attn"):
        ctx.add_cap(
            Capability(
                id="logn-attn",
                label="logN attention",
                kind="attention",
                evidence=[Evidence(signal="logn", config_key="use_logn_attn", value=True)],
            )
        )


def _match_norm(
    ctx: MatchContext, model_type: str, *, rms_despite_ln_eps: bool = False
) -> tuple[NormKind, NormMode]:
    if rms_despite_ln_eps:
        ctx.get("layer_norm_epsilon", "layer_norm_eps", "rms_norm_eps")
        kind = NormKind.RMSNORM
        cap_id = "rmsnorm"
    elif ctx.has("rms_norm_eps"):
        ctx.get("rms_norm_eps")
        kind = NormKind.RMSNORM
        cap_id = "rmsnorm"
    elif ctx.has("layer_norm_epsilon") or ctx.has("layer_norm_eps"):
        ctx.get("layer_norm_epsilon", "layer_norm_eps")
        kind = NormKind.LAYERNORM
        cap_id = "layernorm"
    else:
        ctx.get("rms_norm_eps")
        kind = NormKind.RMSNORM
        cap_id = "rmsnorm"
    if model_type in POST_NORM_MODEL_TYPES:
        mode = NormMode.POST
    elif model_type in SANDWICH_NORM_MODEL_TYPES:
        mode = NormMode.SANDWICH
    else:
        mode = NormMode.PRE
    ctx.add_cap(
        Capability(
            id=cap_id,
            label=f"{kind.value} ({mode.value})",
            kind="norm",
            evidence=[Evidence(signal="norm", config_key="model_type", value=model_type)],
        )
    )
    return kind, mode


def _match_residual(ctx: MatchContext) -> ResidualKind:
    hc = ctx.get("hc_mult")
    if hc:
        ctx.mark("hc_eps", "hc_sinkhorn_iters")
        ctx.add_cap(
            Capability(
                id="mhc",
                label=f"mHC ×{hc}",
                kind="residual",
                evidence=[Evidence(signal="mhc", config_key="hc_mult", value=hc)],
                supported=False,
                notes="mHC detected; phase-1 accountant does not expand Sinkhorn maps",
            )
        )
        ctx.warnings.append("hc_mult present: residual counted as plain add (mHC params omitted)")
        return ResidualKind.MHC
    ctx.add_cap(
        Capability(
            id="residual-add",
            label="residual add",
            kind="residual",
            evidence=[Evidence(signal="residual", config_key="(default)", value="add")],
        )
    )
    return ResidualKind.ADD


def _match_window(ctx: MatchContext) -> int | None:
    sliding = ctx.get("sliding_window")
    use = ctx.get("use_sliding_window", default=True)
    ctx.mark("max_window_layers", "sliding_window_pattern")
    if sliding and use:
        ctx.add_cap(
            Capability(
                id="swa",
                label=f"sliding window {sliding}",
                kind="attention",
                evidence=[Evidence(signal="swa", config_key="sliding_window", value=sliding)],
            )
        )
        return int(sliding)
    return None


def _match_scoring(ctx: MatchContext) -> None:
    scoring = ctx.get("scoring_func")
    if scoring:
        ctx.add_cap(
            Capability(
                id="scoring",
                label=f"router {scoring}",
                kind="ffn",
                evidence=[Evidence(signal="scoring", config_key="scoring_func", value=scoring)],
            )
        )


def _match_quant(ctx: MatchContext) -> None:
    qcfg = ctx.get("quantization_config")
    expert_dtype = ctx.get("expert_dtype")
    if qcfg or expert_dtype:
        fmt = None
        bits = None
        if isinstance(qcfg, dict):
            fmt = qcfg.get("quant_method") or qcfg.get("fmt")
            bits = qcfg.get("bits")
        label = f"quant {expert_dtype or fmt}"
        if bits is not None:
            label = f"quant {fmt}-{bits}"
        ctx.add_cap(
            Capability(
                id="quant",
                label=label,
                kind="precision",
                evidence=[
                    Evidence(signal="quant", config_key="quantization_config", value=fmt),
                    Evidence(signal="expert_dtype", config_key="expert_dtype", value=expert_dtype),
                ],
            )
        )


def _match_unsupported(ctx: MatchContext) -> None:
    if ctx.has("compress_ratios"):
        ctx.mark("compress_ratios", "compress_rope_theta", "index_topk", "index_head_dim", "index_n_heads")
        ctx.add_cap(
            Capability(
                id="csa-hca",
                label="CSA/HCA (unsupported in phase 1)",
                kind="attention",
                evidence=[Evidence(signal="compress", config_key="compress_ratios", value="present")],
                supported=False,
                notes="DeepSeek-V4 mixed sparse attention is recognized but not accounted yet",
            )
        )
        ctx.warnings.append(
            "compress_ratios: CSA/HCA not in phase-1 accountant; attention counted as MLA/GQA skeleton if fields exist"
        )


def _build_layers(
    *,
    layers: int,
    attn_kind: AttentionKind,
    moe: MoEConfig | None,
    sliding: int | None,
    layer_types: Any,
    moe_frequency: int = 1,
    mlp_only: Any = None,
    compact: bool = False,
    cfg: dict[str, Any] | None = None,
    qk_norm: bool = False,
) -> list[LayerSpec]:
    out: list[LayerSpec] = []
    dense_prefix = moe.dense_prefix if moe else 0
    mlp_only_set = {int(i) for i in (mlp_only or [])}
    mlp_types = (cfg or {}).get("mlp_layer_types")
    freq_list = moe_frequency if isinstance(moe_frequency, list) else None
    extras_vary = bool(cfg) and (
        qk_norm
        or sparse_attention_spec(cfg)
        or cfg.get("index_n_heads")
        or cfg.get("attention_bias")
        or cfg.get("use_routing_bias")
        or cfg.get("moe_use_aux_free")
        or freq_list is not None
        or isinstance(mlp_types, list)
    )
    if compact and layer_types is None and moe_frequency == 1 and not mlp_only_set and not extras_vary:
        boundaries = sorted({0, dense_prefix, layers})
    else:
        boundaries = range(layers + 1)
    grouped: dict[tuple, LayerSpec] = {}
    for i, stop in zip(boundaries, list(boundaries)[1:]):
        if freq_list is not None:
            ffn = FfnKind.MOE if moe is not None and i < len(freq_list) and freq_list[i] else FfnKind.DENSE
        elif isinstance(mlp_types, list) and i < len(mlp_types):
            ffn = FfnKind.DENSE if str(mlp_types[i]) == "dense" or moe is None else FfnKind.MOE
        else:
            ffn = FfnKind.DENSE if (
                moe is None or i < dense_prefix or i in mlp_only_set or i % (moe_frequency or 1)
            ) else FfnKind.MOE
        attn = attn_kind
        if isinstance(layer_types, list) and i < len(layer_types):
            lt = str(layer_types[i])
            if "sliding" in lt:
                attn = AttentionKind.SWA
        elif sliding and attn_kind != AttentionKind.MLA:
            attn = AttentionKind.SWA
        extras = layer_extras(cfg or {}, i, qk_norm=qk_norm, ffn_kind=ffn)
        key = (attn, ffn, extras)
        if compact and key in grouped:
            layer = grouped[key]
            layer.count += stop - i
            if layer.ranges[-1][1] == i:
                layer.ranges[-1][1] = stop
            else:
                layer.ranges.append([i, stop])
        else:
            layer = LayerSpec(
                index=i, count=stop - i, ranges=[[i, stop]], attn_kind=attn, ffn_kind=ffn,
                extras=extras, role="main",
            )
            out.append(layer)
            grouped[key] = layer
    return out


def _unread(ctx: MatchContext, *, raw_keys: set[str]) -> list[UnreadField]:
    skip_prefix = "_"
    unread: list[UnreadField] = []
    for key in sorted(raw_keys):
        if key in ctx.consumed or key in METADATA_KEYS or key.startswith(skip_prefix):
            continue
        val = ctx.cfg.get(key)
        preview: Any = val
        if isinstance(val, (list, dict)) and len(str(val)) > 120:
            preview = f"<{type(val).__name__} len={len(val)}>"
        reason = "not consumed by matcher"
        for group in SYNONYM_GROUPS:
            if key in group and any(g in ctx.consumed for g in group):
                reason = "synonym"
                break
        unread.append(UnreadField(key=key, value_preview=preview, reason=reason))
    return unread
