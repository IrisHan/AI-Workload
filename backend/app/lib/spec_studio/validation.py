"""Supported structure boundary. Recognition alone never authorizes accounting."""

from __future__ import annotations

import math

SUPPORTED = {
    "qwen": "QWenLMHeadModel",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "qwen3_moe": "Qwen3MoeForCausalLM",
    "llama": "LlamaForCausalLM",
    "mixtral": "MixtralForCausalLM",
    "deepseek": "DeepseekForCausalLM",
    "deepseek_v2": "DeepseekV2ForCausalLM",
    "deepseek_v3": "DeepseekV3ForCausalLM",
    "deepseek_v32": "DeepseekV32ForCausalLM",
}
SAFE_INTEGER = 2**53 - 1
MOE_FAMILIES = (
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
    "mixtral",
    "qwen3_moe",
)
MLA_FAMILIES = ("deepseek_v2", "deepseek_v3", "deepseek_v32")
MLA_KEYS = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
QUANT_METHODS = (None, "fp8", "gptq", "awq", "bitsandbytes", "compressed-tensors", "mxfp4", "mxfp8")
FFN_ACTIVATIONS = ("silu", "swiglu", "swish", "situ", "swigluoai")
BACKBONE_GATES = ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim")
LAYER_GATES = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
HEAD_GATES = ("num_attention_heads", "n_head", "num_heads", "n_heads")
VOCAB_GATES = ("vocab_size", "padded_vocab_size")


def has_mla_fields(cfg):
    return isinstance(cfg, dict) and all(cfg.get(key) is not None for key in MLA_KEYS)


def has_complete_backbone(cfg):
    if not isinstance(cfg, dict):
        return False
    hidden = any(type(cfg.get(key)) is int and cfg[key] > 0 for key in BACKBONE_GATES)
    layers = any(type(cfg.get(key)) is int and cfg[key] > 0 for key in LAYER_GATES)
    heads = any(type(cfg.get(key)) is int and cfg[key] > 0 for key in HEAD_GATES)
    vocab = any(type(cfg.get(key)) is int and cfg[key] > 0 for key in VOCAB_GATES)
    return hidden and layers and heads and vocab


def methods_authorized(cfg):
    """Accounting is authorized by implemented methods, not a family string."""
    family = cfg.get("model_type") if isinstance(cfg, dict) else None
    spec = cfg.get("linear_attn_config") if isinstance(cfg, dict) else None
    if isinstance(cfg, dict):
        if cfg.get("compress_ratios"):
            return False
        if cfg.get("hc_mult") and not (isinstance(spec, dict) and isinstance(spec.get("kda_layers"), list)):
            return False
    if isinstance(family, str) and family in SUPPORTED:
        return True
    if isinstance(spec, dict) and isinstance(spec.get("kda_layers"), list):
        return True
    if has_mla_fields(cfg):
        return True
    if not has_complete_backbone(cfg):
        return False
    return True


def integer(value, name, low=1, high=1048576):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def shared_expert_multiplicity(shared_width, expert_dim, counted_shared=0):
    """Map shared-expert width onto a count. Independent widths are out of scope."""
    counted = int(counted_shared or 0)
    if shared_width is None:
        return counted
    if type(shared_width) is not int or shared_width < 0:
        raise ValueError("shared expert width must be a non-negative integer")
    if shared_width == 0:
        if counted:
            raise ValueError("shared expert count conflicts with zero shared width")
        return 0
    if type(expert_dim) is not int or expert_dim < 1:
        raise ValueError("shared expert width requires moe_intermediate_size")
    if shared_width % expert_dim:
        raise ValueError("unsupported independent shared expert width")
    derived = shared_width // expert_dim
    if counted and counted != derived:
        raise ValueError("shared expert count does not match shared expert width")
    return derived


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    model_type = cfg.get("model_type")
    authorized = methods_authorized(cfg)
    if not authorized:
        raise ValueError(f"unsupported model_type: {model_type!r}; no reliable parameter total")
    for key in (
        "text_config",
        "vision_config",
        "compress_ratios",
        "layer_configs",
        "shared_layer_mapping",
        "layer_parameter_sharing",
        "num_hidden_groups",
    ):
        if key in cfg:
            raise ValueError(f"unsupported structure: {key}; no reliable parameter total")
    if cfg.get("hc_mult") is not None:
        integer(cfg["hc_mult"], "hc_mult", high=16)
    architectures = cfg.get("architectures", [])
    if (
        isinstance(model_type, str)
        and model_type in SUPPORTED
        and (
            not isinstance(architectures, list)
            or (architectures and architectures != [SUPPORTED[model_type]])
        )
    ):
        raise ValueError("architectures does not match supported model_type")
    for name, aliases, limit in (
        ("hidden_size", ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim"), 65536),
        ("num_hidden_layers", ("num_hidden_layers", "n_layer", "num_layers", "n_layers"), 4096),
        ("num_attention_heads", ("num_attention_heads", "n_head", "num_heads", "n_heads"), 4096),
        ("vocab_size", ("vocab_size", "padded_vocab_size"), 1048576),
        (
            "intermediate_size",
            ("intermediate_size", "n_inner", "ffn_hidden_size", "ffn_dim", "block_ff_dim"),
            1048576,
        ),
    ):
        values = [cfg[key] for key in aliases if key in cfg]
        integer(values[0] if values else None, name, high=limit)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"conflicting aliases for {name}")
    for key in (
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
        "kv_channels",
        "kv_lora_rank",
        "q_lora_rank",
        "v_head_dim",
        "moe_intermediate_size",
    ):
        if cfg.get(key) is not None:
            integer(cfg[key], key, high=65536)
    if cfg.get("head_dim") is not None and cfg["head_dim"] != 0:
        integer(cfg["head_dim"], "head_dim", high=65536)
    if cfg.get("qk_rope_head_dim") is not None:
        integer(cfg["qk_rope_head_dim"], "qk_rope_head_dim", low=0, high=65536)
    if cfg.get("qk_nope_head_dim") is not None:
        integer(cfg["qk_nope_head_dim"], "qk_nope_head_dim", low=0, high=65536)
    for key in ("max_position_embeddings", "seq_length", "sliding_window"):
        if cfg.get(key) is not None:
            integer(cfg[key], key, high=16777216)
    for key in (
        "attention_bias",
        "mlp_bias",
        "no_bias",
        "tie_word_embeddings",
        "use_sliding_window",
        "qk_norm",
        "use_qk_norm",
        "qkv_bias",
        "fp16",
        "bf16",
        "fp32",
    ):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg.get("hidden_act", "silu") not in FFN_ACTIVATIONS:
        raise ValueError("unsupported FFN activation for supported model families")
    if model_type != "qwen" and any(key in cfg for key in ("layer_norm_epsilon", "layer_norm_eps")):
        raise ValueError("unsupported LayerNorm override for RMSNorm model")
    if model_type == "qwen" and not cfg.get("no_bias"):
        raise ValueError("Qwen1 bias layout requires separate validation; only no_bias=true supported")
    if model_type != "qwen" and cfg.get("no_bias"):
        raise ValueError("unsupported no_bias override for model_type")
    if model_type == "qwen" and cfg["intermediate_size"] % 2:
        raise ValueError("Qwen1 intermediate_size must be even")
    for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
        if key in cfg:
            integer(cfg[key], key, low=2, high=2048)
            if isinstance(model_type, str) and model_type in SUPPORTED and model_type not in MOE_FAMILIES:
                raise ValueError("unsupported MoE override for dense architecture")
    for key in (
        "num_experts_per_tok",
        "experts_per_token",
        "num_experts_per_token",
        "moe_topk",
        "moe_top_k",
        "moe_k",
        "num_selected_experts",
    ):
        if key in cfg:
            integer(cfg[key], key, high=2048)
    for key in (
        "n_shared_experts",
        "num_shared_experts",
        "moe_num_shared_experts",
        "first_k_dense_replace",
        "moe_layer_start_index",
        "num_nextn_predict_layers",
    ):
        if key in cfg:
            integer(cfg[key], key, low=0, high=4096)
    freq = cfg.get("moe_layer_freq", 1)
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
    if isinstance(freq, list):
        if type(layers) is not int or len(freq) != layers or any(item not in (0, 1) for item in freq):
            raise ValueError("moe_layer_freq list must be 0/1 with length num_hidden_layers")
    else:
        integer(freq, "moe_layer_freq", high=4096)
    if "decoder_sparse_step" in cfg:
        integer(cfg["decoder_sparse_step"], "decoder_sparse_step", high=4096)
        if "moe_layer_freq" in cfg and cfg["moe_layer_freq"] != cfg["decoder_sparse_step"]:
            raise ValueError("conflicting moe_layer_freq and decoder_sparse_step")
    mlp_only = cfg.get("mlp_only_layers")
    if mlp_only is not None:
        if not isinstance(mlp_only, list) or any(type(i) is not int for i in mlp_only):
            raise ValueError("mlp_only_layers must be a list of layer indices")
        layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        if any(i < 0 or (type(layers) is int and i >= layers) for i in mlp_only):
            raise ValueError("mlp_only_layers index out of range")
    shared_width = next(
        (cfg[key] for key in ("shared_expert_intermediate_size", "shared_intermediate_size") if key in cfg),
        None,
    )
    counted_shared = cfg.get("n_shared_experts", cfg.get("num_shared_experts", cfg.get("moe_num_shared_experts")))
    if shared_width is not None or counted_shared:
        shared_expert_multiplicity(
            shared_width if any(key in cfg for key in ("shared_expert_intermediate_size", "shared_intermediate_size")) else None,
            cfg.get("moe_intermediate_size") or cfg.get("intermediate_size"),
            counted_shared or 0,
        )
    qcfg = cfg.get("quantization_config") or {}
    if not isinstance(qcfg, dict):
        raise ValueError("quantization_config must be an object")
    if qcfg.get("quant_method") not in QUANT_METHODS:
        raise ValueError("unsupported quantization method")
    if qcfg.get("bits") is not None:
        integer(qcfg["bits"], "quantization bits", high=16)
    if qcfg.get("group_size") is not None:
        integer(qcfg["group_size"], "quantization group_size", high=65536)
    if qcfg.get("quant_method") == "gptq":
        if model_type != "qwen" or qcfg.get("bits", 4) != 4 or qcfg.get("group_size", 128) != 128:
            raise ValueError("GPTQ closed-form only validated for Qwen1 4-bit group128")
        if cfg["hidden_size"] % 128 or (cfg["intermediate_size"] // 2) % 128:
            raise ValueError("GPTQ dimensions must align with validated group128 packing")
    if str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16") not in ("bfloat16", "float16", "float32"):
        raise ValueError("unsupported torch_dtype")
    if cfg.get("expert_dtype") is not None:
        raise ValueError("unsupported independent expert_dtype accounting")
    if has_mla_fields(cfg) and any(cfg.get(key) for key in ("index_head_dim", "index_n_heads", "index_topk")):
        for key in ("index_head_dim", "index_n_heads", "index_topk"):
            if cfg.get(key) is not None:
                integer(cfg[key], key, high=65536)
        if cfg.get("q_lora_rank") is not None:
            integer(cfg["q_lora_rank"], "q_lora_rank", high=65536)
        if cfg.get("index_kpool") is not None:
            integer(cfg["index_kpool"], "index_kpool", low=1, high=4096)
    elif model_type == "deepseek_v32":
        for key in ("index_head_dim", "index_n_heads", "index_topk", "q_lora_rank"):
            integer(cfg.get(key), key, high=65536)
    elif any(cfg.get(key) for key in ("index_head_dim", "index_n_heads", "index_topk")):
        if not isinstance(cfg.get("sparse_attention_config"), dict):
            raise ValueError("indexer fields require a sparse attention architecture")


def validate_match(match):
    bb, moe, mla = match["backbone"], match["moe"], match["mla"]
    integer(bb.num_kv_heads, "num_key_value_heads", high=bb.num_heads)
    if bb.num_heads % bb.num_kv_heads:
        raise ValueError("Q heads must be divisible by KV heads")
    if not mla and not any(key in match["cfg"] for key in ("head_dim", "kv_channels")):
        if bb.hidden_size % bb.num_heads:
            raise ValueError("hidden_size must be divisible by heads without head_dim")
    family = match["identity"].model_type
    if family in MLA_FAMILIES and not mla:
        raise ValueError("DeepSeek MLA architecture requires complete MLA dimensions")
    if family in ("deepseek", "deepseek_v2", "deepseek_v3", "deepseek_v32", "mixtral") and not moe:
        raise ValueError("MoE architecture requires expert configuration")
    if mla and family in SUPPORTED and family not in MLA_FAMILIES:
        raise ValueError("unsupported MLA override for model_type")
    if mla and bb.attention_bias:
        raise ValueError("unsupported MLA bias layout")
    if moe:
        integer(moe.n_routed, "experts", high=2048)
        integer(moe.top_k, "top_k", high=moe.n_routed)
        integer(moe.n_shared, "shared experts", low=0, high=2048)
        integer(moe.dense_prefix, "dense prefix", low=0, high=bb.num_layers)
        integer(moe.expert_dim, "expert dimension")
        if moe.topk_method not in (None, "greedy", "group_limited_greedy", "noaux_tc", "noaux-tc"):
            raise ValueError("unsupported router topk_method")
    integer(bb.mtp_layers, "MTP layers", low=0, high=16)
    if bb.mtp_layers and not (mla and moe):
        raise ValueError("MTP requires complete MLA and MoE methods")
    types = match["cfg"].get("layer_types")
    if types is not None:
        if not isinstance(types, list) or len(types) != bb.num_layers:
            raise ValueError("layer_types must have exactly num_hidden_layers entries")
        allowed = {
            "full_attention",
            "sliding_attention",
            "minimax_m3_sparse",
            "deepseek_sparse_attention",
        }
        if any(item not in allowed for item in types):
            raise ValueError("unsupported layer_types entry")
        if "sliding_attention" in types and (not bb.sliding_window or mla):
            raise ValueError("sliding layers require a window and standard attention")


def ensure_safe_numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            ensure_safe_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            ensure_safe_numbers(item)
    elif isinstance(value, bool):
        return
    elif isinstance(value, int):
        if abs(value) > SAFE_INTEGER:
            raise ValueError("analysis exceeds JSON safe integer/finite numeric range")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("analysis exceeds JSON safe integer/finite numeric range")
