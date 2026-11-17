from app.domain.specs import LayerAnalysis, ModelAnalysis, ModelSpec

ENGINE_VERSION = "planner-0.1.0"
FORMULA_VERSION = "dense-gqa-1.0.0"


def validate_legacy_config(config: dict) -> None:
    family = config.get("model_type")
    supported = {"qwen2": "Qwen2ForCausalLM", "llama": "LlamaForCausalLM"}
    if family not in supported or config.get("architectures") != [supported[family]]:
        raise ValueError("Only dense Qwen2ForCausalLM and LlamaForCausalLM are supported")
    if config.get("auto_map") or config.get("quantization_config"):
        raise ValueError("Custom remote code and quantized configurations are not supported")
    if any(config.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ValueError("MoE is not supported")
    if config.get("use_sliding_window") or config.get("rope_scaling"):
        raise ValueError("Sliding window and scaled RoPE require a separate formula version")
    if config.get("hidden_act", "silu") != "silu" or config.get("mlp_bias"):
        raise ValueError("Only bias-free SwiGLU MLP is supported")
    fields = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    )
    for field in fields:
        value = config.get(field)
        if type(value) is not int or not 1 <= value <= 2_000_000:
            raise ValueError(f"{field} must be a positive integer <= 2000000")
    h, heads, kv = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"]
    if h % heads or heads % kv:
        raise ValueError("Hidden size and attention/KV head dimensions must divide evenly")
    if config.get("head_dim", h // heads) != h // heads:
        raise ValueError("Custom head_dim is not supported")
    if config.get("attention_bias") and family == "llama":
        raise ValueError("Llama attention bias is not supported in this formula version")
    if "tie_word_embeddings" in config and type(config["tie_word_embeddings"]) is not bool:
        raise ValueError("tie_word_embeddings must be boolean")


def legacy_parameter_breakdown(config: dict) -> dict[str, int]:
    validate_legacy_config(config)
    h, layers = config["hidden_size"], config["num_hidden_layers"]
    kv = config["num_key_value_heads"] * (h // config["num_attention_heads"])
    bias = h + 2 * kv if config["model_type"] == "qwen2" else 0
    return {
        "embedding": config["vocab_size"] * h,
        "attention": layers * (2 * h * h + 2 * h * kv + bias),
        "mlp": layers * 3 * h * config["intermediate_size"],
        "normalization": (2 * layers + 1) * h,
        "output_head": 0 if config.get("tie_word_embeddings", False) else config["vocab_size"] * h,
    }


def legacy_analyze(model: ModelSpec) -> ModelAnalysis:
    parts = legacy_parameter_breakdown(model.raw_config)
    h, v, f = model.hidden_size, model.vocab_size, model.intermediate_size
    kv = model.kv_heads * (h // model.attention_heads)
    bias = model.architecture == "qwen2"
    layer_rows = [
        ("Token embedding", v * h, "[B, S]", f"[B, S, {h}]"),
        ("Q projection / layer", h * h + (h if bias else 0), f"[B, S, {h}]", f"[B, S, {h}]"),
        ("K projection / layer", h * kv + (kv if bias else 0), f"[B, S, {h}]", f"[B, S, {kv}]"),
        ("V projection / layer", h * kv + (kv if bias else 0), f"[B, S, {h}]", f"[B, S, {kv}]"),
        ("O projection / layer", h * h, f"[B, S, {h}]", f"[B, S, {h}]"),
        ("SwiGLU MLP / layer", 3 * h * f, f"[B, S, {h}]", f"[B, S, {h}]"),
        ("RMSNorm x2 / layer", 2 * h, f"[B, S, {h}]", f"[B, S, {h}]"),
        ("Final RMSNorm", h, f"[B, S, {h}]", f"[B, S, {h}]"),
        ("LM head (unique weights)", parts["output_head"], f"[B, S, {h}]", f"[B, S, {v}]"),
    ]
    return ModelAnalysis(
        model_id=model.id,
        parameters=sum(parts.values()),
        parameter_breakdown=parts,
        layers=[
            LayerAnalysis(name=n, parameters=p, input_shape=i, output_shape=o) for n, p, i, o in layer_rows
        ],
        engine_version=ENGINE_VERSION,
        formula_version=FORMULA_VERSION,
        assumptions=[
            "Exact unique parameter count for the supported configuration, not a weight-file inspection.",
            "Qwen2 uses Q/K/V biases and no O projection bias. Llama uses bias-free projections.",
            "Tied embedding and LM-head weights are counted once; LM-head compute still executes.",
            "Rows marked / layer describe one repeated transformer block; total includes all layers.",
        ],
    )


def validate_config(config: dict) -> None:
    from app.engines.model_structure import inspect_model

    inspect_model(config)


def parameter_breakdown(config: dict) -> dict[str, int]:
    from app.engines.model_structure import breakdown, inspect_model

    return breakdown(inspect_model(config))


def analyze(model: ModelSpec) -> ModelAnalysis:
    from app.engines.model_structure import breakdown, inspect_model

    try:
        validate_legacy_config(model.raw_config)
    except ValueError:
        result = inspect_model(model.raw_config)
        rows = []
        for group in result["groups"]:
            item = next(t for t in result["templates"] if t["id"] == group["template_id"])
            for row in item["modules"]:
                rows.append(LayerAnalysis(
                    name=f"{item['label']} / {row['id']} x{group['count']}",
                    parameters=row["params"]*group["count"],
                    input_shape=row["input_shape"], output_shape=row["output_shape"],
                ))
        rows.extend(LayerAnalysis(name=row["id"], parameters=row["params"],
                                  input_shape=row["input_shape"], output_shape=row["output_shape"])
                    for row in result["shared_modules"])
        return ModelAnalysis(
            model_id=model.id, parameters=result["totals"]["total"], parameter_breakdown=breakdown(result),
            layers=rows, engine_version=result["engine_version"], formula_version=result["formula_version"],
            assumptions=[
                "Config-derived logical parameter inventory, not a weight-file inspection.",
                "Main model includes resident vision weights; optional MTP excluded.",
                "Routed experts: resident count differs from per-token active count.",
                "Checkpoint elements can include quantization scales, integer tables and duplicated shared heads.",
                "Non-matmul activations, mask, RoPE and normalization are not GEMM parameters.",
            ] + model.support_notes,
        )
    return legacy_analyze(model)
