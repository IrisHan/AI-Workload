"""Explicit per-layer overrides with stable residual width and exact group ledgers."""

from copy import deepcopy

from .validation import ensure_safe_numbers, integer, validate_config

OVERRIDES = {
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_bias",
    "mlp_bias",
    "qk_norm",
    "torch_dtype",
    "sliding_window",
    "use_sliding_window",
}
BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def inspect_heterogeneous(cfg, **options):
    from .grouped import config_hash, inspect_grouped

    base = deepcopy(cfg)
    overrides = base.pop("layer_configs")
    validate_config(base)
    count = integer(base.get("num_hidden_layers"), "num_hidden_layers", high=4096)
    if not isinstance(overrides, list) or len(overrides) != count:
        raise ValueError("layer_configs must have exactly num_hidden_layers entries")
    if base["model_type"] not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("layer_configs currently requires a dense standard-attention family")
    if base.get("quantization_config") or base.get("num_nextn_predict_layers"):
        raise ValueError("layer_configs with quantization or MTP requires a separate model contract")
    root = inspect_grouped(base, **options)
    templates, groups, execution, by_signature, by_config = [], [], [], {}, {}
    template_bytes = {}
    for index, override in enumerate(overrides):
        if not isinstance(override, dict) or set(override) - OVERRIDES:
            raise ValueError(
                f"invalid layer_configs keys at layer {index}; residual hidden size must remain fixed"
            )
        local = {**base, **override, "num_hidden_layers": 1}
        if "layer_types" in base:
            local["layer_types"] = [base["layer_types"][index]]
        key = config_hash(local)
        if key not in by_config:
            if len(by_config) >= 256:
                raise ValueError("at most 256 unique layer configurations per analysis")
            by_config[key] = inspect_grouped(local, kv_dtype=options.get("kv_dtype", "bf16"))
        current = by_config[key]
        template = current["templates"][0]
        signature = template["signature"]
        if signature not in by_signature:
            group_id = f"g{len(groups) + 1}"
            by_signature[signature] = group_id
            template = deepcopy(template)
            templates.append(template)
            groups.append(
                dict(
                    id=group_id,
                    template_id=template["id"],
                    count=0,
                    ranges=[],
                    metrics={key: 0 for key in template["metrics"]},
                )
            )
            template_bytes[template["id"]] = current["serving"]["kv_layout"]["bytes_per_token"]
        group_id = by_signature[signature]
        group = next(group for group in groups if group["id"] == group_id)
        group["count"] += 1
        for key, value in template["metrics"].items():
            group["metrics"][key] += value
        if group["ranges"] and group["ranges"][-1][1] == index:
            group["ranges"][-1][1] += 1
        else:
            group["ranges"].append([index, index + 1])
        if execution and execution[-1]["group_id"] == group_id:
            execution[-1]["stop_exclusive"] = index + 1
        else:
            execution.append(dict(group_id=group_id, start=index, stop_exclusive=index + 1))
    root.update(
        templates=templates,
        groups=groups,
        execution=execution,
        branches=[],
        formula_version="heterogeneous-1.0.0",
    )
    root["source"]["config_sha256"] = options.get("config_sha256") or config_hash(cfg)
    root["source"]["configuration_contract"] = "layer_configs-1.0"
    totals = root["totals"]
    totals.update(attn=0, ffn_total=0, ffn_active=0, other=base["hidden_size"])
    weights = sum(row["params"] * BYTES[row["dtype"]] for row in root["shared_modules"])
    dtype_bytes = {}
    matrix = sum(row["matrix_params"] for row in root["shared_modules"])
    kv = 0
    for row in root["shared_modules"]:
        dtype_bytes[row["dtype"]] = dtype_bytes.get(row["dtype"], 0) + row["params"] * BYTES[row["dtype"]]
    for group in groups:
        template = next(template for template in templates if template["id"] == group["template_id"])
        matrix += group["metrics"]["matrix_params"]
        kv += template_bytes[template["id"]] * group["count"]
        for row in template["modules"]:
            n = group["count"]
            category = row["category"]
            if category == "attention":
                totals["attn"] += row["params"] * n
            elif category == "ffn":
                totals["ffn_total"] += row["params"] * n
                totals["ffn_active"] += row["active_params"] * n
            else:
                totals["other"] += row["params"] * n
            size = row["params"] * BYTES[row["dtype"]] * n
            weights += size
            dtype_bytes[row["dtype"]] = dtype_bytes.get(row["dtype"], 0) + size
    totals["total"] = sum(row["params"] for row in root["shared_modules"]) + sum(
        g["metrics"]["params_total"] for g in groups
    )
    totals["active"] = totals["total"]
    totals["active_body"] = totals["total"] - totals["embed"] - totals["lm_head"]
    totals["bytes_by_dtype"] = dtype_bytes
    root["serving"].update(
        nparams_total=totals["total"],
        nparams_active=totals["active"],
        nparams_active_body=totals["active_body"],
        weight_bytes=weights,
        weight_bytes_logical=weights,
        weight_bytes_gib=weights / 1024**3,
        weight_bytes_source="per-module-dtype-estimate",
        flop_gemm_decode=2 * matrix,
        index_total_size=None,
        hub_used_storage=None,
    )
    root["serving"]["kv_layout"].update(
        bytes_per_token=kv, formula="sum(layer KV width * explicit KV dtype bytes)"
    )
    root["diagnostics"].append(
        dict(
            severity="info",
            code="configuration.layer_overrides",
            message="Explicit layer_configs extension; residual width fixed. Checkpoint byte metadata is not reused for overridden tensors.",
        )
    )
    ensure_safe_numbers(root)
    from app.engines.morphology import attach_morphology
    return attach_morphology(root, cfg)
