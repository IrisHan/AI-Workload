"""Static memory components; no activation peak or performance claims."""
from math import prod

from app.engines.cache_layout import cache_components
from app.engines.checkpoint_paths import module_paths
from app.engines.morphology import module_ignored
from app.lib.spec_studio.validation import integer


def quantization_config(config):
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        return quant
    inner = config.get("text_config")
    nested = inner.get("quantization_config") if isinstance(inner, dict) else None
    return nested if isinstance(nested, dict) else {}


def uses_quant_payload_overlay(config):
    family = str(config.get("model_type") or "")
    inner = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    inner_family = str(inner.get("model_type") or "")
    method = quantization_config(config).get("quant_method")
    if method in ("compressed-tensors", "mxfp4", "mxfp8"):
        return True
    if method not in ("fp8", "awq", "gptq"):
        return False
    names = (family, inner_family)
    return any(
        name.startswith(prefix)
        for name in names
        for prefix in ("qwen", "kimi", "minimax", "glm", "ernie", "gpt_oss")
    )


def compressed_tensor_bits(quant):
    groups = quant.get("config_groups") if isinstance(quant, dict) else None
    if isinstance(groups, dict):
        for group in groups.values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict) and type(weights.get("num_bits")) is int:
                return weights["num_bits"]
    fmt = str((quant or {}).get("format") or "").lower()
    if "int4" in fmt or "mxfp4" in fmt:
        return 4
    return None


def _group_size(quant):
    groups = quant.get("config_groups") if isinstance(quant, dict) else None
    if isinstance(groups, dict):
        for group in groups.values():
            weights = group.get("weights") if isinstance(group, dict) else None
            if isinstance(weights, dict) and type(weights.get("group_size")) is int:
                return weights["group_size"], weights
    if isinstance(quant, dict) and type(quant.get("group_size")) is int:
        return quant["group_size"], quant
    return None, {}


def _scale_elem(quant, weights):
    fmt = str((quant or {}).get("format") or "").lower()
    dtype = str((weights or {}).get("scale_dtype") or "").lower()
    if "mxfp4" in fmt or "uint8" in dtype:
        return 1
    return 2


def _aux_bytes(params, bits, group_size, quant, weights, method, shape, count):
    if type(group_size) is not int or group_size <= 0 or params <= 0:
        return 0
    groups = (params + group_size - 1) // group_size
    extra = groups * _scale_elem(quant, weights)
    if method == "compressed-tensors":
        if weights.get("symmetric") is False:
            extra += groups
    elif method == "awq" and quant.get("zero_point", True):
        extra += params * bits // 8 // group_size
    elif method == "gptq":
        extra += params * bits // 8 // group_size
        if shape and len(shape) > 1:
            extra += shape[1] * 4 * (count or 1)
    return extra


def integer_buffer_bytes(config):
    if config.get("model_type") != "deepseek_v4":
        return 0
    layers = config.get("num_hash_layers") or 0
    if type(layers) is not int or layers <= 0:
        return 0
    return layers * config["vocab_size"] * config["num_experts_per_tok"] * 4


def module_payload_bytes(row, config, elem, *, role=None, layer=None, expert=None, block=None):
    quant = quantization_config(config)
    params = row["params"]
    if not uses_quant_payload_overlay(config):
        return params * elem
    method = quant.get("quant_method")
    ignore = quant.get("ignore") or quant.get("modules_to_not_convert") or []
    paths = row.get("checkpoint_paths") or module_paths(row, config, role=role)
    paths = [path.format(layer=layer if layer is not None else "{layer}",
                         expert=expert if expert is not None else "{expert}",
                         block=block if block is not None else "{block}") for path in paths]
    if ignore and module_ignored(row["id"], row.get("category") or "", ignore, role=role, paths=paths):
        return params * elem
    if row.get("category") in {"norm", "embedding", "residual", "position"} or not row.get("matrix_params"):
        return params * elem
    # Grouped inventory may store a Linear's bias in the same row. Bias is
    # unquantized even though matrix_params marks the row as a matrix operator.
    matrix_params = prod(row["shape"]) * (row.get("count") or 1)
    bias_bytes = max(0, params - matrix_params) * elem
    params = min(params, matrix_params)
    if method == "fp8" or method == "mxfp8":
        return params + bias_bytes
    bits = quant.get("bits")
    if method == "mxfp4":
        bits = 4
    elif method in ("awq", "gptq"):
        bits = bits if type(bits) is int else 4
    elif method == "compressed-tensors":
        bits = compressed_tensor_bits(quant)
    if type(bits) is not int or bits not in (2, 3, 4, 8):
        return params * elem + bias_bytes
    group_size, weights = _group_size(quant)
    packed_n = params
    if type(group_size) is int and group_size > 0:
        packed_n = ((params + group_size - 1) // group_size) * group_size
    packed = packed_n * bits // 8
    packed += _aux_bytes(params, bits, group_size, quant, weights, method, row.get("shape"), row.get("count") or 1)
    return packed + bias_bytes


def _walk_payload(analysis, config, elem):
    total_bytes = 0
    groups = {}
    for group in analysis.get("groups") or []:
        groups.setdefault(group["template_id"], []).extend(
            i for start, stop in group["ranges"] for i in range(start, stop))
    quant = quantization_config(config)
    ignore = quant.get("ignore") or quant.get("modules_to_not_convert") or []
    split_experts = any(".experts." in str(p) and any(ch.isdigit() for ch in str(p).split(".experts.", 1)[1])
                        for p in ignore)
    branch_ids = {item.get("id") for item in analysis.get("branches") or []}
    for template in analysis.get("templates") or []:
        layers = groups.get(template["id"], [None])
        for row in template.get("modules") or []:
            copies = row.get("count") or 1
            if split_experts and row["id"].startswith("routed_expert.") and copies > 1:
                single = {**row, "count": 1, "params": row["params"] // copies}
                for layer in layers:
                    total_bytes += sum(module_payload_bytes(single, config, elem, layer=layer, expert=i)
                                       for i in range(copies))
            else:
                total_bytes += sum(module_payload_bytes(row, config, elem, layer=layer) for layer in layers)
    for row in analysis.get("shared_modules") or []:
        if row["id"] in branch_ids or row["id"] in {"vision"} or row.get("category") == "mtp":
            continue
        total_bytes += module_payload_bytes(row, config, elem)
    for branch in analysis.get("branches") or []:
        if not branch.get("included", True):
            continue
        copies = branch.get("count") or 1
        role = branch.get("role")
        for row in branch.get("modules") or []:
            count = row.get("count") or 1
            if role == "vision" and count > 1 and "vision.block." in row["id"]:
                single = {**row, "count": 1, "params": row["params"] // count}
                total_bytes += sum(module_payload_bytes(single, config, elem, role=role, block=i)
                                   for i in range(count)) * copies
            else:
                total_bytes += module_payload_bytes(row, config, elem, role=role) * copies
    return total_bytes


def weight_payload_bytes(total, config, elem, analysis=None):
    """Resident weight estimate. Logical N is unchanged."""
    if analysis and uses_quant_payload_overlay(config):
        return _walk_payload(analysis, config, elem)
    quant = quantization_config(config)
    method = quant.get("quant_method")
    if uses_quant_payload_overlay(config):
        if method == "fp8":
            return total
        if method in ("awq", "gptq"):
            bits = quant.get("bits", 4)
            if type(bits) is int and bits in (2, 3, 4, 8):
                return total * bits // 8
        if method == "compressed-tensors":
            bits = compressed_tensor_bits(quant)
            if type(bits) is int and bits in (2, 3, 4, 8):
                return total * bits // 8
    return total * elem


def memory_components(analysis, config, *, batch_size=1, context_length=4096, dtype="bf16",
                      include_training_state=False):
    b = integer(batch_size, "batch_size", high=4096)
    s = integer(context_length, "context_length", high=16777216)
    if s > analysis["backbone"]["context_length"]:
        raise ValueError("requested context exceeds model capacity")
    if dtype not in ("bf16", "fp32"):
        raise ValueError("static memory dtype must be bf16 or fp32")
    elem = 2 if dtype == "bf16" else 4
    cache_rows = cache_components(analysis, b, s, elem)
    weights = weight_payload_bytes(analysis["totals"]["total"], config, elem, analysis)
    buffers = integer_buffer_bytes(config)
    gradients = analysis["totals"]["total"] * elem if include_training_state else 0
    optimizer = analysis["totals"]["total"]*(12 if elem == 2 else 8) if include_training_state else 0
    cache = 0 if include_training_state else sum(r["total_bytes"] for r in cache_rows)
    overlay = uses_quant_payload_overlay(config)
    assumptions = [
        "Logical unquantized BF16/FP32 resident weights; not source checkpoint quantized bytes."
        if not overlay else
        "Quantized checkpoint: Linear payloads are packed bits/8 with group alignment; "
        "ignore-list modules stay at scenario dtype; group scales, asymmetric zp, AWQ/GPTQ qzeros "
        "and GPTQ g_idx are closed-form extras. Unknown packing layouts remain excluded.",
        "Static memory subtotal only; activations, temporary attention scores, workspace and allocator reserve excluded.",
        "All parameters resident; expert top-k reduces active compute, not weight residency.",
        "Training state assumes full-parameter Adam, same-dtype gradients, FP32 moments and BF16 FP32 master copy; frozen vision parameters not excluded.",
        "KV uses selected dtype; recurrent and compression staging states explicitly use FP32.",
        "V4 expert hash tables are integer_buffer_bytes (int32 upper bound), not floating parameter N.",
        "Batch and context are scenario inputs; no GPU fit or latency claim without a complete runtime peak.",
    ]
    version = "static-memory-components-1.3.0"
    return dict(
        formula_version=version, kind="analytical", calibrated=False,
        input_snapshot=dict(batch_size=b, context_length=s, dtype=dtype,
                            include_training_state=include_training_state, source=analysis["source"],
                            structure_formula=analysis["formula_version"],
                            include_mtp=analysis["serving"]["include_mtp"]),
        weight_bytes=weights, kv_cache_bytes=cache, gradient_bytes=gradients, optimizer_bytes=optimizer,
        integer_buffer_bytes=buffers,
        static_total_bytes=weights+cache+gradients+optimizer+buffers,
        activation_bytes=None, peak_memory_bytes=None, cache_components=cache_rows if not include_training_state else [],
        assumptions=assumptions,
    )
