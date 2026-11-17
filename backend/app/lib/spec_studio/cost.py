"""Versioned analytical operator costs, deliberately separate from measurements."""

import math

from app.engines.cache_layout import retained_length, template_cache
from app.engines.morphology import COST_ATTN, COST_FFN

from .validation import SAFE_INTEGER, ensure_safe_numbers, integer


def _safe_mul(*values):
    total = 1
    for value in values:
        if type(value) is not int or value < 0:
            raise ValueError("operator cost factors must be non-negative integers")
        total *= value
    return total


def _safe_add(*values):
    total = 0
    for value in values:
        total += value
    return total


def _qty(value):
    """Analytical FLOPs/bytes may exceed 2^53; JSON numbers then become IEEE-754 doubles."""
    if isinstance(value, int) and abs(value) > SAFE_INTEGER:
        return float(value)
    return value


def _causal_pairs(phase, sequence, context, window=None, compressed=None):
    if phase == "decode":
        length = context if compressed is None else max(1, context // compressed)
        return retained_length(length, window)
    length = sequence if compressed is None else max(1, sequence // compressed)
    if window and window > 0:
        if length <= window:
            return length * (length + 1) // 2
        return window * (window + 1) // 2 + window * (length - window)
    return length * (length + 1) // 2


def operator_cost(analysis, *, phase, batch_size, sequence_length, context_length, dtype, hardware):
    if phase not in {"prefill", "decode"} or dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError("unsupported phase or compute dtype")
    b = integer(batch_size, "batch_size", high=4096)
    s = integer(sequence_length, "sequence_length", high=16777216)
    context = integer(context_length, "context_length", high=16777216)
    capacity = analysis["backbone"]["context_length"]
    if (s if phase == "prefill" else context) > capacity:
        raise ValueError("requested context exceeds model capacity")
    if analysis["serving"]["include_mtp"]:
        raise ValueError("MTP speculative runtime cost is not calibrated")
    if analysis["status"] != "supported":
        raise ValueError("unsupported analysis")
    for key in ("peak_tflops", "bandwidth_bytes_s", "memory_bytes"):
        value = hardware.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"hardware {key} must be finite and positive")
    if not hardware.get("source") or not hardware.get("version"):
        raise ValueError("hardware source and version required")
    elem = 4 if dtype == "fp32" else 2
    tokens = b * s if phase == "prefill" else b
    rows = []
    kv_bytes = 0
    activation = 0
    skipped_vision = False
    for group in analysis["groups"]:
        template = next(t for t in analysis["templates"] if t["id"] == group["template_id"])
        attn, ffn = template["attn_kind"], template["ffn_kind"]
        if attn not in COST_ATTN or ffn not in COST_FFN:
            raise ValueError(f"operator cost has no method for {attn}+{ffn}")
        bb, count = template["backbone"], group["count"]
        h, heads, c = bb["hidden_size"], bb["num_heads"], bb["head_dim"]
        cache = template_cache(template, b, s if phase == "prefill" else context, elem)
        retained = cache["total_bytes"] * count
        kv_bytes = _safe_add(kv_bytes, retained)
        for module in template["modules"]:
            if not module["matrix_params"]:
                continue
            matrix = module["matrix_params"]
            rows.append(
                dict(
                    group_id=group["id"],
                    operation=module["id"],
                    count=count,
                    flops=_safe_mul(2, tokens, matrix, count),
                    traffic_bytes=(matrix + tokens * (module["shape"][0] + module["shape"][1] if len(module["shape"]) > 1 else 0)) * elem * count
                    if len(module.get("shape") or []) > 1 else (matrix + tokens * h) * elem * count,
                    formula="2*B*query_tokens*matrix_params*layers; MoE matrix_params already use top-k active experts",
                )
            )
        if attn in {"KDA", "GATED_DELTA"}:
            state = template.get("recurrent")
            if state is None:
                raise ValueError("Snapshot lacks recurrent dimensions; create a new structure snapshot")
            rec = _safe_mul(2, tokens, state["heads"], state["key_dim"], state["value_dim"], count)
            rows.append(dict(
                group_id=group["id"], operation="recurrent_state", count=count, flops=rec,
                traffic_bytes=2 * tokens * state["heads"] * state["key_dim"] * state["value_dim"] * 4 * count,
                formula="2*B*query_tokens*recurrent_heads*key_dim*value_dim*layers; FP32 state read/write traffic",
            ))
        else:
            window = bb.get("sliding_window") if attn == "SWA" else None
            ratio = template.get("compress_ratio") or None
            compressed = ratio if attn in {"CSA", "HCA"} and ratio else None
            pairs = _causal_pairs(phase, s, context, window, compressed)
            mla = template.get("mla") or {}
            qk_c = (mla["qk_nope_head_dim"] + mla["qk_rope_head_dim"]) if mla else c
            v_c = mla["v_head_dim"] if mla else c
            qk_flops = _safe_mul(2, b, heads, pairs, qk_c, count)
            av_flops = _safe_mul(2, b, heads, pairs, v_c, count)
            for operation, flops in (("qk_matmul", qk_flops), ("av_matmul", av_flops)):
                rows.append(dict(
                    group_id=group["id"], operation=operation, count=count, flops=flops,
                    traffic_bytes=retained // 2 + tokens * h * elem * count,
                    formula="2*B*heads*causal_pairs*head_dim*layers; MLA uses nope/rope and v dims; CSA/HCA compress sequence",
                ))
        activation = max(activation, tokens * elem * (2 * h + 2 * (bb.get("intermediate_size") or 0)))
    for branch in analysis.get("branches") or []:
        if branch.get("role") in {"vision",} or branch.get("attn_kind") == "VISION":
            skipped_vision = True
            continue
    for module in analysis["shared_modules"]:
        if module["id"] == "lm_head":
            outf, inf = module["shape"]
            rows.append(
                dict(
                    group_id="shared",
                    operation="lm_head",
                    count=1,
                    flops=_safe_mul(2, tokens, inf, outf),
                    traffic_bytes=(inf * outf + tokens * (inf + outf)) * elem,
                    formula="2*B*query_tokens*hidden*vocab (even when tied)",
                )
            )
            activation += tokens * outf * elem
    flops = _qty(_safe_add(*(row["flops"] for row in rows)) if rows else 0)
    weight_bytes = analysis["totals"]["total"] * elem
    traffic = _qty(sum(row["traffic_bytes"] for row in rows))
    kv_bytes = _qty(kv_bytes)
    activation = _qty(activation)
    peak, bandwidth = hardware["peak_tflops"] * 1e12, hardware["bandwidth_bytes_s"]
    for row in rows:
        row["flops"] = _qty(row["flops"])
        row["traffic_bytes"] = _qty(row["traffic_bytes"])
        row["latency_bound_ms"] = max(row["flops"] / peak, row["traffic_bytes"] / bandwidth) * 1000
    assumptions = [
        "Analytical estimate, not measured/calibrated latency; 2 FLOPs per MAC.",
        "Causal triangular prefill and single-query decode; context_length includes current token.",
        "GEMM uses matrix_params (MoE routed experts already top-k active). Non-matmul excluded.",
        "MLA/DSA use latent or indexer dims; KDA/Gated DeltaNet use recurrent state, not S^2.",
        "SWA uses sliding window pairs; CSA/HCA compress the KV sequence by compress_ratio.",
        "Weights counted uniquely for residency; traffic per operator may reload shared tensors.",
        "No launch overhead, interconnect, allocator reserve, queueing, runtime fusion, or kernel calibration.",
        "Compute dtype also sets KV/activation dtype for this scenario; does not modify imported model facts.",
        "FLOPs and HBM traffic are analytical IEEE-754 doubles; exact parameter inventories stay inside 2^53-1.",
    ]
    if skipped_vision:
        assumptions.append("Vision encoder FLOPs omitted; they require an image-grid token count, not text sequence_length.")
    result = dict(
        formula_version="operator-cost-2.2.0",
        kind="analytical",
        calibrated=False,
        input_snapshot=dict(
            phase=phase,
            batch_size=b,
            sequence_length=s,
            context_length=context,
            dtype=dtype,
            hardware=hardware,
            source=analysis["source"],
            analysis_formula=analysis["formula_version"],
            morphology=analysis.get("morphology"),
        ),
        rows=rows,
        flops=flops,
        traffic_bytes=traffic,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_bytes,
        activation_bytes=activation,
        total_memory_bytes=weight_bytes + kv_bytes + activation,
        memory_fits=weight_bytes + kv_bytes + activation <= hardware["memory_bytes"],
        arithmetic_intensity=flops / traffic if traffic else 0,
        latency_bound_ms=max(flops / peak, traffic / bandwidth) * 1000 if traffic else flops / peak * 1000,
        assumptions=assumptions,
    )
    ensure_safe_numbers(result)
    return result
