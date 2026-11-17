"""Validate and adapt frozen Model Center structures without re-parsing families."""
from math import prod

from .ir import Layer, ModelIR, Operator, PlanningError, Weight

SUPPORTED_ATTENTION = {"MHA", "GQA", "MQA", "SWA"}
METHOD_GAPS = {
    "MLA": ("W13a", "Missing absorbed/unabsorbed runtime layout, reconstruction workspace and kernel timing"),
    "DSA": ("W13b", "Missing sparse index selection execution, per-request retention and workspace contract"),
    "MSA": ("W13b", "Missing shared index storage identity and selection schedule"),
    "KDA": ("W13c", "Missing chunk-prefill workspace lifetime and convolution tensor partition contract"),
    "GATED_DELTA": ("W13c", "Missing chunk-prefill temporary program and key/value head partition contract"),
    "CSA": ("W13d", "Missing compression execution temporaries and cross-layer cache storage identity"),
    "HCA": ("W13d", "Missing shared compression state identity and retention execution contract"),
}


def capability_issues(model, analysis):
    issues = []

    def issue(code, message, path, task=None):
        issues.append({"code": code, "message": message, "field_path": path, "task": task,
                       "status": "missing_source" if task else "unsupported"})

    cfg = model.get("raw_config", {})
    cfg = cfg.get("text_config") or cfg
    if cfg.get("quantization_config") or model.get("raw_config", {}).get("quantization_config"):
        issue("QUANTIZED_SOURCE", "Quantized checkpoint requires an explicit load/packing/kernel policy",
              "model.quantization_config")
    if model.get("dtype") not in {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        issue("SOURCE_DTYPE_UNKNOWN", "Source weight dtype is not established", "model.dtype")
    if analysis.get("branches") or analysis.get("serving", {}).get("include_mtp"):
        issue("UNSUPPORTED_BRANCH", "Visual/MTP branch runtime execution is not defined", "structure.branches")
    for template in analysis.get("templates", []):
        kind = template["attn_kind"]
        path = f"structure.templates.{template['id']}"
        if kind not in SUPPORTED_ATTENTION:
            task, reason = METHOD_GAPS.get(kind, ("W13", f"No execution profile for {kind}"))
            issue("UNSUPPORTED_METHOD", reason, path + ".attn_kind", task)
        if template["ffn_kind"] not in {"DENSE", "MOE"}:
            issue("UNSUPPORTED_METHOD", "FFN runtime layout is not defined", path + ".ffn_kind", "W13")
        if template.get("backbone", {}).get("activation", "silu") not in {"silu", "swiglu", "gelu"}:
            issue("UNSUPPORTED_ACTIVATION", "Activation runtime profile is not defined", path + ".activation")
    return issues


def _positive(value, path):
    if type(value) is not int or value <= 0:
        raise PlanningError("SOURCE_MISMATCH", f"Expected positive integer at {path}", path)
    return value


def weight_from_module(row, scope):
    name = row["id"]
    shape = tuple(_positive(n, f"{scope}/{name}.shape") for n in row["shape"])
    count = _positive(row.get("count", 1), f"{scope}/{name}.count")
    alias = row.get("shared_with")
    numel = row["params"]
    bias = 0 if alias else numel // count - prod(shape)
    if numel % count or bias < 0 or (bias and (len(shape) != 2 or bias != shape[0])):
        raise PlanningError("SOURCE_MISMATCH", "Unrecognized weight/bias accounting", f"{scope}/{name}")
    routed = name.startswith("routed_expert.")
    tail = name.rsplit(".", 1)[-1]
    if name in {"embed", "lm_head"}:
        partition = "vocab"
    elif tail in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
        partition = "column"
    elif tail in {"o_proj", "down_proj"}:
        partition = "row"
    elif name in {"attn_norm", "ffn_norm", "final_norm", "router", "router_correction",
                  "q_norm", "k_norm", "shared_expert_gate"}:
        partition = "replicate"
    else:
        raise PlanningError("UNSUPPORTED_METHOD", "No declared partition rule for weight", f"{scope}/{name}")
    dtype = row.get("dtype")
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise PlanningError("QUANTIZED_SOURCE", f"Unsupported storage dtype {dtype}", f"{scope}/{name}")
    return Weight(
        storage_id=f"{scope}/{name}", module_id=name, shape=shape, count=count, bias=bias,
        dtype=dtype, partition=partition, routed=routed,
        alias_of=f"{scope}/{alias}" if alias else None,
        checkpoint_paths=tuple(row.get("checkpoint_paths", [])),
    )


def normalize(model, snapshot):
    analysis = snapshot["analysis"]
    if snapshot["model_id"] != model["id"] or snapshot.get("model_version") != model["version"]:
        raise PlanningError("SOURCE_MISMATCH", "Structure does not belong to selected model version")
    issues = capability_issues(model, analysis)
    if issues:
        raise PlanningError(issues[0]["code"], issues[0]["message"], issues[0]["field_path"],
                            {"issues": issues})
    templates = {t["id"]: t for t in analysis["templates"]}
    groups = {g["id"]: g for g in analysis["groups"]}
    layers = []
    seen = set()
    for segment in analysis["execution"]:
        group = groups[segment["group_id"]]
        template = templates[group["template_id"]]
        bb = template["backbone"]
        for index in range(segment["start"], segment["stop_exclusive"]):
            if index in seen or index != len(layers):
                raise PlanningError("SOURCE_MISMATCH", "Execution layers overlap, are unordered or have gaps")
            seen.add(index)
            scope = f"{snapshot['id']}/{index}"
            weights = tuple(weight_from_module(row, scope) for row in template["modules"])
            moe = template.get("moe") or {}
            operators = tuple(
                Operator(id=f"{scope}/{node['id']}", method_id=node["id"],
                         weight_refs=(f"{scope}/{node['module_id']}",) if node.get("module_id") else (),
                         inputs=tuple(f"{scope}/{edge['source']}:out" for edge in template["edges"]
                                      if edge["target"] == node["id"]),
                         output=f"{scope}/{node['id']}:out", source_node_id=node["id"])
                for node in template["nodes"]
            )
            layer = Layer(
                index=index, instance_id=scope, template_id=template["id"],
                attention=template["attn_kind"], ffn=template["ffn_kind"],
                hidden=_positive(bb["hidden_size"], "hidden_size"),
                heads=_positive(bb["num_heads"], "num_heads"),
                kv_heads=_positive(bb["num_kv_heads"], "num_kv_heads"),
                head_dim=_positive(bb["head_dim"], "head_dim"),
                intermediate=_positive(bb["intermediate_size"], "intermediate_size"),
                experts=moe.get("n_routed", 0), top_k=moe.get("top_k", 0),
                window=(template.get("cache") or {}).get("window"),
                weights=weights, operators=operators, activation=bb.get("activation", "silu"),
            )
            if layer.ffn == "MOE" and not (layer.experts > 0 and 0 < layer.top_k <= layer.experts):
                raise PlanningError("SOURCE_MISMATCH", "Invalid expert count/top-k", scope)
            layers.append(layer)
    if len(layers) != model["layers"]:
        raise PlanningError("SOURCE_MISMATCH", "Layer count differs from frozen model")
    shared = tuple(weight_from_module(row, snapshot["id"] + "/shared") for row in analysis["shared_modules"])
    identities = {w.storage_id for w in shared}
    if any(w.alias_of not in identities for w in shared if w.alias_of):
        raise PlanningError("SOURCE_MISMATCH", "Unknown shared storage alias")
    logical = sum(w.numel for w in shared if not w.alias_of)
    logical += sum(w.numel for layer in layers for w in layer.weights if not w.alias_of)
    if logical != analysis["totals"]["total"]:
        raise PlanningError("SOURCE_MISMATCH", "Normalized parameter ledger does not conserve source total")
    return ModelIR(
        model_id=model["id"], structure_id=snapshot["id"], revision=model["revision"],
        config_sha256=model["config_sha256"], structure_engine=snapshot["engine_version"],
        structure_formula=snapshot["formula_version"], layers=tuple(layers), shared=shared,
        vocab=model["vocab_size"], hidden=model["hidden_size"],
        context_limit=model["max_position_embeddings"], logical_numel=logical,
    )
