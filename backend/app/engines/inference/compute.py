"""Partitioned operator costs from source tensor shapes and actual microbatches."""
from math import prod

from .communication import collective, expert_exchange, framework_exchange, routing_matrix, transfer_round
from .partition import ranks, shard_weight
from .roofline import operator_time


def attention_pairs(cached_tokens, query_tokens, window=None):
    """Sum min(cached+j,window), without expanding long token sequences."""
    if window is None:
        return query_tokens * cached_tokens + query_tokens * (query_tokens + 1) // 2
    growing = min(query_tokens, max(0, window - cached_tokens))
    return growing * cached_tokens + growing * (growing + 1) // 2 + (query_tokens - growing) * window


def _sum_known(values):
    return None if any(value is None for value in values) else sum(values)


def operator_row(name, layer, rank, flops, traffic, sources, design, *, scalar=False, dimensions=None):
    timing = operator_time(flops, traffic, sources.hardware, sources.assumptions,
                           design.runtime.activation_dtype, scalar=scalar)
    return {"id": name, "layer_index": layer, "rank": rank, "flops": flops if not scalar else 0,
            "scalar_equivalent_ops": flops if scalar else 0, "hbm_bytes": traffic,
            "dimensions": dimensions or {}, "formula_id": "scalar-v1" if scalar else "2TIO-v1",
            **timing}


def stage_program(sources, candidate, design, d, p, batches, *, details=False, layer_index=None):
    """Batches[e] = [(cached, query_tokens, produces_logits), ...].

    One synchronous EP wave. Every t lane uses identical expert token routing;
    TP-sharded matrix work differs from replicated dispatch bytes.
    """
    ir = sources.ir
    placements = ranks(candidate, design.resources.gpus_per_node)
    selected = [r for r in placements if r.d == d and r.p == p]
    elem = 4 if design.runtime.activation_dtype == "fp32" else 2
    tokens = [sum(q for _, q, _ in batch) for batch in batches]
    logits = [sum(int(last) for _, _, last in batch) for batch in batches]
    rows, communication = [], []
    stages, segments = [], []
    total_flops = total_scalar = total_hbm = 0

    def add(name, layer, rank, flops, traffic, scalar=False, dimensions=None):
        nonlocal total_flops, total_scalar, total_hbm
        row = operator_row(name, layer, rank.rank, flops, traffic, sources, design,
                           scalar=scalar, dimensions=dimensions)
        rows.append(row)
        total_flops += row["flops"]
        total_scalar += row["scalar_equivalent_ops"]
        total_hbm += traffic
        return row["seconds"]

    def reduce_group(group, payload, kind="all_reduce"):
        event = collective(group, payload, placements, sources.topology, kind)
        communication.append(event)
        return event["seconds"]

    def endpoints(event):
        return sorted({r for transfer in event["transfers"] for r in (transfer["src"], transfer["dst"])})

    def record(seconds, kind="compute", layer=-1, operators=(), category="tp", participants=None):
        components = dict.fromkeys(("compute", "memory", "tp", "pp", "ep", "runtime"), 0.0)
        activity = {}
        if kind == "compute":
            vectors = {}
            for row in operators:
                if row["seconds"] is None:
                    continue
                activity[row["rank"]] = activity.get(row["rank"], 0) + row["seconds"]
                vector = vectors.setdefault(row["rank"], dict.fromkeys(components, 0.0))
                multiplier = row.get("duration_multiplier", 1)
                branch = "compute" if row["bottleneck"] == "compute" else "memory"
                vector[branch] += row["compute_seconds" if branch == "compute" else "hbm_seconds"] * multiplier
                vector["runtime"] += row["launch_seconds"] * multiplier
            if activity:
                components = vectors[max(activity, key=activity.get)]
        else:
            components[category] = seconds
            activity = {r: seconds for r in (participants if participants is not None else
                        communication[-1].get("ranks", [r.rank for r in selected]))}
        stages.append(seconds)
        segments.append({"kind": kind, "layer_index": layer, "seconds": seconds,
                         "components": components, "rank_activity_seconds": activity,
                         "model_flops": sum(row["flops"] for row in operators),
                         "busy_gpu_seconds": _sum_known([row["seconds"] for row in operators])})

    # Embedding lookup uses only touched rows. Vocab-partitioned result is reduced.
    if p == 0 and layer_index is None:
        times = []
        for rank in selected:
            t = tokens[rank.e]
            if t:
                times.append(add("embedding", -1, rank, 0, t * ir.hidden * elem * 2, True))
        record(None if None in times else max(times, default=0), operators=rows[:])
        for e, t in enumerate(tokens):
            group = [r.rank for r in selected if r.e == e]
            record(reduce_group(group, t * ir.hidden * elem), "communication")
    for layer in ir.layers:
        if candidate.stage(layer.index) != p or (layer_index is not None and layer.index != layer_index):
            continue
        matrix = routing_matrix(tokens, layer.experts, layer.top_k) if layer.experts else None
        per_expert = [sum(row[x] for row in matrix) for x in range(layer.experts)] if matrix else []
        # Device pressure factor is a scenario duration multiplier; FLOPs remain conserved.
        imbalance = sources.assumptions["routing_imbalance"]["effective"] if layer.experts else 1
        layer_start = len(rows)
        for rank in selected:
            t = tokens[rank.e]
            owner = candidate.expert_owner(rank)
            times = []
            for weight in layer.weights:
                shard = shard_weight(weight, candidate, owner, design.runtime.weight_dtype,
                                     kv_heads=layer.kv_heads)
                if weight.routed:
                    allocation = per_expert[shard.expert_start:shard.expert_stop]
                    n = sum(allocation)
                    active = sum(value > 0 for value in allocation)
                    multiplier = imbalance
                else:
                    n, active, multiplier = t, weight.count if t else 0, 1
                if not n:
                    continue
                if len(shard.local_shape) == 2:
                    output, width = shard.local_shape
                    executions = n if weight.routed else n * weight.count
                    traffic = (prod(shard.local_shape) + shard.bias) * active
                    traffic *= 4 if design.runtime.weight_dtype == "fp32" or weight.dtype == "fp32" else 2
                    traffic += executions * (width + output) * elem
                    seconds = add(weight.module_id, layer.index, rank, 2 * executions * width * output,
                                  traffic, dimensions={"tokens": executions, "in": width, "out": output})
                    rows[-1]["seconds"] = None if seconds is None else seconds * multiplier
                    rows[-1]["duration_multiplier"] = multiplier
                    rows[-1]["traffic_inputs"] = {
                        "local_weight_shape": list(shard.local_shape), "bias": shard.bias,
                        "active_weight_copies": active, "executions": executions,
                        "weight_element_bytes": 4 if design.runtime.weight_dtype == "fp32" or weight.dtype == "fp32" else 2,
                        "activation_element_bytes": elem}
                    rows[-1]["traffic_formula"] = (
                        "(prod(local_weight_shape)+bias) × active_weight_copies × weight_element_bytes"
                        " + executions × (K+N) × activation_element_bytes")
                    times.append(None if seconds is None else seconds * multiplier)
                    if shard.bias:
                        times.append(add(weight.module_id + ".bias", layer.index, rank,
                                         executions * shard.bias, executions * output * elem * 2, True))
                else:
                    width = prod(shard.local_shape)
                    # q/k norm acts once per local head; rank-1 source is per-head dimension.
                    repetitions = layer.heads // candidate.tp if weight.module_id == "q_norm" else (
                        max(1, layer.kv_heads // candidate.tp) if weight.module_id == "k_norm" else 1)
                    times.append(add(weight.module_id, layer.index, rank, n * repetitions * (5 * width + 16),
                                     n * repetitions * width * elem * 2 + shard.bytes, True))
            if t:
                pairs = sum(attention_pairs(k, q, layer.window) for k, q, _ in batches[rank.e])
                heads = layer.heads // candidate.tp
                kv_width = max(1, layer.kv_heads // candidate.tp) * layer.head_dim
                # Conservative streaming traffic: reread retained K/V for each query.
                traffic = pairs * 2 * kv_width * (4 if design.runtime.kv_dtype == "fp32" else 2)
                traffic += t * (2 * heads * layer.head_dim + 2 * kv_width) * elem
                times.append(add("attention", layer.index, rank, 4 * heads * layer.head_dim * pairs,
                                 traffic, dimensions={"pairs": pairs, "heads": heads, "dim": layer.head_dim}))
                rows[-1]["traffic_inputs"] = {
                    "causal_pairs": pairs, "local_kv_width": kv_width, "local_heads": heads,
                    "head_dim": layer.head_dim, "local_tokens": t, "activation_element_bytes": elem,
                    "kv_element_bytes": 4 if design.runtime.kv_dtype == "fp32" else 2}
                rows[-1]["traffic_formula"] = (
                    "causal_pairs × 2 × local_kv_width × kv_element_bytes + local_tokens"
                    " × (2 × local_heads × head_dim + 2 × local_kv_width) × activation_element_bytes")
                times.append(add("softmax", layer.index, rank, 20 * heads * pairs,
                                 2 * t * heads * layer.head_dim * elem, True))
                times.append(add("rope", layer.index, rank,
                                 6 * t * (heads * layer.head_dim + kv_width),
                                 2 * t * (heads * layer.head_dim + kv_width) * elem, True))
                times.append(add("residual_add", layer.index, rank, 2 * t * layer.hidden,
                                 6 * t * layer.hidden * elem, True))
            # Nonlinear expert work uses routed assignments, not local attention tokens.
            nonlinear = 0
            for weight in layer.weights:
                if weight.module_id.endswith("up_proj"):
                    n = (sum(per_expert[owner * (layer.experts // candidate.ep):
                                        (owner + 1) * (layer.experts // candidate.ep)])
                         if weight.routed else t * weight.count)
                    nonlinear += n * weight.shape[0] // (candidate.expert_tp if weight.routed else candidate.tp)
            if nonlinear:
                times.append(add("ffn_activation", layer.index, rank,
                                 nonlinear * (24 if layer.activation == "gelu" else 20),
                                 nonlinear * elem * 3, True))
            if layer.experts and t:
                times.append(add("router_topk", layer.index, rank, t * layer.experts * (20 + layer.top_k),
                                 t * layer.experts * 8 + t * layer.top_k * 12, True))
        phase_rows = {"attention": {}, "router": {}, "ffn": {}}
        for row in rows[layer_start:]:
            name = row["id"]
            phase = ("router" if name in {"router", "router_correction", "router_topk", "ffn_norm"}
                     else "ffn" if any(key in name for key in
                                      ("expert", "gate_proj", "up_proj", "down_proj", "ffn_activation"))
                     else "attention")
            phase_rows[phase].setdefault(row["rank"], []).append(row)

        def compute_phase(phase, phase_costs=phase_rows, index=layer.index):
            totals = [_sum_known([row["seconds"] for row in values]) for values in phase_costs[phase].values()]
            operators = [row for values in phase_costs[phase].values() for row in values]
            record(None if None in totals else max(totals, default=0), layer=index, operators=operators)

        compute_phase("attention")
        for e, t in enumerate(tokens):
            group = [r.rank for r in selected if r.e == e]
            record(reduce_group(group, t * layer.hidden * elem), "communication", layer.index)
        compute_phase("router")
        exchanges = []
        if matrix and candidate.legacy:
            for lane in range(candidate.tp):
                group = [r.rank for r in selected if r.t == lane]
                event = expert_exchange(matrix, group, layer.hidden, elem, placements, sources.topology)
                communication.append({"kind": "expert_dispatch_combine", "layer_index": layer.index,
                                      "matrix": matrix, **event})
                exchanges.append(event)
                record(event["dispatch"]["seconds"], "communication", layer.index, category="ep",
                       participants=endpoints(event["dispatch"]))
        elif matrix:
            event = framework_exchange(matrix, candidate, selected, layer.hidden, elem,
                                       placements, sources.topology)
            communication.append({"kind": "expert_dispatch_combine", "layer_index": layer.index,
                                  "matrix": matrix, **event})
            exchanges.append(event)
            record(event["dispatch"]["seconds"], "communication", layer.index, category="ep",
                   participants=endpoints(event["dispatch"]))
        compute_phase("ffn")
        if layer.experts:
            for owner in range(candidate.ep):
                group = [r.rank for r in selected if candidate.expert_owner(r) == owner]
                width = layer.experts // candidate.ep
                owner_tokens = sum(per_expert[owner * width:(owner + 1) * width])
                record(reduce_group(group, owner_tokens * layer.hidden * elem), "communication", layer.index)
        for e, t in enumerate(tokens):
            group = [r.rank for r in selected if r.e == e]
            if not layer.experts or any(w.module_id.startswith("shared_expert.") for w in layer.weights):
                record(reduce_group(group, t * layer.hidden * elem), "communication", layer.index)
        for event in exchanges:
            record(event["combine"]["seconds"], "communication", layer.index, category="ep",
                   participants=endpoints(event["combine"]))
            if "output" in event:
                record(event["output"]["seconds"], "communication", layer.index, category="ep",
                       participants=endpoints(event["output"]))
    if p == candidate.pp - 1 and layer_index is None:
        times = []
        for rank in selected:
            n = logits[rank.e]
            if n:
                local_vocab = ir.vocab // candidate.tp
                times.append(add("final_norm", len(ir.layers), rank, n * (5 * ir.hidden + 16),
                                 2 * n * ir.hidden * elem, True))
                times.append(add("lm_head", len(ir.layers), rank, 2 * n * ir.hidden * local_vocab,
                                 (ir.hidden * local_vocab + n * (ir.hidden + local_vocab)) * elem))
                times.append(add("sampling", len(ir.layers), rank, n * local_vocab * 20,
                                 n * local_vocab * 8, True))
        # Each rank runs its final operators sequentially.
        by_rank = {}
        for row in rows:
            if row["layer_index"] == len(ir.layers):
                by_rank.setdefault(row["rank"], []).append(row["seconds"])
        totals = [_sum_known(values) for values in by_rank.values()]
        record(None if None in totals else max(totals, default=0), layer=len(ir.layers),
               operators=[row for row in rows if row["layer_index"] == len(ir.layers)])
        for e, n in enumerate(logits):
            record(reduce_group([r.rank for r in selected if r.e == e], n * 16),
                   "communication", len(ir.layers))
    elif any(tokens) and layer_index is None:
        transfers = [(r.rank, r.rank + candidate.attention_groups * candidate.tp, tokens[r.e] * ir.hidden * elem)
                     for r in selected]
        event = transfer_round(transfers, placements, sources.topology)
        communication.append({"kind": "pipeline_send", **event})
        record(event["seconds"], "communication", len(ir.layers), category="pp",
               participants=endpoints(event))
    unknown = sorted({row["unknown"] for row in rows if row["unknown"]})
    if any(event["seconds"] is None for event in communication):
        unknown.append("Required topology bandwidth is unknown")
    return {"seconds": _sum_known(stages), "flops": total_flops, "scalar_equivalent_ops": total_scalar,
            "hbm_bytes": total_hbm, "communication_seconds": _sum_known([e["seconds"] for e in communication]),
            "unknown": unknown, "segments": segments, "operators": rows if details else [],
            "communication": communication if details else [], "kind": "modeled"}
