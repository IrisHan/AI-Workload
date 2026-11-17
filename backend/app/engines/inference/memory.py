"""Rank-local admission bound and event-ledger memory composition."""
from dataclasses import asdict
from math import ceil

from .cache import cache_allocation, request_quotas
from .lifetimes import activation_program, peak_ledger
from .partition import rank_weights, ranks

COMPONENTS = ("weight", "kv", "activation", "communication", "runtime", "fragmentation", "safety")


def effective(assumptions):
    return {key: value["effective"] for key, value in assumptions.items()}


def compose(weight, kv, activation, communication, capacity, assumptions):
    """All seven quantities refer to the SAME instant or labeled admission bound."""
    a = effective(assumptions)
    kv = ceil(kv / a["kv_efficiency"])
    activation = ceil(activation / a["activation_efficiency"])
    comm = max(communication, a["comm_reserve_bytes"])
    fragmentation = ceil((kv + activation + comm) * (1 / a["memory_efficiency"] - 1))
    components = dict(zip(COMPONENTS, (
        weight, kv, activation, comm, int(a["runtime_fixed_bytes"]), fragmentation,
        ceil(capacity * a["safety_fraction"]),
    )))
    peak = sum(components.values())
    return {"components": components, "peak_bytes": peak, "capacity_bytes": capacity,
            "free_bytes": capacity - peak, "capacity_pass": peak <= capacity,
            "deficit_bytes": max(0, peak - capacity)}


def layer_activation(layer, tokens, candidate, element_bytes, total_ep_tokens=None):
    # Capacity bound allows all legal top-k choices to concentrate on this owner.
    expert_tokens = (total_ep_tokens * min(layer.top_k, layer.experts // candidate.ep)
                     if layer.experts and total_ep_tokens is not None else None)
    program = activation_program(layer, tokens, candidate.tp, element_bytes, expert_tokens=expert_tokens,
                                 expert_tp=candidate.expert_tp if layer.experts else candidate.tp)
    return peak_ledger({}, program)


def admission_memory(ir, candidate, design, hardware, assumptions, *, details=True):
    quotas = request_quotas(design.load.concurrency, candidate)
    elem = 4 if design.runtime.activation_dtype == "fp32" else 2
    capacity = hardware["memory_bytes"]
    result = []
    for rank in ranks(candidate, design.resources.gpus_per_node):
        quota = quotas[rank.d][rank.e]
        layers = [layer for layer in ir.layers if candidate.stage(layer.index) == rank.p]
        tokens = min(quota, design.runtime.max_batch_sequences) * min(
            design.load.prompt_tokens, design.runtime.prefill_chunk_tokens)
        tokens = min(tokens, design.runtime.max_batched_tokens)
        ep_tokens = sum(min(q, design.runtime.max_batch_sequences) for q in quotas[rank.d])
        ep_tokens *= min(design.load.prompt_tokens, design.runtime.prefill_chunk_tokens)
        ep_tokens = min(ep_tokens, design.runtime.max_batched_tokens * candidate.attention_groups)
        shards = rank_weights(ir, candidate, rank, design.runtime.weight_dtype)
        weight = sum(shard.bytes for shard in shards)
        context = design.load.prompt_tokens + design.load.output_tokens
        rows = []
        for layer in layers:
            cache = cache_allocation(layer, [context] * quota, candidate.tp, design.runtime.kv_dtype)
            activation = layer_activation(layer, tokens, candidate, elem, ep_tokens)
            rows.append({"layer_index": layer.index, "instance_id": layer.instance_id,
                         "cache": cache, "activation_peak_bytes": activation["peak_bytes"],
                         "activation_peak_at_step": activation["peak_at"],
                         "activation_timeline": activation["timeline"] if details else []})
        kv = sum(row["cache"]["total_bytes"] for row in rows)
        activation = max((row["activation_peak_bytes"] for row in rows), default=0)
        # Microbatches may queue at stage boundaries; bounded by admitted requests.
        pipeline_buffers = (2 * quota * min(design.load.prompt_tokens, design.runtime.prefill_chunk_tokens)
                            * ir.hidden * elem if candidate.pp > 1 else 0)
        head = tokens * ir.hidden * elem
        if rank.p == candidate.pp - 1:
            last_tokens = min(quota, design.runtime.max_batch_sequences)
            head += last_tokens * (ir.vocab // candidate.tp) * 4  # FP32 sampling logits
        activation = max(activation, head) + pipeline_buffers
        payload = tokens * ir.hidden * elem
        # Separate send/receive pools; routed traffic cannot exceed legal top-k concentration.
        routed_payload = max((ep_tokens * min(layer.top_k, layer.experts // candidate.ep)
                              * (ir.hidden * elem + 12) for layer in layers if layer.experts), default=0)
        communication = 2 * payload if candidate.tp > 1 or candidate.pp > 1 else 0
        if candidate.ep > 1:
            communication += 2 * routed_payload
        ledger = compose(weight, kv, activation, communication, capacity, assumptions)
        result.append({**asdict(rank), **ledger, "kind": "reserved_admission_bound",
                       "resident_request_quota": quota, "reserved_context": context,
                       "pipeline_buffer_bytes": pipeline_buffers,
                       "raw_reserves": {"kv": kv, "activation": activation, "communication": communication,
                                        "local_tokens": tokens, "head_activation": head,
                                        "routed_payload": routed_payload},
                       "weights": [asdict(s) for s in shards] if details else [], "layers": rows})
    worst = max(result, key=lambda row: (row["peak_bytes"], -row["rank"]))
    return {"capacity_pass": all(row["capacity_pass"] for row in result),
            "worst_rank": worst["rank"], "peak_bytes": worst["peak_bytes"],
            "capacity_bytes": capacity, "components": worst["components"],
            "free_bytes": worst["free_bytes"], "ranks": result,
            "kind": "reserved_admission_bound",
            "notes": ["Capacity uses simultaneous conservative admission bounds, not low-load event peaks.",
                      "Page metadata is 8 bytes per layer/request/page; FlashAttention tile is 128.",
                      "Expert activation reserve permits all legal routing concentration without token drops."]}
