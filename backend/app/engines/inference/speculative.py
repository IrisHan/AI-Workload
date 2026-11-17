"""Fixed-context speculative verification cycle, separate from request simulation."""
from .cache import cache_allocation, request_quotas
from .compute import stage_program
from .ir import PlanningError
from .memory import compose, layer_activation
from .partition import rank_weights, ranks


def speculative_cycle(sources, candidate, design):
    spec, runtime, ir = design.runtime.speculative, design.runtime, sources.ir
    if not spec.enabled:
        return None
    q = spec.draft_tokens + 1
    context = design.load.prompt_tokens
    if context + q > ir.context_limit:
        raise PlanningError("SPEC_CONTEXT_LIMIT", "Context plus verify positions exceeds model context")
    accepted = sum(spec.acceptance_rate ** i for i in range(1, q))
    emitted = 1 + accepted
    quotas = request_quotas(design.load.concurrency, candidate)
    chunk = min(context, runtime.prefill_chunk_tokens) if spec.include_prefill else 0
    batches = []
    for replica in quotas:
        groups = []
        for quota in replica:
            count = min(quota, runtime.max_batch_sequences)
            if count * q + chunk > runtime.max_batched_tokens:
                raise PlanningError("SPEC_TOKEN_BUDGET", "Verify batch plus prefill exceeds token budget",
                                    "runtime.max_batched_tokens")
            groups.append([(context, q, q)] * count + ([(0, chunk, int(chunk == context))] if chunk else []))
        batches.append(groups)
    programs = []
    for d, groups in enumerate(batches):
        for p in range(candidate.pp):
            programs.append({"replica": d, "stage": p,
                             **stage_program(sources, candidate, design, d, p, groups, details=True)})
    # All communication is serialized, compute across independent replicas may overlap.
    compute_by_replica = []
    communication = [p["communication_seconds"] for p in programs]
    for d in range(candidate.independent_replicas):
        values = [s["seconds"] for p in programs if p["replica"] == d
                  for s in p["segments"] if s["kind"] == "compute"]
        compute_by_replica.append(None if None in values else sum(values))
    target = (None if None in compute_by_replica or None in communication else
              max(compute_by_replica, default=0) + sum(communication))
    total = None if target is None or spec.draft_latency_ms is None else target + spec.draft_latency_ms / 1000
    elem = 4 if runtime.activation_dtype == "fp32" else 2
    memory = []
    for rank in ranks(candidate, design.resources.gpus_per_node):
        groups = batches[rank.d]
        local = groups[rank.e]
        tokens = sum(row[1] for row in local)
        all_tokens = sum(sum(row[1] for row in group) for group in groups)
        layers = [layer for layer in ir.layers if candidate.stage(layer.index) == rank.p]
        weight = sum(w.bytes for w in rank_weights(ir, candidate, rank, runtime.weight_dtype))
        kv = sum(cache_allocation(layer, [k + query for k, query, _ in local], candidate.tp,
                                  runtime.kv_dtype)["total_bytes"] for layer in layers)
        activation = max((layer_activation(layer, tokens, candidate, elem, all_tokens)["peak_bytes"]
                          for layer in layers), default=0)
        head = tokens * ir.hidden * elem
        if rank.p == candidate.pp - 1:
            head += sum(row[2] for row in local) * (ir.vocab // candidate.tp) * 4
        activation = max(activation, head)
        if candidate.pp > 1:
            activation += 2 * tokens * ir.hidden * elem
        comm = 2 * tokens * ir.hidden * elem if candidate.tp > 1 or candidate.pp > 1 else 0
        if candidate.ep > 1:
            comm += 2 * max((all_tokens * min(layer.top_k, layer.experts // candidate.ep)
                             * (ir.hidden * elem + 12) for layer in layers if layer.experts), default=0)
        ledger = compose(weight, kv, activation, comm, sources.hardware["memory_bytes"], sources.assumptions)
        draft = spec.draft_memory_bytes
        peak = ledger["peak_bytes"] + (draft or 0)
        memory.append({"rank": rank.rank, "target_components": ledger["components"],
                       "target_peak_bytes": ledger["peak_bytes"], "draft_bytes": draft,
                       "peak_bytes": peak if draft is not None else None,
                       "capacity_bytes": ledger["capacity_bytes"],
                       "capacity_status": "unknown" if draft is None else
                       "pass" if peak <= ledger["capacity_bytes"] else "fail"})
    decode_count = sum(min(quota, runtime.max_batch_sequences) for replica in quotas for quota in replica)
    outputs = decode_count * emitted
    flops = sum(p["flops"] for p in programs)
    peak = sources.hardware.get("compute", {}).get(runtime.activation_dtype, {}).get("dense")
    missing = []
    if spec.draft_latency_ms is None:
        missing.append("Draft latency is unknown")
    if spec.draft_memory_bytes is None:
        missing.append("Draft memory is unknown")
    if target is None:
        missing.append("Target timing is unknown")
    capacity = "fail" if any(r["capacity_status"] == "fail" for r in memory) else (
        "unknown" if spec.draft_memory_bytes is None else "pass")
    return {
        "kind": "modeled", "scope": "fixed_context_mixed_cycle" if chunk else "fixed_context_decode_cycle",
        "status": "capacity_fail" if capacity == "fail" else "unknown" if missing else "modeled",
        "baseline_scope": "Non-speculative Baseline",
        "draft_tokens": spec.draft_tokens, "acceptance_rate": spec.acceptance_rate,
        "expected_accepted_tokens": accepted, "expected_emitted_tokens": emitted,
        "verify_positions_per_request": q, "decode_requests": decode_count,
        "context_tokens": context, "prefill_tokens": chunk * sum(len(g) for g in batches),
        "extra_prefill_request_per_group": bool(chunk),
        "target_seconds": target, "draft_seconds": None if spec.draft_latency_ms is None else spec.draft_latency_ms / 1000,
        "cycle_seconds": total, "output_tokens": outputs,
        "output_tokens_s": outputs / total if total and capacity != "fail" else None,
        "tokens_s_per_gpu": outputs / total / candidate.gpu_count if total and capacity != "fail" else None,
        "tpot_ms": total * 1000 / emitted if total and capacity != "fail" else None,
        "mfu": flops / (peak * 1e12 * candidate.gpu_count * total) if peak and total else None,
        "flops": flops, "hbm_bytes": sum(p["hbm_bytes"] for p in programs),
        "capacity_status": capacity, "memory": memory, "programs": programs,
        "unknown": missing,
        "notes": ["Expected output includes one bonus/replacement token.",
                  "Draft latency is per parallel cycle; draft memory is per GPU.",
                  "Zero draft cost is an explicit optimistic assumption.",
                  "Fixed context only; no end-to-end queue, acceptance tree or rollback prediction.",
                  "Mixed prefill uses one additional request slot per attention group."],
    }
