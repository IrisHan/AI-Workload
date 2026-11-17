"""Calculation evidence from frozen inputs and engine outputs, never display-value arithmetic."""
from math import prod

from .metrics import UNITS

VERSION = "calculation-trace-1"
LABELS = {"compute": "Compute", "memory": "Memory", "tp": "TP Communication",
          "pp": "PP Communication", "ep": "EP Communication", "runtime": "Runtime / Launch",
          "scheduler": "Scheduler"}


def fact(key, value, unit="", source="Derived", label=None, source_ref=None, formula=None):
    return {"key": key, "label": label or key.replace("_", " ").title(), "value": value,
            "unit": unit, "source": source, "source_ref": source_ref, "formula": formula}


def scope_for(sources, candidate, **extra):
    return {"model_revision": sources.ir.revision, "config_sha256": sources.ir.config_sha256,
            "structure_id": sources.ir.structure_id, "hardware_sha256": sources.hardware.get("content_sha256"),
            "candidate_key": candidate.key, "tp": candidate.tp, "pp": candidate.pp,
            "ep": candidate.ep, "dp": candidate.dp, **extra}


def assumptions(sources):
    return [fact(k, v["effective"], "bytes" if k.endswith("_bytes") else "µs" if k.endswith("_us")
                 else "GB/s" if k.endswith("_gb_s") else "", "Override" if v.get("user_override") is not None
                 else "Assumption", source_ref=f"frozen.sources.assumptions.{k}",
                 formula=v.get("reason")) for k, v in sources.assumptions.items()]


def trace(key, title, value, unit, scope, inputs, formula, *, steps=None, notes=None, assumed=None, reason=None):
    return {"id": key, "title": title, "status": "unavailable" if value is None or reason else "available",
            "scope": scope, "inputs": inputs, "steps": steps or [], "formula": formula,
            "assumptions": assumed or [], "result": fact(key, value if not reason else None, unit),
            "sources": [{"path": v["source_ref"], "type": v["source"]} for v in inputs + (assumed or [])
                        if v.get("source_ref")], "notes": notes or [], "unavailable_reason": reason}


def memory_explanation(sources, candidate, design, row):
    scope = scope_for(sources, candidate, rank=row["rank"], kind="reserved_admission_bound")
    a = assumptions(sources)
    c, raw = row["components"], row["raw_reserves"]
    weights = row["weights"]
    weight_inputs = [fact(s["storage_id"], s["bytes"], "bytes", label=s["module_id"],
                          source_ref=f"IR storage {s['storage_id']} ← frozen.sources.structure.analysis",
                          formula=f"(prod({s['local_shape']}) + {s['bias']}) × {s['count']} × dtype bytes; "
                          f"{s['partition']}; experts [{s['expert_start']}, {s['expert_stop']})") for s in weights]
    weight_inputs.insert(0, fact("runtime_dtype", design.runtime.weight_dtype, source="Workload",
                                source_ref="frozen.design.runtime.weight_dtype"))
    cache = {key: sum(layer["cache"][key] for layer in row["layers"])
             for key in ("payload_bytes", "metadata_bytes", "tail_waste_bytes")}
    kv_inputs = [fact("prompt", design.load.prompt_tokens, "tokens", "Workload"),
                 fact("output", design.load.output_tokens, "tokens", "Workload"),
                 fact("quota", row["resident_request_quota"], "requests"),
                 fact("tp", candidate.tp), fact("dtype", design.runtime.kv_dtype, source="Workload")]
    kv_inputs += [fact("resident_layers", len(row["layers"])), fact("context", row["reserved_context"], "tokens"),
                  fact("reserved_tokens", row["reserved_context"] * row["resident_request_quota"], "tokens")]
    for layer_row in row["layers"]:
        layer = sources.ir.layers[layer_row["layer_index"]]
        kv_inputs.append(fact(f"layer_{layer.index}", layer_row["cache"]["total_bytes"], "bytes",
                              source_ref=f"IR layer {layer.index} ← frozen.sources.structure.analysis",
                              formula=f"ceil(retained/{design.runtime.kv_page_tokens}) × "
                              f"({design.runtime.kv_page_tokens} × 2 × {max(1, layer.kv_heads // candidate.tp)}"
                              f" × {layer.head_dim} × {4 if design.runtime.kv_dtype == 'fp32' else 2} + 8)"
                              f" × {row['resident_request_quota']}; retained=min(context, {layer.window})"
                              if layer.window else
                              f"ceil({row['reserved_context']}/16) × (16 × 2 × "
                              f"{max(1, layer.kv_heads // candidate.tp)} × {layer.head_dim} × "
                              f"{4 if design.runtime.kv_dtype == 'fp32' else 2} + 8) × {row['resident_request_quota']}"))
    base = [fact(key, value, "bytes") for key, value in c.items()]
    traces = {
        "weight": trace("weight", "Weights / GPU", c["weight"], "bytes", scope, weight_inputs,
                        "Σ unique rank-local storage: (prod(local shape) + bias) × count × dtype bytes",
                        notes=["PP/EP ownership first; TP/ETP shards matrix axes. Norm/router may replicate; aliases counted once."]),
        "kv": trace("kv", "KV Cache / GPU", c["kv"], "bytes", scope, kv_inputs,
                    "ceil((Σ paged payload + Σ page metadata) / kv_efficiency)",
                    steps=[{"label": k, "value": v, "unit": "bytes", "formula": "Σ local layers"}
                           for k, v in cache.items()], assumed=[v for v in a if v["key"] == "kv_efficiency"],
                    notes=["Tail waste is already in payload; metadata adds 8 bytes per layer/request/page.",
                           "KV heads replicate when TP exceeds KV heads. Windowed layers retain min(context, window)."]),
        "activation": trace("activation", "Peak Activation / GPU", c["activation"], "bytes", scope,
                            [fact("layer_peak", max((r["activation_peak_bytes"] for r in row["layers"]), default=0), "bytes"),
                             fact("head", raw["head_activation"], "bytes"),
                             fact("pipeline_buffers", row["pipeline_buffer_bytes"], "bytes")],
                            "ceil((max(layer simultaneous peaks, head) + pipeline buffers) / activation_efficiency)",
                            assumed=[v for v in a if v["key"] == "activation_efficiency"]),
        "communication": trace("communication", "Shared Communication", c["communication"], "bytes", scope,
                               [fact("message_pools", raw["communication"], "bytes"),
                                fact("local_tokens", raw["local_tokens"]), fact("hidden", sources.ir.hidden, source="Model"),
                                fact("routed_payload", raw["routed_payload"], "bytes")],
                               "max(2 × local_tokens × hidden × activation bytes if TP/PP > 1, 0)"
                               " + 2 × routed payload if EP > 1; then max(message pools, comm_reserve_bytes)",
                               assumed=[v for v in a if v["key"] == "comm_reserve_bytes"]),
        "runtime": trace("runtime", "Shared Runtime", c["runtime"], "bytes", scope, [],
                         "runtime_fixed_bytes", assumed=[v for v in a if v["key"] == "runtime_fixed_bytes"]),
        "fragmentation": trace("fragmentation", "Fragmentation", c["fragmentation"], "bytes", scope,
                               [v for v in base if v["key"] in ("kv", "activation", "communication")],
                               "ceil((KV + Activation + Communication) × (1 / memory_efficiency − 1))",
                               assumed=[v for v in a if v["key"] == "memory_efficiency"]),
        "safety": trace("safety", "Safety Reserve", c["safety"], "bytes", scope,
                        [fact("capacity", row["capacity_bytes"], "bytes", "Hardware",
                              source_ref="frozen.sources.hardware.memory_bytes")], "ceil(capacity × safety_fraction)",
                        assumed=[v for v in a if v["key"] == "safety_fraction"]),
        "peak": trace("peak", "Admission Peak", row["peak_bytes"], "bytes", scope, base, "Σ seven admission components"),
        "free": trace("free", "Free Memory", row["free_bytes"], "bytes", scope,
                      [fact("capacity", row["capacity_bytes"], "bytes", "Hardware"),
                       fact("peak", row["peak_bytes"], "bytes")], "capacity − peak; safety is included in peak once"),
    }
    fraction = row["peak_bytes"] / row["capacity_bytes"]
    traces["utilization"] = trace("utilization", "Memory Utilization", fraction, "fraction", scope,
                                 [fact("peak", row["peak_bytes"], "bytes"),
                                  fact("capacity", row["capacity_bytes"], "bytes", "Hardware")], "peak / capacity")
    return {"version": VERSION, "scope": scope, "traces": traces, "utilization": fraction,
            "capacity_gb": row["capacity_bytes"] / 1e9, "capacity_pass": row["capacity_pass"]}


def summary_explanation(sources, candidate, design, plan):
    scope = scope_for(sources, candidate, kind="primary_simulation_measurement_window")
    sim, m = plan["performance"], plan["performance"]["metrics"]
    e = sim["explanation"]
    a = assumptions(sources)
    context = [fact("model", sources.model["name"], source="Model", source_ref=sources.model.get("source_url")),
               fact("hardware", sources.hardware["name"], source="Hardware",
                    source_ref=sources.hardware.get("source_url")),
               fact("gpu_count", candidate.gpu_count), fact("mode", design.mode, source="Workload")]
    context += [fact(k, v, source="Workload", source_ref=f"frozen.design.load.{k}")
                for k, v in design.load.model_dump().items()]
    context += [fact(k, getattr(candidate, k)) for k in ("tp", "pp", "ep", "dp", "expert_tp")]
    context += [fact(k, getattr(design.runtime, k), source="Workload", source_ref=f"frozen.design.runtime.{k}")
                for k in ("max_batch_sequences", "prefill_chunk_tokens", "kv_page_tokens", "weight_dtype")]
    all_weights = {w.storage_id: w for w in (*sources.ir.shared, *(w for layer in sources.ir.layers for w in layer.weights))
                   if not w.alias_of}
    params = [fact(w.storage_id, w.numel, "parameters", "Model", label=f"{w.module_id} · {w.dtype}",
                   source_ref=f"IR storage {w.storage_id} ← frozen.sources.structure.analysis",
                   formula=f"(prod({list(w.shape)}) + {w.bias}) × {w.count}") for w in all_weights.values()]
    scope_model = scope_for(sources, candidate, kind="source_model")
    traces = {"parameters": trace("parameters", "Total Parameters", sources.ir.logical_numel, "parameters",
                                  scope_model, params, "Σ unique source storage numel"),
              "source_weight": trace("source_weight", "Source Weight", sources.structure["analysis"]["serving"]["weight_bytes"],
                                     "bytes", scope_model, params, "Σ source numel × source dtype bytes",
                                     notes=["Checkpoint dtype; runtime dtype and replicated/sharded storage can differ."])}
    formulas = {
        "iteration_time_ms": "Σ completed decode-wave (last stage finish − first stage start) / wave count",
        "itl_ms": "token-sample-weighted wave service + mean between-wave wait",
        "ttft_ms": "mean(first token − arrival) = admission wait + service TTFT",
        "service_ttft_ms": "mean(first token − admitted)",
        "output_tokens_s": "(completed × output length + unfinished window outputs − completed pre-window outputs) / window",
        "achieved_qps": "completed requests / measurement window",
        "mfu": "window matrix FLOPs / (dense TFLOPS × 10^12 × GPU count × window seconds)",
        "gpu_activity_fraction": "Σ GPU union(clipped compute and communication activity) / GPU-window seconds",
        "modeled_busy_fraction": "Σ compute segment busy GPU seconds / GPU-window seconds (excludes communication)",
        "prefill_busy_fraction": "prefill compute segment busy GPU seconds / GPU-window seconds",
        "decode_busy_fraction": "decode compute segment busy GPU seconds / GPU-window seconds",
        "prefill_tokens_s": "prompt tokens of requests whose first token lands in window / window seconds",
        "tokens_s_per_gpu": "output tokens/s / GPU count",
        "prefill_tokens_s_per_gpu": "prefill tokens/s / GPU count",
        "offered_qps": "frozen configured request arrival rate",
        "ttft_max_ms": "max(first token − arrival) over first tokens in window",
        "itl_max_ms": "max(consecutive token timestamps difference) over intervals ending in window",
        "end_to_end_ms": "mean(last token − arrival) for completed requests in window",
        "completion_seconds": "last request completion timestamp for complete offline workload",
    }
    evidence = [fact(k, v, "ms" if k.endswith("_ms") else "") for k, v in e.items()
                if isinstance(v, (int, float))]
    evidence += [fact(k, m[k], UNITS.get(k, "")) for k in
                 ("window_seconds", "completed_requests", "emitted_tokens", "model_flops_in_window",
                  "service_ttft_ms", "iteration_time_ms")]
    peak = sources.hardware.get("compute", {}).get(design.runtime.activation_dtype, {}).get("dense")
    evidence += [fact("dense_peak", peak, "TFLOPS", "Hardware",
                      source_ref=f"frozen.sources.hardware.compute.{design.runtime.activation_dtype}.dense"),
                 fact("gpu_count", candidate.gpu_count)]
    for k, unit in UNITS.items():
        traces[k] = trace(k, k.replace("_", " ").title(), m.get(k), unit, scope, evidence,
                          formulas.get(k, "Unavailable: verified price/power measurements required"),
                          assumed=a, reason=e["reason"] if e["status"] == "unavailable" else None,
                          notes=sim["notes"])
    traces["effective_tokens"] = trace("effective_tokens", "Effective Output / Iteration",
                                      e["effective_tokens_per_iteration"], "tokens/wave", scope, evidence,
                                      "actual output tokens from completed decode waves / completed decode wave count",
                                      notes=["Baseline: each request emits one token per decode. Not speculative accepted tokens."])
    cycle = plan.get("speculative_cycle")
    if cycle:
        cycle_scope = scope_for(sources, candidate, kind=cycle["scope"])
        cycle_inputs = [fact(k, v, "s" if k.endswith("_seconds") else "", "Assumption" if k in
                            ("acceptance_rate", "draft_seconds") else "Derived")
                        for k, v in cycle.items() if isinstance(v, (int, float))]
        cycle_inputs += [fact("dense_peak", peak, "TFLOPS", "Hardware"),
                         fact("gpu_count", candidate.gpu_count)]
        for k, unit, formula in (
            ("expected_accepted_tokens", "tokens", "Σ acceptance_rate^i, i=1…draft_tokens"),
            ("expected_emitted_tokens", "tokens", "1 + expected accepted tokens"),
            ("cycle_seconds", "s", "target verify time + per-cycle draft latency"),
            ("tpot_ms", "ms", "1000 × cycle seconds / expected emitted tokens per request"),
            ("output_tokens_s", "tokens/s", "decode requests × expected emitted tokens / cycle seconds"),
            ("mfu", "fraction", "target matrix FLOPs / (dense peak × 1e12 × GPU count × cycle seconds)"),
        ):
            traces["spec_" + k] = trace("spec_" + k, "Speculative " + k.replace("_", " "), cycle[k],
                                       unit, cycle_scope, cycle_inputs, formula, assumed=a, notes=cycle["notes"])
    breakdown = []
    for k, v in e["decode_breakdown_ms"].items():
        formula = ("Σ wave internal resource wait / completed decode waves" if k == "scheduler" else
                   "Σ critical-rank selected roofline branch / completed decode waves" if k in ("compute", "memory")
                   else "Σ critical-rank launch × routing multiplier / completed decode waves" if k == "runtime"
                   else f"Σ serialized {k.upper()} communication duration / completed decode waves")
        component_inputs = [fact("component_total", e["decode_component_totals_ms"][k], "ms"),
                            fact("completed_decode_waves", e["decode_wave_count"], "waves")]
        if k in ("tp", "pp", "ep"):
            component_inputs += [fact(name, value, "bytes/s" if name.endswith("_bytes_s") else "",
                                      "Derived", source_ref=f"frozen.sources.topology.{name}")
                                 for name, value in sources.topology.items() if isinstance(value, (int, float))]
            formula += ("; collective=ring steps × max(shared domain bytes / (injection BW × efficiency)"
                        " + hop latency); PP/EP=sum store-and-forward transfer hops")
        t = trace("decode_" + k, LABELS[k], v, "ms", scope, component_inputs, formula, assumed=a,
                  reason=e["reason"] if e["status"] == "unavailable" else None)
        breakdown.append(t)
    valid = e["status"] == "available" and m["iteration_time_ms"] is not None
    values = e["decode_breakdown_ms"]
    grouped = {"Compute Bound": values["compute"], "Memory Bound": values["memory"],
               "Communication Bound": sum(values[k] or 0 for k in ("tp", "pp", "ep")),
               "Runtime Bound": values["runtime"], "Scheduler Bound": values["scheduler"]}
    topology = [fact("nic_spec", sources.hardware.get("node", {}).get("nic_gb_s_per_gpu"), "GB/s/GPU",
                     "Hardware", source_ref="frozen.sources.hardware.node.nic_gb_s_per_gpu"),
                fact("hbm_spec", sources.hardware.get("memory_bandwidth_bytes_s"), "bytes/s",
                     "Hardware", source_ref="frozen.sources.hardware.memory_bandwidth_bytes_s")]
    for k, v in design.resources.topology_overrides.model_dump().items():
        if v is not None:
            topology.append(fact(k, v, "bytes/s" if "bandwidth" in k else "", "Override",
                                 source_ref=f"frozen.design.resources.topology_overrides.{k}",
                                 formula="User topology override saved in design revision"))
    topology += [fact(k, v, "bytes/s" if k.endswith("_bytes_s") else "", "Derived",
                      source_ref=f"frozen.sources.topology.{k}")
                 for k, v in sources.topology.items() if isinstance(v, (int, float))]
    return {"version": VERSION, "scope": scope, "context": context, "traces": traces,
            "template_layer_counts": {t["id"]: sum(layer.template_id == t["id"] for layer in sources.ir.layers)
                                      for t in sources.structure["analysis"]["templates"]},
            "assumptions": a, "topology": topology,
            "performance": {**e, "breakdown": breakdown,
                            "bottleneck": max(grouped, key=grouped.get) if valid else "Unavailable"}}


def inspection_explanation(sources, candidate, design, result):
    scope = scope_for(sources, candidate, **{**result["scope"], "kind": "representative_batch"},
                      memory_kind="reserved_admission_bound",
                      result_hash=result["result_hash"])
    nodes = {}
    layer = sources.ir.layers[result["scope"]["layer"]]
    template = next(t for t in sources.structure["analysis"]["templates"] if t["id"] == layer.template_id)
    mappings = {"silu": "ffn_activation", "gelu": "ffn_activation", "multiply": "ffn_activation",
                "rope_q": "rope", "rope_k": "rope", "qk_matmul": "attention", "av_matmul": "attention",
                "attn_add": "residual_add", "ffn_add": "residual_add", "softmax": "softmax"}
    for node in template["nodes"]:
        shards = [s for s in result["shards"] if s["module_id"] == node.get("module_id")]
        op_id = node.get("module_id") or mappings.get(node["id"])
        ops = [o for o in result["timing"]["operators"] if op_id and o["id"] in (op_id, op_id + ".bias")]
        count = sum((prod(s["local_shape"]) + s["bias"]) * s["count"] for s in shards)
        weight = sum(s["bytes"] for s in shards)
        full = next((w for w in layer.weights if w.module_id == node.get("module_id")), None)
        local = shards[0]["local_shape"] if shards else None
        shapes = []
        for op in ops:
            dims = op["dimensions"]
            if "in" in dims:
                shapes += [fact("input", [dims["tokens"], dims["in"]], "shape"),
                           fact("weight", local, "shape"), fact("output", [dims["tokens"], dims["out"]], "shape"),
                           fact("M", dims["tokens"]), fact("K", dims["in"]), fact("N", dims["out"])]
        inputs = [fact("full_shape", list(full.shape) if full else None, "shape", "Model"),
                  fact("local_shape", local, "shape"), fact("tp", candidate.tp),
                  fact("expert_tp", candidate.expert_tp),
                  fact("runtime_weight_dtype", design.runtime.weight_dtype, source="Workload"),
                  fact("source_weight_dtype", full.dtype if full else None, source="Model"),
                  fact("count", sum(s["count"] for s in shards)),
                  fact("bias", sum(s["bias"] for s in shards)),
                  fact("local_tokens", result["local_tokens"], "tokens")]
        traces = [trace(node["id"] + "_weight", "Weight and Shape", weight, "bytes", scope, inputs,
                        "PP/EP ownership → shard source axis by TP/ETP; (prod(local shape) + bias) × count × dtype bytes",
                        steps=[{"label": "Parameters", "value": count, "unit": "parameters",
                                "formula": "Σ (prod(local shape) + bias) × count"}])]
        for op in ops:
            scalar = bool(op["scalar_equivalent_ops"])
            f = [fact(k, op.get(k), u, src) for k, u, src in (
                ("flops", "FLOPs", "Derived"), ("scalar_equivalent_ops", "ops", "Derived"),
                ("hbm_bytes", "bytes", "Derived"), ("peak_tflops", "TFLOPS", "Hardware"),
                ("hbm_bandwidth_bytes_s", "bytes/s", "Override" if
                 sources.assumptions["hbm_bandwidth_gb_s"]["effective"] else "Hardware"))]
            formula = ("scalar-equivalent profile; FP32 peak" if scalar else
                       "FLOPs = 2 × M × K × N" if "in" in op["dimensions"] else
                       "attention FLOPs = 4 × local heads × head dim × causal pairs")
            f += [fact(k, v) for k, v in op.get("traffic_inputs", {}).items()]
            f += [fact(k, v) for k, v in op["dimensions"].items()]
            traces.append(trace(op["id"], op["id"] + " Latency", op["seconds"], "s", scope, f + shapes,
                                formula + "; compute=ops/(peak×1e12×compute_eff); memory=traffic/(HBM BW×HBM eff); "
                                "total=(max(compute,memory)+launch)×duration multiplier; HBM traffic="
                                + op.get("traffic_formula", "scalar runtime profile traffic (see operator source)"),
                                steps=[{"label": k, "value": op.get(k), "unit": "s", "formula": k}
                                       for k in ("compute_seconds", "hbm_seconds", "launch_seconds")],
                                assumed=assumptions(sources),
                                notes=[(f"Selected: {op.get('bottleneck', 'unknown')}; "
                                        f"duration multiplier {op.get('duration_multiplier', 1)}")]))
        nodes[node["id"]] = {"parameters": count, "weight_bytes": weight, "shapes": shapes,
                             "full_shape": list(full.shape) if full else None, "local_shape": local,
                             "traces": traces, "operators": ops,
                             "attribution": "Shared aggregate; do not add again for another node" if
                             node["id"] in mappings and node["id"] != "softmax" else
                             "No independent runtime kernel modeled" if not ops else "Independent operator",
                             "expert_ranges": [[s["expert_start"], s["expert_stop"]] for s in shards
                                               if s["expert_start"] is not None]}
    steps = [{"step": step, "bytes": sum(a["bytes"] for a in result["allocations"]
                                       if a["start"] <= step < a["stop"]),
              "allocation_ids": [a["id"] for a in result["allocations"] if a["start"] <= step < a["stop"]]}
             for step in range(14)]
    lifetime = trace("lifetime", "Layer Activation Peak", result["activation_peak_bytes"], "bytes", scope,
                     [fact(str(s["step"]), s["bytes"], "bytes", label=f"Step {s['step']}") for s in steps],
                     "max over kernel steps (Σ live tensors); lifetimes are half-open [allocate, free)",
                     notes=["Ordinal kernel steps, not elapsed seconds; tensors do not all remain resident together."])
    return {"version": VERSION, "scope": scope, "nodes": nodes, "steps": steps, "lifetime": lifetime}
