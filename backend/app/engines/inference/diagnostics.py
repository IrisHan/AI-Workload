"""Bounded event diagnostics, derived exclusively from a frozen simulation."""
from collections import defaultdict
from dataclasses import asdict

from .cache import cache_allocation
from .communication import routing_matrix
from .compute import stage_program
from .lifetimes import Allocation, activation_program, peak_ledger
from .memory import COMPONENTS, compose
from .partition import rank_weights, ranks


def event_memory(sources, candidate, design, simulation, *, allocation_budget=100000):
    """Map ordinal kernel lifetimes onto modeled layer intervals.

    Timing inside a layer is linear interpolation, explicitly an assumption.
    Prefix truncation never becomes a capacity proof; admission is separate.
    """
    placements = ranks(candidate, design.resources.gpus_per_node)
    allocations = defaultdict(list)
    cache_changes = defaultdict(list)
    ends = {r["id"]: r["finished"] for r in simulation["requests"]}
    elem = 4 if design.runtime.activation_dtype == "fp32" else 2
    until = simulation["simulated_until"]
    complete = simulation["trace_complete"]
    count = 0
    processed = 0
    for wave in simulation["waves"]:
        # Worst case bound before expanding rank/layer/allocation identities.
        estimate = sum(20 * (len(sources.ir.layers) + candidate.pp) for r in placements
                       if r.d == wave["replica"])
        if count + estimate > allocation_budget:
            complete = False
            break
        processed += 1
        for stage in wave["stages"]:
            start, stop = stage["start"], min(stage["stop"], until)
            if stop <= start:
                continue
            layer_segments = defaultdict(list)
            for segment in stage["segments"]:
                layer_segments[segment["layer_index"]].append(segment)
            selected = [r for r in placements if r.d == wave["replica"] and r.p == stage["stage"]]
            tokens = stage["tokens_per_group"]
            for rank in selected:
                key = rank.rank
                prefix = f'{wave["id"]}/{key}'
                local = wave["contexts"][rank.e]
                for layer in sources.ir.layers:
                    if candidate.stage(layer.index) != rank.p:
                        continue
                    segments = layer_segments[layer.index]
                    begin = min(s["start"] for s in segments)
                    finish = min(max(s["stop"] for s in segments), until)
                    for request_id, (cached, query, _) in zip(wave["requests"][rank.e], local):
                        size = cache_allocation(layer, [cached + query], candidate.tp,
                                                design.runtime.kv_dtype)["total_bytes"]
                        cache_changes[(key, layer.index, request_id)].append((begin, size))
                    if finish <= begin:
                        continue
                    expert_tokens = None
                    if layer.experts:
                        matrix = routing_matrix(tokens, layer.experts, layer.top_k)
                        width = layer.experts // candidate.ep
                        owner = candidate.expert_owner(rank)
                        expert_tokens = sum(sum(row[owner * width:(owner + 1) * width]) for row in matrix)
                    program = activation_program(layer, tokens[rank.e], candidate.tp, elem,
                                                 expert_tokens=expert_tokens,
                                                 expert_tp=candidate.expert_tp if layer.experts else candidate.tp)
                    for a in program:
                        allocations[key].append(Allocation(
                            f"{prefix}/{a.id}", a.component, a.bytes,
                            begin + (finish - begin) * a.start / 14,
                            begin + (finish - begin) * a.stop / 14, a.source))
                    # Send/receive pools coexist with the layer's activations.
                    communication = 2 * tokens[rank.e] * layer.hidden * elem if candidate.tp > 1 else 0
                    if candidate.ep > 1 and layer.experts:
                        communication += 2 * max(tokens[rank.e] * layer.top_k, expert_tokens) * (
                            layer.hidden * elem + 12)
                    if communication:
                        allocations[key].append(Allocation(f"{prefix}/{layer.index}/comm",
                                                          "communication", communication,
                                                          begin, finish, layer.instance_id))
                # Embedding/head pools live only during their own operators.
                for index, label in ((-1, "embedding"), (len(sources.ir.layers), "head")):
                    spans = layer_segments.get(index, [])
                    if not spans or (label == "head" and rank.p != candidate.pp - 1):
                        continue
                    begin = min(s["start"] for s in spans)
                    finish = min(max(s["stop"] for s in spans), until)
                    size = tokens[rank.e] * sources.ir.hidden * elem
                    if label == "head":
                        size += sum(last for _, _, last in local) * (sources.ir.vocab // candidate.tp) * 4
                    if finish > begin:
                        allocations[key].append(Allocation(f"{prefix}/{label}", "activation", size,
                                                          begin, finish, label))
                if candidate.pp > 1 and tokens[rank.e]:
                    ready = wave["stages"][rank.p - 1]["stop"] if rank.p else start
                    consumed = min(wave["stages"][rank.p + 1]["stop"], until) \
                        if rank.p + 1 < candidate.pp else stop
                    if consumed > ready:
                        allocations[key].append(Allocation(f"{prefix}/pipeline", "activation",
                                                          2 * tokens[rank.e] * sources.ir.hidden * elem,
                                                          ready, consumed, "pipeline-buffer"))
        count = sum(len(rows) for rows in allocations.values()) + sum(len(rows) for rows in cache_changes.values())
    if processed < len(simulation["waves"]):
        # Only display the fully traced prefix, excluding untraced intervals.
        until = min(until, simulation["waves"][processed]["stages"][0]["start"])
    for (key, layer, request), changes in cache_changes.items():
        changes.sort()
        for index, (start, size) in enumerate(changes):
            stop = min(changes[index + 1][0] if index + 1 < len(changes) else ends.get(request, until), until)
            if stop > start:
                allocations[key].append(Allocation(f"kv/{layer}/{request}/{index}", "kv", size,
                                                  start, stop, sources.ir.layers[layer].instance_id))
    result = []
    for rank in placements:
        weight = sum(s.bytes for s in rank_weights(sources.ir, candidate, rank, design.runtime.weight_dtype))
        trimmed = [Allocation(a.id, a.component, a.bytes, a.start, min(a.stop, until), a.source)
                   for a in allocations[rank.rank] if a.start < min(a.stop, until)]
        raw = peak_ledger({}, trimmed)["timeline"]
        timeline = []
        maxima = dict.fromkeys(COMPONENTS, 0)
        for point in raw:
            components = point["components"]
            ledger = compose(weight, components.get("kv", 0), components.get("activation", 0),
                             components.get("communication", 0), sources.hardware["memory_bytes"],
                             sources.assumptions)
            for component, value in ledger["components"].items():
                maxima[component] = max(maxima[component], value)
            timeline.append({"at": point["at"], "total_bytes": ledger["peak_bytes"],
                             "components": ledger["components"]})
        peak = max(timeline, key=lambda row: row["total_bytes"])
        stride = max(1, len(timeline) // 500)
        sampled = sorted({p["at"]: p for p in timeline[::stride] + [peak, timeline[-1]]}.values(),
                         key=lambda p: p["at"])
        result.append({**asdict(rank), "peak_bytes": peak["total_bytes"], "peak_at": peak["at"],
                       "component_at_peak": peak["components"], "component_maxima": maxima,
                       "capacity_bytes": sources.hardware["memory_bytes"],
                       "free_bytes": sources.hardware["memory_bytes"] - peak["total_bytes"],
                       "timeline": sampled, "timeline_sampled": len(sampled) < len(timeline)})
    return {"ranks": result, "coverage": "complete" if complete else "prefix",
            "until": until, "kind": "modeled", "allocations": count,
            "assumption": "Ordinal activation lifetimes interpolated over layer time; IO pools conservative.",
            "capacity_proof": False}


def representative_layers(sources, candidate, design, simulation):
    """One representative wave per phase/replica, with exact source shard refs."""
    seen, rows, communication = set(), [], []
    for wave in simulation["waves"]:
        marker = (wave["phase"], wave["replica"])
        if marker in seen:
            continue
        seen.add(marker)
        for stage in wave["stages"]:
            cost = stage_program(sources, candidate, design, wave["replica"], stage["stage"],
                                 wave["contexts"], details=True)
            for row in cost["operators"]:
                rows.append({**row, "stage": stage["stage"], "phase": wave["phase"], "wave_id": wave["id"]})
            for row in cost["communication"]:
                communication.append({**row, "stage": stage["stage"], "phase": wave["phase"],
                                      "wave_id": wave["id"]})
    return rows, communication
