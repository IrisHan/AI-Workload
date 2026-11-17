"""Deterministic colocated request/PP event scheduler, bounded by work budget."""
import heapq
import random
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from math import ceil

from .cache import request_quotas
from .compute import stage_program


@dataclass
class Request:
    id: int
    arrival: float
    d: int = -1
    e: int = -1
    admitted: float = 0
    context: int = 0
    produced: int = 0
    busy: bool = False
    first: float | None = None
    last: float | None = None
    intervals: list = field(default_factory=list)
    window_outputs: int = 0


def add_interval(intervals, start, stop):
    """Maintain disjoint sorted half-open intervals, including out-of-order PP receives."""
    if stop <= start:
        return
    i = bisect_left(intervals, (start, float("-inf")))
    if i and intervals[i - 1][1] >= start:
        i -= 1
    j = i
    while j < len(intervals) and intervals[j][0] <= stop:
        start, stop = min(start, intervals[j][0]), max(stop, intervals[j][1])
        j += 1
    intervals[i:j] = [(start, stop)]


def simulate(sources, candidate, design, event_budget=1000000, *, cancelled=None, details=False):
    """Stage compute resources are independent; shared communication is serialized.

    A request belongs to one (d,e) for its lifetime. EP waves synchronize all
    groups, while ready subsets allow continuous batching and PP microbatches.
    """
    load, runtime = design.load, design.runtime
    quotas = request_quotas(load.concurrency, candidate)
    active = [[[] for _ in range(candidate.attention_groups)] for _ in range(candidate.independent_replicas)]
    stage_free = [[0.0] * candidate.pp for _ in range(candidate.independent_replicas)]
    rounds = [0] * candidate.independent_replicas
    sequential_busy = [False] * candidate.independent_replicas
    link_free = 0.0
    pending = deque()
    heap = []
    rng = random.Random(load.seed)
    next_arrival, next_id = 0.0, 0
    now = 0.0
    generated = completed = emitted = prompt_completed = work = wave_id = 0
    flops = hbm = comm_time = measured_flops = 0.0
    phase_busy = {"prefill": 0.0, "decode": 0.0}
    decode_iterations, wave_starts = [], {}
    component_keys = ("compute", "memory", "tp", "pp", "ep", "runtime", "scheduler")
    decode_components = dict.fromkeys(component_keys, 0.0)
    wave_components = {}
    activity = [[] for _ in range(candidate.gpu_count)]
    itl_service = itl_wait = 0.0
    decode_outputs = completed_before_window = completed_in_window_outputs = 0
    ttfts, service_ttfts, intervals, end_to_end = [], [], [], []
    waves, request_rows = [], []
    cache = {}
    status, reason = "completed", None
    online = design.mode == "online"
    horizon = load.duration_seconds if online else float("inf")
    last_finish = 0.0
    unknown = []

    def has_arrival():
        return next_arrival < horizon if online else next_id < load.request_count

    def enqueue_arrivals(at):
        nonlocal next_id, next_arrival, generated, work
        while has_arrival() and next_arrival <= at and work < event_budget:
            pending.append(Request(next_id, next_arrival))
            next_id += 1
            generated += 1
            work += 1
            if online:
                next_arrival += (rng.expovariate(load.request_rate) if load.arrival == "poisson"
                                 else 1 / load.request_rate)
            else:
                next_arrival = 0.0

    def admit(at):
        for d in range(candidate.independent_replicas):
            if not runtime.continuous_batching and any(active[d]):
                continue
            for e in range(candidate.attention_groups):
                while pending and len(active[d][e]) < quotas[d][e]:
                    request = pending.popleft()
                    request.d, request.e, request.admitted = d, e, at
                    active[d][e].append(request)

    def select(d):
        selected = {"prefill": [[] for _ in range(candidate.attention_groups)],
                    "decode": [[] for _ in range(candidate.attention_groups)]}
        force = rounds[d] % runtime.fairness_interval == runtime.fairness_interval - 1
        for e in range(candidate.attention_groups):
            ready = [r for r in active[d][e] if not r.busy]
            prefill = sorted((r for r in ready if r.context < load.prompt_tokens),
                             key=lambda r: (r.context, r.id))
            decode = sorted((r for r in ready if r.context >= load.prompt_tokens), key=lambda r: r.id)
            budget, slots = runtime.max_batched_tokens, runtime.max_batch_sequences
            order = [("prefill", prefill), ("decode", decode)] if force else [
                ("decode", decode), ("prefill", prefill)]
            for phase, requests in order:
                for request in requests:
                    if not slots or not budget:
                        break
                    count = min(runtime.prefill_chunk_tokens, load.prompt_tokens - request.context, budget) \
                        if phase == "prefill" else 1
                    last = phase == "decode" or request.context + count == load.prompt_tokens
                    selected[phase][e].append((request, count, last))
                    budget -= count
                    slots -= 1
                    request.busy = True
        rounds[d] += 1
        return selected

    enqueue_arrivals(0)
    while True:
        if work >= event_budget:
            status, reason = "partial", "event_budget"
            break
        if cancelled and cancelled():
            status, reason = "cancelled", "cancel_requested"
            break
        enqueue_arrivals(now)
        admit(now)
        scheduled = False
        for d in range(candidate.independent_replicas):
            if sequential_busy[d] and not runtime.continuous_batching:
                continue
            selection = select(d)
            for phase in ("prefill", "decode"):
                batch = selection[phase]
                if not any(batch):
                    continue
                batch_cost = [[(r.context, q, last) for r, q, last in group] for group in batch]
                previous = now
                stage_events = []
                components = dict.fromkeys(component_keys, 0.0)
                for p in range(candidate.pp):
                    if work >= event_budget:
                        status, reason = "partial", "event_budget"
                        break
                    key = (d, p, tuple(tuple(row) for row in batch_cost))
                    if key not in cache:
                        if len(cache) > 4096:
                            cache.clear()
                        cache[key] = stage_program(sources, candidate, design, d, p, batch_cost)
                    cost = cache[key]
                    work += 1
                    if cost["seconds"] is None:
                        unknown.extend(cost["unknown"])
                        status, reason = "unknown", "required_timing_unknown"
                        break
                    start = max(previous, stage_free[d][p])
                    if p:
                        components["scheduler"] += start - previous
                    communication = cost["communication_seconds"]
                    finish = start
                    segments = []
                    for segment in cost["segments"]:
                        begin = max(finish, link_free) if segment["kind"] == "communication" and (
                            segment["seconds"]) else finish
                        components["scheduler"] += begin - finish
                        finish = begin + segment["seconds"]
                        for name, value in segment["components"].items():
                            components[name] += value
                        for gpu, seconds in segment["rank_activity_seconds"].items():
                            add_interval(activity[gpu], max(begin, load.warmup_seconds),
                                         min(begin + seconds, horizon))
                        if segment["kind"] == "communication" and segment["seconds"]:
                            link_free = finish
                        if segment["kind"] == "compute" and segment["seconds"]:
                            overlap = max(0.0, min(finish, horizon) - max(begin, load.warmup_seconds))
                            fraction = overlap / segment["seconds"]
                            measured_flops += segment["model_flops"] * fraction
                            phase_busy[phase] += segment["busy_gpu_seconds"] * fraction
                        segments.append({**segment, "start": begin, "stop": finish})
                    stage_free[d][p] = finish
                    previous = finish
                    stage_events.append({"stage": p, "start": start, "stop": finish, "segments": segments,
                                         "tokens_per_group": [sum(q for _, q, _ in g) for g in batch]})
                    flops += cost["flops"]
                    hbm += cost["hbm_bytes"]
                    comm_time += communication
                if status != "completed":
                    break
                heapq.heappush(heap, (previous, wave_id, d, phase, batch))
                wave_starts[wave_id] = stage_events[0]["start"]
                wave_components[wave_id] = components
                if details and len(waves) < 2000:
                    waves.append({"id": wave_id, "replica": d, "phase": phase, "stages": stage_events,
                                  "requests": [[r.id for r, _, _ in group] for group in batch],
                                  "contexts": batch_cost})
                wave_id += 1
                sequential_busy[d] = True
                scheduled = True
            if status != "completed":
                break
        if status != "completed":
            break
        # Each ready request was selected or is waiting behind other ready work.
        # Submit another stage-0 wave at this logical time if capacity remains.
        if scheduled and runtime.continuous_batching and any(
            not r.busy for replica in active for group in replica for r in group
        ):
            continue
        arrival = next_arrival if has_arrival() else float("inf")
        finish = heap[0][0] if heap else float("inf")
        future = min(arrival, finish)
        if future == float("inf"):
            break
        if future >= horizon:
            now = horizon
            status = "censored" if heap or pending or any(any(g for g in replica) for replica in active) else status
            break
        now = future
        while heap and heap[0][0] <= now:
            at, finished_wave, d, phase, batch = heapq.heappop(heap)
            wave_start = wave_starts.pop(finished_wave)
            components = wave_components.pop(finished_wave)
            if phase == "decode" and wave_start >= load.warmup_seconds:
                decode_iterations.append((at - wave_start) * 1000)
                decode_outputs += sum(int(last) for group in batch for _, _, last in group)
                for name, value in components.items():
                    decode_components[name] += value * 1000
            work += 1
            sequential_busy[d] = False
            for group in batch:
                for request, count, last in group:
                    request.context += count
                    request.busy = False
                    if not last:
                        continue
                    request.produced += 1
                    measured = at >= load.warmup_seconds
                    if request.first is None:
                        request.first = at
                        if measured:
                            ttfts.append((at - request.arrival) * 1000)
                            service_ttfts.append((at - request.admitted) * 1000)
                            prompt_completed += load.prompt_tokens
                    elif request.last is not None:
                        interval = (at - request.last) * 1000
                        request.intervals.append(interval)
                        if measured:
                            intervals.append(interval)
                            itl_service += (at - wave_start) * 1000
                            itl_wait += (wave_start - request.last) * 1000
                    request.last = at
                    emitted += int(measured)
                    request.window_outputs += int(measured)
                    if request.produced == load.output_tokens:
                        active[d][request.e].remove(request)
                        last_finish = max(last_finish, at)
                        completed += int(measured)
                        if measured:
                            completed_before_window += request.produced - request.window_outputs
                            completed_in_window_outputs += request.window_outputs
                            end_to_end.append((at - request.arrival) * 1000)
                        if details and len(request_rows) < 10000:
                            request_rows.append({"id": request.id, "arrival": request.arrival,
                                                 "admitted": request.admitted, "first": request.first,
                                                 "finished": at, "d": d, "e": request.e,
                                                 "output_tokens": request.produced, "context": request.context})
        if work >= event_budget:
            status, reason = "partial", "event_budget"
            break
    duration = (min(now, load.duration_seconds) if online else last_finish) - load.warmup_seconds
    duration = max(0.0, duration)
    # A normal online run measures the entire requested window including idle tail.
    if online and status in {"completed", "censored"}:
        duration = load.duration_seconds - load.warmup_seconds
    outstanding = len(pending) + sum(len(group) for replica in active for group in replica)
    def mean(values):
        return sum(values) / len(values) if values else None

    throughput = emitted / duration if duration else None
    gpu_window = duration * candidate.gpu_count
    peak = sources.hardware.get("compute", {}).get(runtime.activation_dtype, {})
    dense = peak.get("dense") if peak.get("status") == "supported" else None
    complete = status == "completed" and not outstanding
    ordered = sorted(end_to_end)
    percentiles = {name: ordered[ceil(len(ordered) * fraction) - 1]
                   if complete and len(ordered) >= minimum else None
                   for name, fraction, minimum in (("p50_ms", .5, 20), ("p95_ms", .95, 100),
                                                   ("p99_ms", .99, 10000))}
    metrics = {
        "ttft_ms": mean(ttfts), "ttft_max_ms": max(ttfts, default=None),
        "itl_max_ms": max(intervals, default=None), "service_ttft_ms": mean(service_ttfts),
        "itl_ms": mean(intervals) if load.output_tokens > 1 else None,
        "end_to_end_ms": mean(end_to_end), "output_tokens_s": throughput,
        "tokens_s_per_gpu": throughput / candidate.gpu_count if throughput is not None else None,
        "prefill_tokens_s": prompt_completed / duration if duration else None,
        "achieved_qps": completed / duration if duration else None, "offered_qps": load.request_rate,
        "completed_requests": completed, "emitted_tokens": emitted, "censored_count": outstanding,
        "generated_requests": generated, "window_seconds": duration,
        "completion_seconds": last_finish if not online and not outstanding else None,
        "modeled_busy_fraction": sum(phase_busy.values()) / gpu_window if gpu_window else None,
        "gpu_activity_fraction": sum(stop - start for spans in activity for start, stop in spans)
        / gpu_window if gpu_window else None,
        "prefill_busy_fraction": phase_busy["prefill"] / gpu_window if gpu_window else None,
        "decode_busy_fraction": phase_busy["decode"] / gpu_window if gpu_window else None,
        "mfu": measured_flops / (dense * 1e12 * gpu_window) if dense and gpu_window else None,
        "model_flops_in_window": measured_flops,
        "iteration_time_ms": mean(decode_iterations), "decode_iteration_count": len(decode_iterations),
        "prefill_tokens_s_per_gpu": prompt_completed / gpu_window if gpu_window else None,
        "concurrency": load.concurrency, "replica_count": candidate.independent_replicas,
        "queue_length": len(pending), "overloaded": bool(pending), "distribution": load.arrival,
        "seed": load.seed, "warmup_seconds": load.warmup_seconds, "sample_count": len(ttfts),
        "percentiles": percentiles, "percentile_sample_count": len(ordered),
        "percentile_policy": "nearest-rank; minimum samples P50=20, P95=100, P99=10000; no censoring",
        "percentile_reason": None if complete and len(ordered) >= 10000 else
        "Incomplete/censored run" if not complete else "Insufficient samples for one or more percentiles",
        "cost_per_million_output_tokens": None, "energy_per_million_output_tokens": None,
        "cost_reason": "Verified price date and billing basis required",
        "energy_reason": "Runtime power measurement required; TDP is not operating power",
    }
    if status in {"unknown", "partial", "cancelled"}:
        for key in ("ttft_ms", "service_ttft_ms", "itl_ms", "output_tokens_s", "tokens_s_per_gpu",
                    "prefill_tokens_s", "achieved_qps", "modeled_busy_fraction", "prefill_busy_fraction",
                    "decode_busy_fraction", "mfu", "model_flops_in_window", "iteration_time_ms",
                    "prefill_tokens_s_per_gpu", "gpu_activity_fraction"):
            metrics[key] = None
    explanation = {
        "status": "available" if status in {"completed", "censored"} else "unavailable",
        "reason": reason, "scope": "primary_simulation_measurement_window",
        "decode_breakdown_ms": {key: value / len(decode_iterations) if decode_iterations else None
                                for key, value in decode_components.items()},
        "decode_component_totals_ms": dict(decode_components),
        "decode_wall_total_ms": sum(decode_iterations),
        "decode_wave_count": len(decode_iterations), "decode_output_tokens": decode_outputs,
        "effective_tokens_per_iteration": decode_outputs / len(decode_iterations) if decode_iterations else None,
        "itl_samples": len(intervals), "itl_wave_service_ms": itl_service / len(intervals) if intervals else None,
        "itl_between_wave_wait_ms": itl_wait / len(intervals) if intervals else None,
        "ttft_admission_wait_ms": mean([a - b for a, b in zip(ttfts, service_ttfts)]),
        "completed_full_outputs": completed * load.output_tokens,
        "completed_outputs_before_window": completed_before_window,
        "unfinished_outputs_in_window": emitted - completed_in_window_outputs,
        "gpu_activity_seconds": sum(stop - start for spans in activity for start, stop in spans),
        "gpu_window_seconds": gpu_window,
    }
    if explanation["status"] == "unavailable":
        explanation["decode_breakdown_ms"] = dict.fromkeys(component_keys)
        explanation["decode_component_totals_ms"] = dict.fromkeys(component_keys)
        explanation["decode_wall_total_ms"] = None
        for key in ("effective_tokens_per_iteration", "itl_wave_service_ms", "itl_between_wave_wait_ms",
                    "ttft_admission_wait_ms", "gpu_activity_seconds"):
            explanation[key] = None
    sla = "unknown" if outstanding or status != "completed" else "pass"
    for key, target in (("ttft_max_ms", design.targets.ttft_ms), ("itl_max_ms", design.targets.itl_ms)):
        value = metrics[key]
        if target is not None and value is not None and value > target:
            sla = "fail"
    return {"status": status, "reason": reason, "metrics": metrics, "sla": sla, "explanation": explanation,
            "unknown": sorted(set(unknown)), "events_processed": work, "wave_count": wave_id,
            "flops": flops, "hbm_bytes": hbm, "communication_seconds": comm_time,
            "waves": waves, "requests": request_rows, "kind": "modeled",
            "trace_complete": details and len(waves) == wave_id and len(request_rows) >= completed,
            "simulated_until": now,
            "notes": ["Fixed-length request simulation; values and percentiles are modeled, not measurements.",
                      "Work in a compute segment is uniformly distributed when clipping the measurement window.",
                      "Iteration time is complete decode-wave service time, including pipeline/communication waits.",
                      "Global communication resource serializes concurrent groups without overlap credit."]}
