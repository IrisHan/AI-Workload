"""Bounded deterministic candidate search; never overstates search coverage."""
import time
from collections import Counter
from dataclasses import asdict
from itertools import combinations
from math import comb, gcd

from .ir import Candidate, PlanningError
from .memory import admission_memory
from .metrics import metric_values
from .partition import default_boundaries, framework_config, groups, make_candidate, ranks, validate_candidate
from .scheduler import simulate
from .speculative import speculative_cycle


def divisors(value, limit):
    return [n for n in range(1, min(value, limit) + 1) if value % n == 0]


def boundaries(ir, pp, locked=None):
    n = len(ir.layers)
    if locked is not None:
        return [tuple(locked)], True
    if pp == 1:
        return [()], True
    if pp > n:
        return [], True
    if comb(n - 1, pp - 1) <= 256:
        return list(combinations(range(1, n), pp - 1)), True
    # Contiguous minimax partition DP with source weight costs, then neighbors.
    weights = [sum(w.numel for w in layer.weights) for layer in ir.layers]
    weights[0] += sum(w.numel for w in ir.shared if w.module_id == "embed")
    weights[-1] += sum(w.numel for w in ir.shared if w.module_id != "embed")
    prefix = [0]
    for weight in weights:
        prefix.append(prefix[-1] + weight)
    costs = {(0, 0): (0, ())}
    for stages in range(1, pp + 1):
        for stop in range(stages, n + 1):
            choices = []
            for start in range(stages - 1, stop):
                if (stages - 1, start) in costs:
                    before, cuts = costs[stages - 1, start]
                    choices.append((max(before, prefix[stop] - prefix[start]), cuts + ((start,) if start else ())))
            costs[(stages, stop)] = min(choices)
    balanced = costs[(pp, n)][1]
    candidates = {balanced, tuple(n * p // pp for p in range(1, pp))}
    for index in range(len(balanced)):
        for direction in (-1, 1):
            cuts = list(balanced)
            cuts[index] += direction
            if 0 < cuts[0] and cuts[-1] < n and len(set(cuts)) == len(cuts) and cuts == sorted(cuts):
                candidates.add(tuple(cuts))
    return sorted(candidates), False


def parallel_domains(ir, budget=256, layout="orthogonal-tp-ep-v1"):
    """The same source-derived divisors serve the search and its input UI."""
    head_divisor = ir.vocab
    for layer in ir.layers:
        head_divisor = gcd(head_divisor, gcd(layer.heads, layer.kv_heads))
    for weight in (*ir.shared, *(w for layer in ir.layers for w in layer.weights)):
        if weight.partition in {"column", "vocab", "row"}:
            head_divisor = gcd(head_divisor, weight.shape[1 if weight.partition == "row" else 0])
    expert_divisor = 0
    for layer in ir.layers:
        if layer.experts:
            expert_divisor = gcd(expert_divisor, layer.experts)
    tps = divisors(head_divisor, budget)
    if layout != "orthogonal-tp-ep-v1":
        tps = [tp for tp in range(1, budget + 1)
               if ir.vocab % tp == 0 and all(
                   layer.heads % tp == 0 and (layer.kv_heads % tp == 0 or tp % layer.kv_heads == 0)
                   and all(w.shape[1 if w.partition == "row" else 0] % tp == 0
                           for w in layer.weights if not w.routed
                           and w.module_id not in {"k_proj", "v_proj"}
                           and w.partition in {"column", "vocab", "row"})
                   for layer in ir.layers)]
    return {"tp": tps, "ep": divisors(expert_divisor or 1, budget),
            "pp": list(range(1, min(len(ir.layers), budget) + 1))}


def candidate_stream(ir, design):
    parallel = design.parallelism
    if parallel.manual_candidate:
        yield make_candidate(ir, design, parallel.manual_candidate), True
        return
    budget = design.resources.node_count * design.resources.gpus_per_node
    domains = parallel_domains(ir, budget, parallel.layout)
    if parallel.layout != "orthogonal-tp-ep-v1":
        moe = any(layer.experts for layer in ir.layers)
        for used in range(1, budget + 1):
            for tp in parallel.allowed_tp or domains["tp"]:
                for pp in parallel.allowed_pp or domains["pp"]:
                    if used % (tp * pp):
                        continue
                    dp = used // (tp * pp)
                    if dp < 1 or (dp > 1 and not parallel.dp_scaling):
                        continue
                    eps = ([tp * dp if moe and parallel.expert_parallel else 1]
                           if parallel.layout == "vllm-v0.10.2" else parallel.allowed_ep or domains["ep"])
                    for ep in eps:
                        cut = (tuple(parallel.pp_boundaries) if parallel.pp_boundaries is not None
                               else default_boundaries(len(ir.layers), pp, parallel.layout))
                        yield Candidate(tp, pp, ep, dp, cut, parallel.layout, moe), True
        return
    tps = parallel.allowed_tp or domains["tp"]
    eps = parallel.allowed_ep or domains["ep"]
    pps = parallel.allowed_pp or domains["pp"]
    partition_cache = {}
    for used in range(1, budget + 1):
        for tp in sorted(set(tps)):
            for ep in sorted(set(eps)):
                if used % (tp * ep):
                    continue
                for pp in sorted(set(pps)):
                    single = tp * ep * pp
                    if single > used or used % single:
                        continue
                    dp = used // single
                    if dp > 1 and not parallel.dp_scaling:
                        continue
                    if pp not in partition_cache:
                        partition_cache[pp] = boundaries(ir, pp, parallel.pp_boundaries)
                    cuts, exhaustive = partition_cache[pp]
                    for cut in cuts:
                        yield Candidate(tp, pp, ep, dp, cut), exhaustive


def evaluate(sources, candidate, design, event_budget, *, cancelled=None, details=False):
    validate_candidate(sources.ir, candidate, design)
    cycle = speculative_cycle(sources, candidate, design)
    if cycle and cycle["capacity_status"] == "fail":
        raise PlanningError("SPEC_CAPACITY", "Target verification plus draft memory exceeds HBM",
                            "runtime.speculative.draft_memory_bytes", {"memory": cycle["memory"]})
    memory = admission_memory(sources.ir, candidate, design, sources.hardware, sources.assumptions,
                              details=details)
    if not memory["capacity_pass"]:
        rank = memory["ranks"][memory["worst_rank"]]
        raise PlanningError("CAPACITY", "Per-rank admission peak exceeds HBM", "resources", {
            "rank": rank["rank"], "observed": rank["peak_bytes"], "limit": rank["capacity_bytes"],
            "deficit": rank["deficit_bytes"], "components": rank["components"],
            "suggestion": "Increase GPU budget or reduce concurrency/context; inspect pipeline boundaries.",
        })
    simulation = simulate(sources, candidate, design, event_budget, cancelled=cancelled, details=details)
    targeted = design.targets.ttft_ms is not None or design.targets.itl_ms is not None
    if design.targets.policy == "hard" and targeted and simulation["sla"] != "pass":
        raise PlanningError("SLA_" + simulation["sla"].upper(), "Hard SLA is not established by simulation",
                            "targets", {"metrics": simulation["metrics"]})
    placement = ranks(candidate, design.resources.gpus_per_node)
    rank_groups = groups(candidate, placement)
    nodes = {rank.rank: rank.node for rank in placement}
    cross = sum(len({nodes[r] for r in group["ranks"]}) > 1 for group in rank_groups
                if group["kind"] in {"tp", "ep"})
    return {
        "id": candidate.key, "candidate": asdict(candidate), "gpu_count": candidate.gpu_count,
        "tp": candidate.tp, "pp": candidate.pp, "ep": candidate.ep, "etp": candidate.expert_tp, "dp": candidate.dp,
        "node_count": len(set(nodes.values())),
        "unused_gpu_count": design.resources.node_count * design.resources.gpus_per_node - candidate.gpu_count,
        "pp_boundaries": list(candidate.boundaries), "capacity_status": "pass",
        "execution_status": simulation["status"], "target_status": simulation["sla"],
        "recommended": simulation["status"] in {"completed", "censored"},
        "recommendation_reasons": ["Per-rank conservative admission capacity passes",
                                   "Ranked by GPU count, cross-node groups, peak memory and communication",
                                   "SLA: " + simulation["sla"]],
        "minimum_proven": candidate.gpu_count == 1, "cross_node_groups": cross,
        "recommendation_status": "verified_scenario" if simulation["status"] in {"completed", "censored"}
        else "capacity_only",
        "memory": memory, "performance": simulation, "speculative_cycle": cycle,
        "metric_values": metric_values(simulation),
        "framework_config": framework_config(sources.ir, candidate, design),
        "placement": [asdict(rank) for rank in placement], "groups": rank_groups,
        "assumptions": sources.assumptions,
        "notes": ["Analytical scenario; no GPU execution measurement.",
                  "Canonical contiguous placement; unsearched placements may improve the result."],
    }


def ranking(plan):
    timing = plan["performance"]["communication_seconds"]
    ttft = plan["performance"]["metrics"]["ttft_ms"]
    return (plan["recommendation_status"] != "verified_scenario", plan["gpu_count"],
            plan["cross_node_groups"], plan["memory"]["peak_bytes"],
            timing if timing is not None else float("inf"),
            ttft if ttft is not None else float("inf"), plan["id"])


def search(sources, design, options, *, cancelled=None, progress=None):
    started = time.monotonic()
    deadline = started + options.max_seconds
    plans, rejected = [], []
    reasons = Counter()
    generated = evaluated = events = 0
    all_partitions = True
    status, stop_reason, last = "completed", None, None
    for candidate, exhaustive in candidate_stream(sources.ir, design):
        all_partitions &= exhaustive
        if cancelled and cancelled():
            status, stop_reason = "cancelled", "cancel_requested"
            break
        if generated >= options.max_candidates or time.monotonic() >= deadline or events >= options.event_budget:
            status, stop_reason = "partial", ("max_candidates" if generated >= options.max_candidates
                                            else "event_budget" if events >= options.event_budget else "wall_time")
            break
        generated += 1
        last = candidate.key
        try:
            plan = evaluate(sources, candidate, design, options.event_budget - events,
                            cancelled=lambda: time.monotonic() >= deadline or bool(cancelled and cancelled()))
            evaluated += 1
            events += plan["performance"]["events_processed"]
            if plan["performance"]["status"] == "cancelled":
                status, stop_reason = ("partial", "wall_time") if time.monotonic() >= deadline else (
                    "cancelled", "cancel_requested")
                break
            if plan["performance"]["status"] == "partial":
                status, stop_reason = "partial", "event_budget"
            plans.append(plan)
            plans.sort(key=ranking)
            del plans[options.top_k:]
        except PlanningError as exc:
            reasons[exc.code] += 1
            if len(rejected) < 100:
                rejected.append({"candidate": asdict(candidate), **exc.as_dict()})
        if progress:
            progress({"generated": generated, "evaluated": evaluated,
                      "events": events, "last_candidate": last, "reasons": dict(reasons)})
        if status != "completed":
            break
    if status == "completed" and not all_partitions:
        status, stop_reason = "partial", "heuristic_partitions"
    verified = [plan for plan in plans if plan["recommendation_status"] == "verified_scenario"]
    outcome = "feasible" if verified else "inconclusive" if status != "completed" or plans else "infeasible"
    return {"status": status, "outcome": outcome, "stop_reason": stop_reason, "plans": plans,
            "rejections": rejected, "coverage": {
                "generated": generated, "evaluated": evaluated, "rejected": sum(reasons.values()),
                "reasons": dict(reasons), "events": events, "all_partitions": all_partitions,
                "placement_space": "canonical-contiguous-v1", "all_placements": False,
                "last_candidate": last, "minimum_proven": bool(verified and verified[0]["gpu_count"] == 1),
            }}
