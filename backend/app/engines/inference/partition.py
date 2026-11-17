"""Orthogonal TP/EP partition rules and contiguous pipeline placement."""
from math import prod

from .ir import Candidate, ModelIR, PlanningError, Rank, Shard, Weight


def make_candidate(ir, design, spec):
    values = spec.model_dump() if hasattr(spec, "model_dump") else dict(spec)
    return Candidate(**{**values, "boundaries": tuple(values["boundaries"]),
                        "layout": design.parallelism.layout,
                        "moe": any(layer.experts for layer in ir.layers)})


def default_boundaries(layer_count, pp, layout):
    sizes = [layer_count // pp] * pp
    remainder = layer_count % pp
    if layout == "vllm-v0.10.2":
        for i in range(remainder):
            sizes[-2 - i] += 1
    else:
        sizes[-1] += remainder
    return tuple(sum(sizes[:i]) for i in range(1, pp))


def framework_config(ir, candidate, design):
    c = candidate
    if c.legacy:
        return {"layout": c.layout, "args": [], "env": {}, "status": "legacy_not_exportable"}
    cuts = (0, *c.boundaries, len(ir.layers))
    counts = [b - a for a, b in zip(cuts, cuts[1:])]
    vllm = c.layout == "vllm-v0.10.2"
    args = (["--tensor-parallel-size", str(c.tp), "--pipeline-parallel-size", str(c.pp),
             "--data-parallel-size", str(c.dp)] if vllm else
            ["--tp-size", str(c.tp), "--pp-size", str(c.pp), "--dp-size", str(c.dp),
             "--ep-size", str(c.ep)])
    if vllm and c.moe and design.parallelism.expert_parallel:
        args.append("--enable-expert-parallel")
    return {"layout": c.layout, "status": "parallelism_only",
            "args": args, "env": {("VLLM" if vllm else "SGLANG") + "_PP_LAYER_PARTITION":
                                 ",".join(map(str, counts))},
            "stage_layer_counts": counts,
            "notes": ["Parallelism arguments only; model, distributed launch and draft setup remain required."]}


def validate_candidate(ir: ModelIR, candidate: Candidate, design):
    c = candidate
    budget = design.resources.node_count * design.resources.gpus_per_node
    if min(c.tp, c.pp, c.ep, c.dp) < 1 or c.gpu_count > budget:
        raise PlanningError("GPU_BUDGET", "Candidate exceeds resource budget", "resources")
    if len(c.boundaries) != c.pp - 1 or tuple(sorted(set(c.boundaries))) != c.boundaries:
        raise PlanningError("INVALID_BOUNDARIES", "Pipeline requires pp-1 strictly increasing boundaries")
    if any(b <= 0 or b >= len(ir.layers) for b in c.boundaries) or c.pp > len(ir.layers):
        raise PlanningError("INVALID_BOUNDARIES", "Every pipeline stage must contain at least one layer")
    parallel = design.parallelism
    if not c.legacy:
        if c.layout == "vllm-v0.10.2":
            expected = c.tp * c.dp if c.moe and parallel.expert_parallel else 1
            if c.ep != expected:
                raise PlanningError("FRAMEWORK_EP", f"vLLM requires EP={expected} for this configuration")
        elif c.tp % c.ep:
            raise PlanningError("FRAMEWORK_EP", "SGLang EP must divide TP")
        if c.layout == "sglang-v0.5.3":
            if c.dp > 1 and design.resources.node_count > 1:
                raise PlanningError("FRAMEWORK_DP", "SGLang standard DP is restricted to one node")
            if c.pp > 1 and design.runtime.speculative.enabled:
                raise PlanningError("FRAMEWORK_SPEC", "SGLang v0.5.3 does not support PP with speculative decoding")
    for name in ("tp", "pp", "ep"):
        allowed = getattr(parallel, f"allowed_{name}")
        if allowed is not None and getattr(c, name) not in allowed:
            raise PlanningError("PARALLEL_LOCK", f"{name} excluded by global lock", f"parallelism.allowed_{name}")
    if parallel.pp_boundaries is not None and tuple(parallel.pp_boundaries) != c.boundaries:
        raise PlanningError("PARALLEL_LOCK", "Candidate differs from locked pipeline boundaries")
    for lock in parallel.layer_locks:
        if lock.layer_index >= len(ir.layers):
            raise PlanningError("INVALID_LAYER", "Layer lock outside model", "parallelism.layer_locks")
        if c.tp not in lock.allowed_tp:
            raise PlanningError("LAYER_LOCK", f"Layer {lock.layer_index} excludes TP={c.tp}",
                                "parallelism.layer_locks", {"layer": lock.layer_index})
    if not any(layer.experts for layer in ir.layers) and c.ep != 1:
        raise PlanningError("DENSE_EP", "Dense model requires EP=1")
    if ir.vocab % c.tp:
        raise PlanningError("VOCAB_DIVISIBILITY", "Vocabulary must be divisible by TP")
    for layer in ir.layers:
        for name, value in (("query heads", layer.heads), ("KV heads", layer.kv_heads)):
            replicated_kv = name == "KV heads" and not c.legacy and c.tp % value == 0
            if value % c.tp and not replicated_kv:
                raise PlanningError("HEAD_DIVISIBILITY", f"Layer {layer.index}: {name} not divisible by TP",
                                    details={"layer": layer.index, "value": value, "tp": c.tp})
        if layer.experts and layer.experts % c.ep:
            raise PlanningError("EXPERT_DIVISIBILITY", "Expert count must be divisible by EP",
                                details={"layer": layer.index})
        for weight in layer.weights:
            shard_weight(weight, c, 0, design.runtime.weight_dtype, kv_heads=layer.kv_heads)


def ranks(candidate: Candidate, gpus_per_node: int):
    return tuple(
        Rank(rank=r, d=r // (candidate.pp * candidate.attention_groups * candidate.tp),
             p=(r // (candidate.attention_groups * candidate.tp)) % candidate.pp,
             e=(r // candidate.tp) % candidate.attention_groups, t=r % candidate.tp,
             node=r // gpus_per_node, gpu=r % gpus_per_node)
        for r in range(candidate.gpu_count)
    )


def groups(candidate: Candidate, placements):
    result = []
    for kind, fixed in (("tp", ("d", "p", "e")), ("ep", ("d", "p", "t")),
                        ("pp", ("d", "e", "t")), ("dp", ("p", "e", "t"))):
        grouped = {}
        for rank in placements:
            key = tuple(getattr(rank, axis) for axis in fixed)
            grouped.setdefault(key, []).append(rank.rank)
        for coordinates, members in grouped.items():
            result.append({"id": kind + ":" + ":".join(map(str, coordinates)),
                           "kind": kind, "ranks": members, "fixed": dict(zip(fixed, coordinates))})
    if not candidate.legacy:
        result = [g for g in result if g["kind"] != "ep"]
        for kind in ("etp", "ep"):
            grouped = {}
            for rank in placements:
                flat = rank.e * candidate.tp + rank.t
                coord = flat // candidate.expert_tp if kind == "etp" else flat % candidate.expert_tp
                grouped.setdefault((rank.d, rank.p, coord), []).append(rank.rank)
            for coords, members in grouped.items():
                result.append({"id": kind + ":" + ":".join(map(str, coords)),
                               "kind": kind, "ranks": members,
                               "fixed": dict(zip(("d", "p", "coordinate"), coords))})
    return result


def shard_weight(weight: Weight, candidate: Candidate, e: int, dtype: str, *, kv_heads=None):
    shape = list(weight.shape)
    bias, count = weight.bias, weight.count
    expert_start = expert_stop = None
    if weight.routed:
        if count % candidate.ep:
            raise PlanningError("EXPERT_DIVISIBILITY", f"{weight.module_id}: experts not divisible by EP")
        count //= candidate.ep
        expert_start, expert_stop = e * count, (e + 1) * count
    if weight.partition in {"column", "vocab", "row"}:
        axis = 1 if weight.partition == "row" else 0
        tp = candidate.expert_tp if weight.routed else candidate.tp
        if not candidate.legacy and weight.module_id in {"k_proj", "v_proj"} and kv_heads:
            tp = min(tp, kv_heads)
        if shape[axis] % tp:
            raise PlanningError("TENSOR_DIVISIBILITY", f"{weight.module_id}: axis not divisible by TP")
        shape[axis] //= tp
        if weight.partition == "column":
            bias //= tp
    # Router correction and explicitly FP32 state buffers retain their precision.
    elem = 4 if dtype == "fp32" or weight.dtype == "fp32" else 2
    return Shard(
        storage_id=weight.alias_of or weight.storage_id, module_id=weight.module_id,
        local_shape=tuple(shape), count=count, bias=bias,
        bytes=(prod(shape) + bias) * count * elem, partition=weight.partition,
        expert_start=expert_start, expert_stop=expert_stop,
    )


def rank_weights(ir: ModelIR, candidate: Candidate, rank: Rank, dtype: str):
    """Alias deduplication occurs per physical rank, never across PP stages."""
    result = {}
    for layer in ir.layers:
        if candidate.stage(layer.index) == rank.p:
            for weight in layer.weights:
                shard = shard_weight(weight, candidate, candidate.expert_owner(rank), dtype,
                                     kv_heads=layer.kv_heads)
                result.setdefault(shard.storage_id, shard)
    for weight in ir.shared:
        stage = 0 if weight.module_id == "embed" else candidate.pp - 1
        if stage == rank.p:
            shard = shard_weight(weight, candidate, rank.e, dtype)
            result.setdefault(shard.storage_id, shard)
    return tuple(result.values())
