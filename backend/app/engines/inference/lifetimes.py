"""Allocation identities and a deterministic simultaneous-memory sweep."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Allocation:
    id: str
    component: str
    bytes: int
    start: float
    stop: float
    source: str


def peak_ledger(base, allocations):
    """Half-open lifetimes [start, stop); frees precede allocations at ties."""
    events = []
    seen = set()
    for alloc in allocations:
        if alloc.id in seen or alloc.bytes < 0 or alloc.stop <= alloc.start:
            raise ValueError("Invalid or duplicate allocation identity/lifetime")
        seen.add(alloc.id)
        events.extend([(alloc.start, 1, alloc.id, alloc.component, alloc.bytes),
                       (alloc.stop, 0, alloc.id, alloc.component, -alloc.bytes)])
    current = dict(base)
    maxima = dict(base)
    peak, peak_at, components = sum(current.values()), 0.0, dict(current)
    timeline = [{"at": 0.0, "total_bytes": peak, "components": dict(current)}]
    for at, _, _, component, delta in sorted(events):
        current[component] = current.get(component, 0) + delta
        maxima[component] = max(maxima.get(component, 0), current[component])
        total = sum(current.values())
        if total > peak:
            peak, peak_at, components = total, at, dict(current)
        timeline.append({"at": at, "total_bytes": total, "components": dict(current)})
    return {"peak_bytes": peak, "peak_at": peak_at, "components": components,
            "component_maxima": maxima, "timeline": timeline}


def activation_program(layer, tokens, tp, element_bytes, *, expert_tokens=None, expert_tp=None):
    """Unfused norm/FlashAttention/SwiGLU; time is ordinal kernel step, not seconds.

    Input residual survives until the final add. Attention scores are streamed
    in fixed 128x128 tiles, with FP32 accumulators. Different layers reuse pools.
    """
    h = tokens * layer.hidden * element_bytes
    q = tokens * (layer.heads // tp) * layer.head_dim * element_bytes
    kv = tokens * max(1, layer.kv_heads // tp) * layer.head_dim * element_bytes
    routed = tokens if expert_tokens is None else expert_tokens
    i = routed * (layer.intermediate // (expert_tp or tp)) * element_bytes
    rows = [
        ("residual", h, 0, 14), ("norm", h, 1, 5),
        ("q", q, 2, 7), ("k", kv, 3, 7), ("v", kv, 4, 7),
        ("flash-output", q, 5, 8),
        ("flash-workspace", min(tokens, 128) * (layer.heads // tp) * (128 + layer.head_dim) * 4, 5, 7),
        ("o", h, 7, 9), ("ffn-norm", h, 9, 13),
        ("gate", i, 10, 12), ("up", i, 10, 12), ("product", i, 11, 13),
        ("down", routed * layer.hidden * element_bytes, 12, 14),
    ]
    if layer.experts:
        rows += [("router-logits", tokens * layer.experts * 4, 9, 11),
                 ("route-indices", tokens * layer.top_k * 12, 10, 14),
                 ("dispatch-input", routed * layer.hidden * element_bytes, 10, 14)]
        # Shared experts execute sequentially, retaining the routed output.
        shared_width = sum(w.shape[0] * w.count for w in layer.weights
                           if not w.routed and w.module_id.endswith("gate_proj"))
        rows += [("shared-ffn", tokens * (3 * shared_width // tp + layer.hidden) * element_bytes, 13, 14)]
    return tuple(Allocation(f"{layer.instance_id}/{name}", "activation", size, start, stop,
                            layer.instance_id) for name, size, start, stop in rows if size)
