"""Paged request state. Integer byte accounting, no request sharing."""
from math import ceil


def distribute(total, groups):
    """Stable exact quotas: larger groups first, no phantom requests."""
    return tuple(total // groups + (i < total % groups) for i in range(groups))


def request_quotas(concurrency, candidate):
    return tuple(distribute(count, candidate.attention_groups)
                 for count in distribute(concurrency, candidate.independent_replicas))


def kv_bytes(layer, context, tp, dtype="bf16", page_tokens=16, *, paged=True):
    retained = min(context, layer.window) if layer.window else context
    allocated = ceil(retained / page_tokens) * page_tokens if paged else retained
    return allocated * 2 * max(1, layer.kv_heads // tp) * layer.head_dim * (4 if dtype == "fp32" else 2)


def cache_allocation(layer, contexts, tp, dtype, page_tokens=16):
    payload = sum(kv_bytes(layer, s, tp, dtype, page_tokens) for s in contexts)
    logical = sum(kv_bytes(layer, s, tp, dtype, page_tokens, paged=False) for s in contexts)
    pages = sum(ceil((min(s, layer.window) if layer.window else s) / page_tokens) for s in contexts)
    # Runtime policy: an int64 page index per request/layer/page.
    return {"payload_bytes": payload, "metadata_bytes": pages * 8,
            "tail_waste_bytes": payload - logical, "total_bytes": payload + pages * 8}
