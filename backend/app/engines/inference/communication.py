"""Conservative directed transfers with shared injection/root/NIC domains."""
from collections import defaultdict
from math import ceil


def ring_time(group_size, message_bytes, bandwidth, latency, kind="all_reduce"):
    if group_size == 1 or message_bytes == 0:
        return 0.0
    if bandwidth is None or bandwidth <= 0:
        return None
    factor = 2 if kind == "all_reduce" else 1
    return factor * (group_size - 1) * (latency + message_bytes / group_size / bandwidth)


def transfer_round(transfers, placements, topology):
    """Each transfer is (src,dst,bytes). Three store-and-forward hop stages.

    Links on a given hop run concurrently but share endpoint injection and node
    uplinks. A PCIe root is conservatively shared by all local sends/receives.
    """
    lookup = {r.rank: r for r in placements}
    loads = defaultdict(float)
    paths = []
    active = set()
    for src, dst, size in transfers:
        if src == dst or not size:
            continue
        a, b = lookup[src], lookup[dst]
        cross = a.node != b.node
        path = [f"gpu:{src}:out"]
        if cross:
            path += [f"node:{a.node}:out", f"node:{b.node}:in"]
            loads[("network", f"gpu:{src}:out")] += size
            loads[("network", f"gpu:{dst}:in")] += size
            loads[("network", f"node:{a.node}:out")] += size
            loads[("network", f"node:{b.node}:in")] += size
            active.add("network")
        path += [f"gpu:{dst}:in"]
        # Network transfer also traverses GPU-side local fabric at both ends.
        for hop, rank, direction in (("send", a, "out"), ("receive", b, "in")):
            loads[(hop, f"gpu:{rank.rank}:{direction}")] += size
            if "pcie-root" in topology["sharing_domains"]:
                loads[(hop, f"root:{rank.node}")] += size
            active.add(hop)
        paths.append({"src": src, "dst": dst, "bytes": size, "path": path})
    times = []
    unknown = []
    for hop in ("send", "network", "receive"):
        if hop not in active:
            continue
        bottleneck = 0.0
        for (stage, domain), size in loads.items():
            if stage != hop:
                continue
            if hop == "network":
                bandwidth = topology["node_network_bytes_s"] if domain.startswith("node:") else (
                    topology["inter_injection_bytes_s"])
            else:
                bandwidth = topology["intra_injection_bytes_s"]
            if not bandwidth:
                unknown.append(domain)
            else:
                efficiency = (topology.get("inter_link_efficiency", topology["link_efficiency"])
                              if hop == "network" else topology["link_efficiency"])
                bottleneck = max(bottleneck, size / (bandwidth * efficiency))
        latency = topology["inter_latency_s"] if hop == "network" else topology["intra_latency_s"] / 2
        # Local switch traverses source/dest concurrently; use half local cost at each hop.
        times.append(latency + (bottleneck if "network" in active else bottleneck / 2))
    return {"seconds": None if unknown else sum(times), "transfers": paths,
            "sharing_loads": [{"hop": hop, "domain": domain, "bytes": size}
                              for (hop, domain), size in sorted(loads.items())],
            "unknown_domains": sorted(set(unknown))}


def collective(group, message_bytes, placements, topology, kind="all_reduce"):
    size = len(group)
    if size <= 1 or not message_bytes:
        return {"seconds": 0.0, "message_bytes": message_bytes, "kind": kind,
                "ranks": list(group), "steps": 0, "round": None, "network_bytes": 0}
    steps = (size - 1) * (2 if kind == "all_reduce" else 1)
    # Pad the last shard rather than transmit fractional bytes.
    shard = ceil(message_bytes / size)
    round_result = transfer_round([(r, group[(i + 1) % size], shard) for i, r in enumerate(group)],
                                  placements, topology)
    return {"seconds": None if round_result["seconds"] is None else steps * round_result["seconds"],
            "message_bytes": message_bytes, "kind": kind, "ranks": list(group),
            "steps": steps, "round": round_result, "network_bytes": shard * size * steps,
            "algorithm": "ring-store-forward-v1"}


def routing_matrix(tokens_per_group, experts, top_k, *, concentrated=False):
    """Exact integer expert assignments; one request token selects distinct experts.

    Uniform cycling is a reproducible scenario, not an assertion about gate logits.
    The concentrated case chooses the same legal K experts for every token.
    """
    ep = len(tokens_per_group)
    matrix = [[0] * experts for _ in range(ep)]
    offset = 0
    for source, tokens in enumerate(tokens_per_group):
        assignments = tokens * top_k
        if concentrated:
            matrix[source][:top_k] = [tokens] * top_k
        else:
            base, remainder = divmod(assignments, experts)
            matrix[source] = [base] * experts
            for j in range(remainder):
                matrix[source][(offset + j) % experts] += 1
            offset = (offset + assignments) % experts
    return matrix


def expert_exchange(matrix, group, hidden, element_bytes, placements, topology):
    experts = len(matrix[0])
    per_owner = experts // len(group)
    dispatch = []
    for source, row in enumerate(matrix):
        for owner in range(len(group)):
            tokens = sum(row[owner * per_owner:(owner + 1) * per_owner])
            if owner != source and tokens:
                dispatch.append((group[source], group[owner], tokens * (hidden * element_bytes + 12)))
    forward = transfer_round(dispatch, placements, topology)
    # Combine returns weighted activations; preserve route identity (same metadata budget).
    backward = transfer_round([(dst, src, size) for src, dst, size in dispatch], placements, topology)
    return {"dispatch": forward, "combine": backward,
            "network_bytes": 2 * sum(size for _, _, size in dispatch),
            "seconds": None if forward["seconds"] is None or backward["seconds"] is None else (
                forward["seconds"] + backward["seconds"])}


def framework_exchange(matrix, candidate, selected, hidden, element_bytes, placements, topology):
    """Representative attention rank dispatches to each expert TP lane.

    After expert all-reduce, one owner returns each result, then the source
    broadcasts combined outputs to its attention TP lanes. No overlap credit.
    """
    width = len(matrix[0]) // candidate.ep
    forward, backward, broadcast = [], [], []
    for source, row in enumerate(matrix):
        src = next(r.rank for r in selected if r.e == source and r.t == 0)
        for owner in range(candidate.ep):
            count = sum(row[owner * width:(owner + 1) * width])
            lanes = [r.rank for r in selected if candidate.expert_owner(r) == owner]
            size = count * (hidden * element_bytes + 12)
            forward.extend((src, dst, size) for dst in lanes)
            backward.append((lanes[0], src, size))
        # top-k weighted contributions may remain separate until source combine.
        size = sum(row) * hidden * element_bytes
        broadcast.extend((src, r.rank, size) for r in selected if r.e == source and r.t)
    dispatch = transfer_round(forward, placements, topology)
    combine = transfer_round(backward, placements, topology)
    output = transfer_round(broadcast, placements, topology)
    times = [dispatch["seconds"], combine["seconds"], output["seconds"]]
    return {"dispatch": dispatch, "combine": combine, "output": output,
            "seconds": None if None in times else sum(times),
            "network_bytes": sum(size for src, dst, size in forward + backward + broadcast if src != dst),
            "algorithm": "representative-dispatch-etp-reduce-combine-broadcast-v1"}
