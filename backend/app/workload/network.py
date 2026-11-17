"""Read-only network projection of frozen placement and representative transfers."""
from collections import defaultdict
from itertools import pairwise


def network_graph(summary, sources, events, *, events_complete=True):
    placement = [r for r in summary["placement"] if r["d"] == 0]
    lookup = {r["rank"]: r for r in placement}
    topology = sources["topology"]
    hardware = sources["hardware"]
    local_medium = ("NVLink / NVSwitch" if "switched_sxm" in topology["template"]
                    else "PCIe" if "pcie" in topology["template"] else "Local fabric")
    nodes = [{"id": f'rank-{r["rank"]}', "label": f'GPU {r["gpu"]}',
              "rank_refs": [r["rank"]], "category": "gpu", **r} for r in placement]
    # A saved operation can participate in several links. Keep its identity so
    # operation time is never multiplied by the number of transfers in that link.
    samples = defaultdict(list)
    expert_samples = defaultdict(list)
    for index, event in enumerate(events):
        kind = event.get("kind")
        blocks = ([event.get("round")] if event.get("round") else
                  [event[k] for k in ("dispatch", "combine", "output") if event.get(k)])
        if not blocks and event.get("transfers"):
            blocks = [event]
        transfers = [t for block in blocks if block for t in block.get("transfers", [])
                     if t["src"] in lookup and t["dst"] in lookup]
        if not transfers:
            continue
        category = "ep" if kind == "expert_dispatch_combine" else "pp" if kind == "pipeline_send" else "collective"
        by_pair = defaultdict(list)
        for transfer in transfers:
            by_pair[tuple(sorted((transfer["src"], transfer["dst"])))].append(transfer)
        for pair, paths in by_pair.items():
            sample = {"operation": index, "phase": event["phase"], "wave_id": event["wave_id"],
                      "seconds": event.get("seconds"), "kind": kind,
                      "bytes": sum(t["bytes"] for t in paths) * event.get("steps", 1),
                      "paths": paths, "ranks": event.get("ranks", [])}
            samples[(category, pair)].append(sample)
            if category == "ep":
                for rank in pair:
                    expert_samples[rank].append(sample)

    def statistics(items):
        result = {}
        for phase in ("prefill", "decode"):
            selected = [s for s in items if s["phase"] == phase]
            operations = {s["operation"]: s for s in selected}
            times = [s["seconds"] for s in operations.values()]
            result[phase] = {
                "bytes": sum(s["bytes"] for s in selected) if selected else None,
                "operation_seconds": sum(times) if times and None not in times else None,
                "operation_count": len(operations),
                "wave_ids": sorted({s["wave_id"] for s in selected}),
                "sample_paths": [p for s in selected for p in s["paths"]][:4],
            }
        return result

    def link(a, b):
        cross = a["node"] != b["node"]
        bandwidth = topology.get("inter_injection_bytes_s" if cross else "intra_injection_bytes_s")
        efficiency = topology.get("inter_link_efficiency", topology["link_efficiency"]) if cross else topology["link_efficiency"]
        return {"distance": "inter_node" if cross else "intra_node",
                "medium": "NIC / Rail fabric" if cross else local_medium,
                "bandwidth_bytes_s": bandwidth,
                "effective_bandwidth_bytes_s": bandwidth * efficiency if bandwidth else None,
                "efficiency": efficiency,
                "latency_seconds": topology["inter_latency_s" if cross else "intra_latency_s"],
                "path": ([f'Node {a["node"]} / GPU {a["gpu"]}', f'NIC {a["gpu"]}', "Rail fabric",
                          f'NIC {b["gpu"]}', f'Node {b["node"]} / GPU {b["gpu"]}'] if cross else
                         [f'GPU {a["gpu"]}', local_medium, f'GPU {b["gpu"]}'])}

    edges, seen = [], set()
    for group in summary["groups"]:
        kind = group["kind"]
        members = [r for r in group["ranks"] if r in lookup]
        if kind in ("dp", "ep") or len(members) < 2:
            continue
        identity = ("collective" if kind in ("tp", "etp") else kind, tuple(members))
        if identity in seen:
            continue
        seen.add(identity)
        pairs = list(pairwise(members))
        if kind != "pp" and len(members) > 2:
            pairs.append((members[-1], members[0]))
        for a, b in pairs:
            items = samples[(identity[0], tuple(sorted((a, b))))]
            if kind != "pp":
                items = [s for s in items if set(s["ranks"]) == set(members)]
            edges.append({"id": f'{group["id"]}:{a}:{b}', "source": f"rank-{a}", "target": f"rank-{b}",
                          "kind": kind, "group_id": group["id"], "rank_refs": [a, b],
                          **link(lookup[a], lookup[b]), "statistics": statistics(items)})

    # A stage-level exchange hub avoids O(EP²) lines; samples retain the real
    # GPU pairs, including framework dispatch that crosses expert TP lanes.
    if summary["candidate"]["ep"] > 1:
        for stage in sorted({r["p"] for r in placement}):
            members = [r for r in placement if r["p"] == stage]
            hub = f"exchange-{stage}"
            nodes.append({"id": hub, "label": f"EP exchange · PP {stage}", "category": "exchange",
                          "stage": stage, "node": members[0]["node"], "rank_refs": [r["rank"] for r in members]})
            for rank in members:
                peers = [lookup[p] for s in expert_samples[rank["rank"]] for t in s["paths"]
                         for p in (t["src"], t["dst"]) if p != rank["rank"]]
                peer = next((p for p in peers if p["node"] != rank["node"]), None)
                # Without samples, topology still describes the possible stage exchange.
                peer = peer or next((p for p in members if p["node"] != rank["node"]), members[0])
                edges.append({"id": f'{hub}:{rank["rank"]}', "source": f'rank-{rank["rank"]}', "target": hub,
                              "kind": "ep", "group_id": hub, "rank_refs": [rank["rank"]],
                              **link(rank, peer), "path": ["GPU", "EP exchange (logical hub)"],
                              "statistics": statistics(expert_samples[rank["rank"]])})
    return nodes, edges, {
        "version": "network-view-1", "replica": 0, "replica_count": summary["candidate"]["dp"],
        "gpu_count": len(placement), "local_medium": local_medium,
        "intra_bandwidth_bytes_s": topology.get("intra_injection_bytes_s"),
        "inter_bandwidth_bytes_s": topology.get("inter_injection_bytes_s"),
        "node_network_bytes_s": topology.get("node_network_bytes_s"),
        "nic_per_gpu_bytes_s": (hardware.get("node", {}).get("nic_gb_s_per_gpu") or 0) * 1e9 or None,
        "fabric": topology.get("fabric", "frozen-reference"),
        "hardware_sha256": topology.get("hardware_sha256"),
        "iteration": summary.get("metric_values", {}).get("iteration_time_ms"),
        "diagnostic_coverage": "representative_waves" if events_complete else "truncated_representative_waves",
        "timing_scope": "Participating operation time; shared by links, not additive across edges.",
    }
