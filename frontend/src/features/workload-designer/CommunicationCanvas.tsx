import { useEffect, useMemo, useRef, useState } from "react";
import { BaseEdge, Controls, EdgeLabelRenderer, Handle, MarkerType, Position, ReactFlow, getSmoothStepPath,
  type EdgeProps, type Node, type NodeProps } from "@xyflow/react";
import type { Graph, GraphNode } from "../../types/workload";
import { bytes, number } from "./state";
import "@xyflow/react/dist/style.css";

const colors: Record<string, string> = { tp: "#3672a6", etp: "#3672a6", pp: "#825da1", ep: "#b97532" };
const bandwidth = (value?: number | null) => value == null ? "—" : `${number(value / 1e9)} GB/s`;
function Device({ data }: NodeProps) {
  return <><Handle type="target" position={Position.Left} id="left" />
    <Handle type="source" position={Position.Right} id="right" />
    <Handle type="target" position={Position.Top} id="top" />
    <Handle type="source" position={Position.Top} id="sendTop" />
    <Handle type="source" position={Position.Bottom} id="bottom" />
    {data.label as React.ReactNode}</>;
}
function Machine({ data }: NodeProps) {
  return <div className="wc-network-machine-title">{data.label as React.ReactNode}</div>;
}
function NetworkLink(props: EdgeProps) {
  const { sourceX: sx, sourceY: sy, targetX: tx, targetY: ty, data } = props;
  let [path, x, y] = getSmoothStepPath({ ...props, offset: 8, borderRadius: 4 });
  if (data?.cross) {
    const bus = Number(data.busX);
    path = `M ${sx} ${sy} V ${sy + 24} H ${bus} V ${ty - 24} H ${tx} V ${ty}`;
    x = sx; y = sy + 15;
  } else if (data?.overhead) {
    y = Math.min(sy, ty) - 22 - Number(data.lane ?? 0) % 2 * 13;
    x = (sx + tx) / 2;
    path = `M ${sx} ${sy} V ${y} H ${tx} V ${ty}`;
  } else if (data?.wrap) {
    y = Math.max(sy, ty) + 72; x = (sx + tx) / 2;
    path = `M ${sx} ${sy} H ${sx + 8} V ${y} H ${tx - 8} V ${ty} H ${tx}`;
  }
  return <><BaseEdge id={props.id} path={path} style={props.style} markerEnd={props.markerEnd} interactionWidth={18} />
    <EdgeLabelRenderer><button className="wc-network-edge-label nodrag nopan"
      aria-label={`Inspect ${props.label} ${props.source} to ${props.target}`}
      onClick={() => (data?.inspect as (() => void) | undefined)?.()} style={{
      transform: `translate(-50%, -50%) translate(${x}px,${y}px)`, color: props.style?.stroke,
    }}>{props.label}</button></EdgeLabelRenderer></>;
}
const nodeTypes = { device: Device, machine: Machine };
const edgeTypes = { network: NetworkLink };

export function CommunicationCanvas({ graph, selectedRank, phase, onSelect }: {
  graph: Graph; selectedRank: number; phase: string; onSelect: (node: GraphNode) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(700);
  const [filter, setFilter] = useState("all");
  const [edgeId, setEdgeId] = useState("");
  const [related, setRelated] = useState(false);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const network = graph.network;
  const selected = graph.edges.find(e => e.id === edgeId);
  const stat = selected?.statistics?.[phase];
  const layout = useMemo(() => {
    const devices = graph.nodes.filter(n => n.category === "gpu");
    const machines = [...new Set(devices.map(n => n.node!))].sort((a, b) => a - b);
    const cols = Math.min(8, Math.max(1, ...machines.map(m => devices.filter(n => n.node === m).length)));
    const rackWidth = Math.max(320, cols * 100 + 40);
    let y = 18;
    const nodes: Node[] = [];
    const hubY: Record<string, number> = {};
    for (const machine of machines) {
      const local = devices.filter(n => n.node === machine).sort((a, b) => a.gpu! - b.gpu!);
      const hubs = graph.nodes.filter(n => n.category === "exchange" && n.node === machine);
      const height = Math.ceil(local.length / 8) * 144 + 92 + hubs.length * 34;
      hubs.forEach((hub, i) => { hubY[hub.id] = y + Math.ceil(local.length / 8) * 144 + 72 + i * 34; });
      nodes.push({ id: `node-${machine}`, position: { x: 24, y }, type: "machine",
        data: { label: `Node ${machine} · ${network?.local_medium} · ${bandwidth(network?.intra_bandwidth_bytes_s)}` },
        className: "wc-network-machine", selectable: false, style: { width: rackWidth, height } });
      local.forEach((n, index) => nodes.push({
        id: n.id, parentId: `node-${machine}`, extent: "parent", type: "device",
        position: { x: 20 + index % 8 * 100, y: 80 + Math.floor(index / 8) * 144 },
        data: { label: <><strong>{n.label}</strong><span>Rank {n.rank}</span><span>TP {n.t} · PP {n.p}</span><span title="Attention group">AG {n.e}</span></> },
        ariaLabel: `${n.label} Rank ${n.rank} Node ${machine}`, className: "wc-network-gpu",
        style: { width: 84, height: 88, borderColor: n.rank === selectedRank ? "#176849" : "#c5d1c9",
          borderWidth: n.rank === selectedRank ? 2 : 1 },
      }));
      y += height + 76;
    }
    for (const hub of graph.nodes.filter(n => n.category === "exchange")) {
      nodes.push({ id: hub.id, position: { x: rackWidth / 2 - 66, y: hubY[hub.id] },
        type: "device", data: { label: hub.label }, className: "wc-network-hub",
        style: { width: 180, height: 28 } });
    }
    if ((network?.replica_count ?? 1) > 1) {
      nodes.push({ id: "other-replicas", position: { x: 24, y }, data: {
        label: `Other DP × ${network!.replica_count - 1} · Independent inference replicas` },
      className: "wc-network-replicas", selectable: false, style: { width: rackWidth, height: 46 } });
      y += 70;
    }
    return { nodes, width: rackWidth + 72, height: y };
  }, [graph, network, selectedRank]);
  const zoom = Math.min(1, Math.max(.65, (width - 12) / layout.width));
  const edges = useMemo(() => graph.edges.filter(e => (filter === "all" || e.kind === filter)
    && (!related || e.rank_refs?.includes(selectedRank))).map(e => ({
    id: e.id, source: e.source, target: e.target, type: "network",
    data: { overhead: e.kind === "pp" && e.distance === "intra_node", cross: e.distance === "inter_node",
      busX: layout.width - 8 - Number(e.rank_refs?.[0] ?? 0) % 8 * 3,
      inspect: () => setEdgeId(e.id),
      lane: e.rank_refs?.[0], wrap: (e.kind === "tp" || e.kind === "etp") && e.distance === "intra_node"
        && Number(e.source.split("-")[1]) > Number(e.target.split("-")[1]) },
    sourceHandle: e.distance === "inter_node" ? "bottom" : e.kind === "tp" || e.kind === "etp" ? "right"
      : e.kind === "pp" && e.distance === "intra_node" ? "sendTop" : "bottom",
    targetHandle: e.distance === "inter_node" ? "top" : e.kind === "tp" || e.kind === "etp" ? "left" : "top",
    label: `${e.kind?.toUpperCase()}${e.distance === "inter_node" ? ` · NIC ${bandwidth(e.bandwidth_bytes_s)}` : ""}`,
    markerEnd: e.kind === "pp" ? { type: MarkerType.ArrowClosed, color: colors.pp } : undefined,
    interactionWidth: 18,
    style: { stroke: colors[e.kind ?? "tp"], strokeWidth: e.id === edgeId ? 3 : 1.5,
      opacity: !related && e.rank_refs?.includes(selectedRank) === false ? .42 : 1 },
    labelStyle: { fontSize: 10, fill: colors[e.kind ?? "tp"] },
    labelBgStyle: { fill: "#fff" }, zIndex: e.id === edgeId ? 4 : 2,
  })), [graph, filter, related, selectedRank, edgeId, layout.width]);
  return <div className="wc-network" ref={container}>
    <div className="wc-toolbar wc-network-toolbar">
      <span>DP 0 / {network?.replica_count ?? 1}</span>
      <label>Links<select aria-label="Communication type" value={filter} onChange={e => setFilter(e.target.value)}>
        {["all", ...["tp", "etp", "pp", "ep"].filter(k => graph.edges.some(e => e.kind === k))]
          .map(k => <option key={k} value={k}>{k.toUpperCase()}</option>)}
      </select></label>
      <label className="wc-network-related"><input type="checkbox" checked={related} onChange={e => setRelated(e.target.checked)} />Selected GPU</label>
    </div>
    <div className="wc-network-scroll">
      <div className="wc-network-flow" style={{ height: Math.max(320, layout.height * zoom + 24), minWidth: layout.width * zoom }}>
        <ReactFlow key={`${graph.plan_id}:${Math.round(width)}`} nodes={layout.nodes} edges={edges}
          nodeTypes={nodeTypes} edgeTypes={edgeTypes}
          defaultViewport={{ x: 4, y: 4, zoom }} minZoom={.1} maxZoom={2}
          nodesDraggable={false} nodesConnectable={false} deleteKeyCode={null} zoomOnScroll={false} preventScrolling={false}
          onNodeClick={(_, node) => { const source = graph.nodes.find(n => n.id === node.id); if (source?.rank != null) onSelect(source); }}
          onEdgeClick={(_, edge) => setEdgeId(edge.id)}>
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
    {selected && <section className="wc-network-detail" aria-label="Link details">
      <div><strong>{selected.kind?.toUpperCase()} · {selected.group_id}</strong>
        <button aria-label="Close link details" onClick={() => setEdgeId("")}>×</button></div>
      <dl>
        <dt>Path</dt><dd>{selected.path?.join(" → ")}</dd>
        <dt>Distance</dt><dd>{selected.distance === "inter_node" ? "Inter-node · NIC" : "Intra-node"} · {selected.medium}</dd>
        <dt>Bandwidth / GPU</dt><dd>{bandwidth(selected.bandwidth_bytes_s)} → {bandwidth(selected.effective_bandwidth_bytes_s)} effective</dd>
        {selected.distance === "inter_node" && <><dt>Node uplink</dt><dd>{bandwidth(network?.node_network_bytes_s)} shared</dd></>}
        <dt>Link startup</dt><dd>{number((selected.latency_seconds ?? 0) * 1e6)} μs</dd>
        <dt>{phase === "decode" ? "Decode" : "Prefill"} transferred</dt><dd>{stat?.bytes == null ? "No saved sample" : bytes(stat.bytes)}</dd>
        <dt>Operation time</dt><dd>{stat?.operation_seconds == null ? "No saved timing" : `${number(stat.operation_seconds * 1e3)} ms`}
          {stat?.operation_count ? ` · ${stat.operation_count} operations` : ""}</dd>
      </dl>
      <details><summary>Timing & paths</summary>
        <p>{network?.timing_scope} {network?.diagnostic_coverage}. Wave {stat?.wave_ids.join(", ") || "—"}.</p>
        <p>{network?.fabric} · Network {bandwidth(network?.inter_bandwidth_bytes_s)} / GPU · NIC spec {bandwidth(network?.nic_per_gpu_bytes_s)}</p>
        {stat?.sample_paths.map((p, i) => <p key={i}>R{p.src} → R{p.dst} · {bytes(p.bytes)} · {p.path.join(" → ")}</p>)}
      </details>
    </section>}
    {graph.truncated && <p role="status">Network view truncated</p>}
  </div>;
}
