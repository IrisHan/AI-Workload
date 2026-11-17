import { useEffect, useMemo, useRef, useState } from "react";
import { Background, Handle, Position, ReactFlow, MarkerType, useReactFlow,
  type Node, type NodeProps, type Edge } from "@xyflow/react";
import type { ElkNode } from "elkjs";
import { Maximize, Minus, Plus } from "lucide-react";
import "@xyflow/react/dist/style.css";
import { categoryTone, moeLayout, nodeCluster, type ClusterId, type MoeLayout, type OpNode, type Template, type GraphScope } from "./analysis";
import { fmtParams } from "./format";
import { nodeMeaning } from "./NodeDetails";

type BlockData = { label: string; category: string; params: string; selected: boolean; cluster?: string };
type ClusterData = { label: string };
const layoutEngine = () => import("elkjs/lib/elk.bundled.js").then(module => new module.default());
const ATTN_ORDER = ["q_path", "kv_path", "score", "out"] as const;
const FFN_ORDER = ["router", "latent", "routed", "shared"] as const;
const VISION_ORDER = ["patch", "encoder", "projector"] as const;
const CLUSTER_ORDER = [...ATTN_ORDER, ...FFN_ORDER, "merge", ...VISION_ORDER] as const;

function clusterTitle(id: ClusterId, moe: MoeLayout | null): string {
  if (id === "q_path") return "Q path";
  if (id === "kv_path") return "KV / compressor";
  if (id === "score") return "Score / recurrent";
  if (id === "out") return "Output proj";
  if (id === "router") return `Router / top-k ${moe?.top_k ?? "?"}/${moe?.n_routed ?? "?"}`;
  if (id === "latent") return moe?.latent_dim ? `Latent down/up · ${moe.latent_dim}` : "Latent";
  if (id === "routed") return `Routed experts ×${moe?.n_routed ?? "?"}`;
  if (id === "shared")
    return moe && moe.n_shared > 1 ? `Shared expert · ${moe.n_shared}×W` : "Shared expert ×1";
  if (id === "patch") return "Patch embed";
  if (id === "encoder") return "Vision encoder";
  if (id === "projector") return "Merger / projector";
  return "Expert sum";
}

function BlockNode({ data }: NodeProps<Node<BlockData>>) {
  const cluster = data.cluster ? `cluster-${data.cluster}` : "";
  return <div className={`op-node ${categoryTone(data.category)} ${cluster} ${data.selected ? "selected" : ""}`}>
    <Handle type="target" position={Position.Top} />
    <span className="op-category">{data.cluster ? `${data.category} · ${data.cluster}` : data.category}</span>
    <strong>{data.label}</strong>
    <span className="mono">{data.params}</span>
    <Handle type="source" position={Position.Bottom} />
  </div>;
}

function ClusterFrame({ data }: NodeProps<Node<ClusterData>>) {
  return <div className="moe-cluster-frame"><span>{data.label}</span></div>;
}

const nodeTypes = { block: BlockNode, cluster: ClusterFrame };

function paramLabel(module: Template["modules"][number], multiplier: number, moe: MoeLayout | null): string {
  if (module.shared_with) return "共享权重";
  const total = module.params * multiplier;
  const count = module.count ?? 1;
  if (count > 1)
    return `${count} × ${fmtParams(module.params / count)} = ${fmtParams(total)}`;
  if (module.id.includes("shared_expert.") && moe && moe.n_shared > 1)
    return `${moe.n_shared}×W · ${fmtParams(total)}`;
  return `${fmtParams(total)} params`;
}

type ElkChild = ElkNode;

function collectLayout(node: ElkNode, ox = 0, oy = 0) {
  const x = ox + (node.x ?? 0);
  const y = oy + (node.y ?? 0);
  const positions: Record<string, { x: number; y: number }> = {};
  const frames: Record<string, { x: number; y: number; width: number; height: number }> = {};
  if (typeof node.id === "string" && node.id.startsWith("cluster:"))
    frames[node.id] = { x, y, width: node.width ?? 240, height: node.height ?? 120 };
  if (node.children?.length) {
    for (const child of node.children) {
      const inner = collectLayout(child, x, y);
      Object.assign(positions, inner.positions);
      Object.assign(frames, inner.frames);
    }
  } else if (node.id && !node.id.startsWith("cluster:"))
    positions[node.id] = { x, y };
  return { positions, frames };
}

function leaf(id: string, wide: boolean) {
  return { id, width: wide ? 228 : 194, height: wide ? 88 : 80 };
}

function elkGraph(template: Template, visible: OpNode[], edges: Template["edges"], moe: MoeLayout | null): ElkNode {
  const wide = Boolean(moe) || template.attn_kind !== "GQA" || template.role === "vision";
  const base = {
    "elk.algorithm": "layered",
    "elk.direction": "DOWN",
    "elk.spacing.nodeNode": "36",
    "elk.layered.spacing.nodeNodeBetweenLayers": "40",
  };
  const elkEdges = edges.map((edge, i) => ({ id: `e${i}`, sources: [edge.source], targets: [edge.target] }));
  const buckets: Partial<Record<ClusterId, OpNode[]>> = {};
  const loose: OpNode[] = [];
  for (const node of visible) {
    const cluster = nodeCluster(node);
    if (cluster) (buckets[cluster] ??= []).push(node);
    else loose.push(node);
  }
  const clusterNode = (id: ClusterId): ElkNode | null => buckets[id]?.length ? {
    id: `cluster:${id}`,
    layoutOptions: {
      "elk.algorithm": "layered", "elk.direction": "DOWN",
      "elk.padding": "[top=36,left=16,bottom=16,right=16]",
      "elk.spacing.nodeNode": "22",
      "elk.layered.spacing.nodeNodeBetweenLayers": "28",
    },
    children: buckets[id]!.map(node => leaf(node.id, true)),
  } : null;
  const row = (id: string, ids: readonly ClusterId[]): ElkNode | null => {
    const cols = ids.map(clusterNode).filter((item): item is ElkNode => item != null);
    if (!cols.length) return null;
    return {
      id,
      layoutOptions: {
        "elk.algorithm": "layered", "elk.direction": "RIGHT",
        "elk.padding": "[top=8,left=8,bottom=8,right=8]",
        "elk.spacing.nodeNode": "32",
      },
      children: cols,
    };
  };
  const children: ElkNode[] = [...loose.map(node => leaf(node.id, wide))];
  const attn = row("cluster:attn", ATTN_ORDER);
  const ffn = row("cluster:moe", FFN_ORDER);
  const vision = row("cluster:vision", VISION_ORDER);
  const merge = clusterNode("merge");
  if (attn) children.push(attn);
  if (ffn) children.push(ffn);
  if (vision) children.push(vision);
  if (merge) children.push(merge);
  if (!attn && !ffn && !vision && !merge)
    return { id: template.id, layoutOptions: base, children: visible.map(node => leaf(node.id, false)), edges: elkEdges };
  return {
    id: template.id,
    layoutOptions: { ...base, "elk.hierarchyHandling": "INCLUDE_CHILDREN" },
    children, edges: elkEdges,
  };
}

export function LayerFlow({ template, scope, selectedId, multiplier, onSelect, nodeSubtitles, initialZoomFloor = 0.1, focusSelection = false }: {
  template: Template; scope: GraphScope; selectedId: string; multiplier: number;
  onSelect: (id: string) => void; nodeSubtitles?: Record<string, string>; initialZoomFloor?: number; focusSelection?: boolean;
}) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [frames, setFrames] = useState<Record<string, { x: number; y: number; width: number; height: number }>>({});
  const [error, setError] = useState("");
  const { fitView, zoomIn, zoomOut, setCenter, getZoom } = useReactFlow();
  const wrapRef = useRef<HTMLDivElement>(null);
  const focusId = focusSelection ? selectedId || template.nodes[0]?.id || "" : "";
  const moe = useMemo(() => moeLayout(template), [template]);
  const visible = useMemo(() => {
    if (scope === "all") return template.nodes;
    return template.nodes.filter(node => categoryTone(node.category) === scope);
  }, [template, scope]);
  const visibleEdges = useMemo(() => {
    const ids = new Set(visible.map(node => node.id));
    return template.edges.filter(edge => ids.has(edge.source) && ids.has(edge.target));
  }, [template, visible]);

  useEffect(() => {
    let canceled = false;
    setPositions({});
    setFrames({});
    setError("");
    const graph = elkGraph(template, visible, visibleEdges, moe);
    void layoutEngine().then(elk => elk.layout(graph)).then(result => {
      if (canceled) return;
      const layout = collectLayout(result);
      setPositions(layout.positions);
      setFrames(layout.frames);
    }).catch(() => {
      if (canceled) return;
      if (moe || template.role === "vision" || template.attn_kind === "VISION") {
        const fallback: ElkNode = {
          id: template.id,
          layoutOptions: { "elk.algorithm": "layered", "elk.direction": "DOWN",
            "elk.spacing.nodeNode": "36", "elk.layered.spacing.nodeNodeBetweenLayers": "40" },
          children: visible.map(node => leaf(node.id, true)),
          edges: visibleEdges.map((edge, i) => ({ id: `e${i}`, sources: [edge.source], targets: [edge.target] })),
        };
        void layoutEngine().then(elk => elk.layout(fallback)).then(result => {
          if (canceled) return;
          const layout = collectLayout(result);
          setPositions(layout.positions);
          setFrames({});
        }).catch(() => { if (!canceled) setError("算子布局失败"); });
        return;
      }
      if (!canceled) setError("算子布局失败");
    });
    return () => { canceled = true; };
  }, [template, visible, visibleEdges, moe]);

  const nodes: Node[] = useMemo(() => {
    const clusterNodes: Node<ClusterData>[] = Object.entries(frames)
      .filter(([id]) => !id.endsWith(":moe") && !id.endsWith(":attn") && !id.endsWith(":vision") && id.startsWith("cluster:"))
      .map(([id, box]) => {
        const key = id.slice("cluster:".length) as ClusterId;
        return {
          id, type: "cluster", position: { x: box.x, y: box.y }, draggable: false, selectable: false,
          style: { width: box.width, height: box.height, zIndex: 0 },
          data: { label: moe && CLUSTER_ORDER.includes(key) ? clusterTitle(key, moe) : clusterTitle(key, moe) },
        };
      });
    const blocks: Node<BlockData>[] = visible.map(node => {
      const module = template.modules.find(item => item.id === node.module_id);
      const cluster = nodeCluster(node) ?? undefined;
      return {
        id: node.id, type: "block", position: positions[node.id] ?? { x: 0, y: 0 }, zIndex: 1,
        data: {
          label: node.label, category: node.category, cluster,
          params: nodeSubtitles?.[node.id] ?? (module
            ? paramLabel(module, multiplier, moe)
            : nodeMeaning(node.id) ? `${nodeMeaning(node.id)!.kind} · 无独立权重` : "未提供参数账本"),
          selected: selectedId === node.id || selectedId === node.module_id,
        },
        style: moe || template.role === "vision" || ATTN_ORDER.includes(cluster as typeof ATTN_ORDER[number])
          ? { width: 228, height: 88 } : undefined,
        ariaLabel: node.label,
      };
    });
    return [...clusterNodes, ...blocks];
  }, [template, visible, positions, frames, multiplier, selectedId, moe, nodeSubtitles]);
  const edges: Edge[] = visibleEdges.map((edge, i) => ({
    id: `e${i}`, source: edge.source, target: edge.target,
    type: "smoothstep", label: edge.label ?? undefined,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
    style: { stroke: "#87958e", strokeWidth: 1.4 },
  }));

  useEffect(() => {
    if (!Object.keys(positions).length) return;
    const node = wrapRef.current;
    if (!node) return;
    let frame = 0;
    const fit = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        // Mobile detail panes keep the graph mounted while its container is hidden.
        if (node.clientWidth <= 0 || node.clientHeight <= 0) return;
        const selected = positions[focusId];
        if (focusSelection && selected) {
          const zoom = getZoom();
          void setCenter(selected.x + 114, selected.y + 44,
            { zoom: Math.max(initialZoomFloor, Number.isFinite(zoom) ? zoom : 1), duration: 0 });
        } else void fitView({ padding: 0.12, minZoom: initialZoomFloor, maxZoom: 1 });
      });
    };
    const observer = new ResizeObserver(fit);
    observer.observe(node);
    fit();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [positions, fitView, initialZoomFloor, focusSelection, focusId, getZoom, setCenter]);

  return <div className="flow-wrap" data-testid="operator-graph" ref={wrapRef}>
    <div className="graph-tools" role="toolbar" aria-label="画布工具">
      <button title="放大" aria-label="放大" onClick={() => void zoomIn()}><Plus size={17} /></button>
      <button title="缩小" aria-label="缩小" onClick={() => void zoomOut()}><Minus size={17} /></button>
      <button title="适配画布" aria-label="适配画布" onClick={() => void fitView({ padding: 0.12, maxZoom: 1 })}><Maximize size={17} /></button>
    </div>
    {error ? <p role="alert">{error}</p> : !Object.keys(positions).length
      ? <div className="graph-loading" role="status">布局中…</div>
      : <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} minZoom={0.1} maxZoom={2}
        nodesDraggable={false} nodesConnectable={false}
        onNodeClick={(_, node) => { if (node.type !== "cluster") onSelect(node.id); }}
        onNodeDoubleClick={(_, node) => { if (node.type !== "cluster") onSelect(node.id); }}
        fitView={!focusSelection} fitViewOptions={{ padding: 0.12, minZoom: initialZoomFloor }}>
        <Background color="#d8dedb" gap={24} />
      </ReactFlow>}
  </div>;
}
