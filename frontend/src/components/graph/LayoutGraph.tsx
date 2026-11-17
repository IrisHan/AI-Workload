import { useEffect, useMemo, useState } from "react";
import { Background, Controls, ReactFlow, ReactFlowProvider, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";

/** Business-neutral graph layout. A changed input cancels stale ELK results. */
export function LayoutGraph({ nodes, edges, onSelect }: {
  nodes: Node[]; edges: Edge[]; onSelect: (id: string) => void;
}) {
  const [positions, setPositions] = useState<Record<string, { x: number; y: number }>>({});
  const [error, setError] = useState("");
  const topology = JSON.stringify({ nodes: nodes.map(n => n.id), edges: edges.map(e => [e.id, e.source, e.target]) });
  useEffect(() => {
    let cancelled = false;
    setPositions({}); setError("");
    const graph = JSON.parse(topology) as { nodes: string[]; edges: string[][] };
    void import("elkjs/lib/elk.bundled.js").then(async module => {
      const result = await new module.default().layout({
        id: "root", layoutOptions: { "elk.algorithm": "layered", "elk.direction": "DOWN",
          "elk.spacing.nodeNode": "24", "elk.layered.spacing.nodeNodeBetweenLayers": "32" },
        children: graph.nodes.map(id => ({ id, width: 194, height: 80 })),
        edges: graph.edges.map(([id, source, target]) => ({ id, sources: [source], targets: [target] })),
      });
      if (!cancelled) setPositions(Object.fromEntries((result.children ?? []).map(n =>
        [n.id, { x: n.x ?? 0, y: n.y ?? 0 }])));
    }).catch(() => { if (!cancelled) setError("图布局失败；可使用下方节点列表定位。"); });
    return () => { cancelled = true; };
  }, [topology]);
  const layout = useMemo(() => nodes.map(n => ({ ...n, position: positions[n.id] ?? n.position })), [nodes, positions]);
  return <div className="wc-flow">
    {error ? <p role="alert">{error}</p> : !Object.keys(positions).length ? <p role="status">布局中…</p> :
      <ReactFlowProvider><ReactFlow key={nodes.map(n => n.id).join("|")} nodes={layout} edges={edges}
        fitView fitViewOptions={{ maxZoom: 1, padding: 0.15 }} minZoom={0.05} maxZoom={2}
        nodesDraggable={false} nodesConnectable={false} onNodeClick={(_, n) => onSelect(n.id)}
        onNodeDoubleClick={(_, n) => onSelect(n.id)}>
        <Controls showInteractive={false} /><Background gap={24} color="#e3e6e8" />
      </ReactFlow></ReactFlowProvider>}
  </div>;
}
