import { useMemo, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { LayerFlow } from "../spec-studio/LayerFlow";
import type { GraphScope, Template } from "../spec-studio/analysis";
import type { Inspection } from "../../types/workload";
import { bytes } from "./state";
import "../spec-studio/studio.css";

export function SourceCanvas({ template, selectedId, onSelect, inspection }: {
  template: Template; selectedId: string; onSelect: (id: string) => void; inspection?: Inspection;
}) {
  const [scope, setScope] = useState<GraphScope>("all");
  const [listOpen, setListOpen] = useState(false);
  const subtitles = useMemo(() => inspection ? Object.fromEntries(template.nodes.map(n => {
    const shard = inspection.shards.find(s => s.module_id === n.module_id);
    return [n.id, !inspection.resident ? "Not resident on selected GPU" : shard ?
      `${shard.local_shape.join(" × ")} · ${bytes(shard.bytes)}` : "No independent weights"];
  })) : undefined, [template, inspection]);
  return <div className="spec-studio wc-source-canvas">
    {inspection && <div className="wc-graph-context">Layer {inspection.scope.layer} · GPU {inspection.placement.rank}
      {" · "}{inspection.scope.phase} · TP {String(inspection.explanation?.scope.tp ?? "—")}
      {" · "}{inspection.local_tokens} local tokens · Representative Batch</div>}
    <div className="scope-bar">{(["all", "attention", "ffn"] as GraphScope[]).map(s => <button key={s}
      aria-pressed={scope === s} onClick={() => setScope(s)}>{s === "all" ? "Full Layer" : s === "ffn" ? "FFN" : "Attention"}</button>)}
      <span>{inspection ? `GPU ${inspection.placement.rank} · Local Shards` : "Source Parameters"}</span></div>
    <ReactFlowProvider><LayerFlow template={template} selectedId={selectedId} onSelect={onSelect}
      scope={scope} multiplier={1} nodeSubtitles={subtitles} initialZoomFloor={0.55} focusSelection /></ReactFlowProvider>
    <details open={listOpen} onToggle={e => setListOpen(e.currentTarget.open)}><summary>Operator List</summary><div className="wc-node-list">
      {template.nodes.map(n => <button key={n.id} onClick={() => { onSelect(n.id); setListOpen(false); }}>{n.label}</button>)}
    </div></details>
  </div>;
}
