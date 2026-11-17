import type { Components, Memory, RankDetail } from "../../types/workload";
import { CalculationTrace } from "./CalculationTrace";
import { bytes, number } from "./state";

export const componentNames: Record<keyof Components, string> = {
  weight: "Weights", kv: "KV Cache", activation: "Activations", communication: "Communication Buffer",
  runtime: "Runtime Buffer", fragmentation: "Fragmentation", safety: "Safety Reserve",
};
export function MemoryRows({ components }: { components: Components }) {
  return <dl className="wc-values">{Object.entries(componentNames).map(([key, name]) =>
    <div key={key}><dt>{name}</dt><dd>{bytes(components[key as keyof Components])}</dd></div>)}</dl>;
}
export function MemorySummary({ memory }: { memory: Memory | RankDetail }) {
  const e = memory.explanation;
  return <section className="wc-memory-summary" aria-label="GPU Memory">
    <h3>GPU Memory · Admission Bound</h3>
    <p className="wc-memory-total"><strong>{bytes(memory.peak_bytes)} / {bytes(memory.capacity_bytes)}</strong>
      <span>{e ? e.capacity_pass ? "PASS" : "FAIL" : memory.free_bytes >= 0 ? "PASS" : "FAIL"}</span></p>
    {e && <><meter min={0} max={1} value={e.utilization} aria-label="GPU Memory Utilization" />
      <p className="muted">Hardware capacity {number(e.capacity_gb)} GB (SI) · {number(e.utilization * 100)}% reserved</p></>}
    <p>Free <strong>{bytes(memory.free_bytes)}</strong> · Safety 已计入 Peak 一次。</p>
    <details><summary>Why this fits · Memory Breakdown</summary>
      <p className="muted">七项准入预留相加得到 Peak；Buffer、Fragmentation、Safety 为 GPU 共享项，不分摊给算子。</p>
      {Object.entries(componentNames).map(([key, name]) => <div className="wc-memory-component" key={key}>
        <div><strong>{name}</strong><span className="wc-inline-value">{bytes(memory.components[key as keyof Components])}
          <CalculationTrace trace={e?.traces[key]} label={name} inline /></span></div>
      </div>)}
      {["peak", "free", "utilization"].map(key => <CalculationTrace key={key} trace={e?.traces[key]} />)}
    </details>
  </section>;
}
