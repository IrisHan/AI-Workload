import { useState } from "react";
import type { GraphNode, Inspection, RankDetail, Candidate } from "../../types/workload";
import { bytes, number } from "./state";
import { CalculationTrace, Facts } from "./CalculationTrace";
import { MemorySummary } from "./MemorySummary";
import { RawDetails } from "./Evidence";
export { MemoryRows, componentNames } from "./MemorySummary";

export function Inspector({ node, inspection, rank, pending, error, stale, candidate }: {
  node?: GraphNode; inspection?: Inspection; rank?: RankDetail; pending: boolean; error?: string; stale: boolean;
  candidate?: Candidate;
}) {
  const [step, setStep] = useState(5);
  const explained = node ? inspection?.explanation?.nodes[node.source_node_id ?? node.id] : undefined;
  const live = inspection?.explanation?.steps.find(s => s.step === step);
  const allocations = inspection?.allocations.filter(a => live?.allocation_ids.includes(a.id)) ?? [];
  return <aside className="wc-inspector" aria-label="Selection details">
    <header><h2>{node?.label ?? "GPU Details"}</h2><span>GPU {rank?.rank ?? inspection?.placement.rank ?? "—"}</span></header>
    {stale && <p className="wc-notice">历史结果：输入已修改，解释仍绑定冻结运行。</p>}
    {error && <p role="alert" className="bad">{error}</p>}
    {pending && <p role="status">加载分片…</p>}
    {inspection && <section>
      <h3>Context</h3>
      <p>Layer {inspection.scope.layer} · {inspection.scope.phase} · GPU {inspection.placement.rank}</p>
      {candidate && <p>TP {candidate.tp} / PP {candidate.pp} / EP {candidate.ep} / DP {candidate.dp}</p>}
      <p className="muted">Representative Batch · 配额与 token budget 下的代表批，不是 Online 首个 wave。</p>
      {!inspection.resident ? <p>此 GPU 不持有所选层。</p> : <>
        <dl className="wc-values">
          <div><dt>Local Tokens <span className="wc-source-tag">Derived</span></dt><dd>{number(inspection.local_tokens)}</dd></div>
          <div><dt>Resident Requests</dt><dd>{inspection.resident_request_quota}</dd></div>
        </dl>
        {explained ? <>
          <h3>Tensor Shape</h3>
          {explained.shapes.length ? <Facts items={explained.shapes} /> : <p className="muted">此节点没有独立 GEMM 张量形状。</p>}
          {explained.full_shape && <p>Full Weight {explained.full_shape.join(" × ")} → Local {explained.local_shape?.join(" × ")}</p>}
          {explained.expert_ranges.map(([start, stop], i) => <p key={i}>Experts [{start}, {stop})</p>)}
          <h3>Memory</h3><dl className="wc-values">
            <div><dt>Parameters</dt><dd>{number(explained.parameters)}</dd></div>
            <div><dt>Weight</dt><dd>{bytes(explained.weight_bytes)}</dd></div>
          </dl>
          <h3>Compute & Latency</h3><p className="muted">{explained.attribution}</p>
          {explained.operators.map((o, i) => <div className="wc-operator-time" key={`${o.id}-${i}`}>
            <strong>{o.id} · {number(o.seconds == null ? null : o.seconds * 1000)} ms</strong>
            <dl className="wc-values">
              <div><dt>Matrix FLOPs / Scalar ops</dt><dd>{number(o.flops)} / {number(o.scalar_equivalent_ops)}</dd></div>
              <div><dt>HBM Traffic</dt><dd>{bytes(o.hbm_bytes)}</dd></div>
              <div><dt>Selected</dt><dd>{o.bottleneck === "hbm" ? "MEMORY BOUND" : o.bottleneck === "compute" ? "COMPUTE BOUND" : "Unavailable"}</dd></div>
              {[["Compute", o.compute_seconds], ["Memory", o.hbm_seconds], ["Launch", o.launch_seconds]].map(([label, seconds]) =>
                <div key={String(label)}><dt>{label}</dt><dd>{number(seconds == null ? null : Number(seconds) * 1000)} ms</dd></div>)}
            </dl>
            <CalculationTrace trace={explained.traces[i + 1]} />
          </div>)}
        </> : inspection.explanation ? <p className="muted">选择算子查看 Shape、Memory 和 Compute。</p> : null}
        <h3>Lifetime</h3>
        <p className="muted">不同 Kernel Step 有不同 live tensors；它们不会全部长期同时驻留。Layer Peak = max(step live bytes)。</p>
        <label>Kernel Step {step}<input aria-label="Kernel Step" type="range" min="0" max="13" value={step}
          onChange={e => setStep(Number(e.target.value))} /></label>
        <dl className="wc-values"><div><dt>Live Activations</dt><dd>{bytes(live?.bytes)}</dd></div>
          <div><dt>Layer Peak</dt><dd>{bytes(inspection.activation_peak_bytes)}</dd></div></dl>
        <details><summary>Live Tensors · Step {step}</summary>
          {allocations.map(a => <div className="wc-allocation" key={a.id}><span>{a.id.split("/").at(-1)}</span><span>{bytes(a.bytes)}</span></div>)}
        </details>
        <h3>Calculation</h3>
        {explained && <CalculationTrace trace={explained.traces[0]} />}
        <CalculationTrace trace={inspection.explanation?.lifetime} />
        <details><summary>Layer KV Cache</summary>
          <dl className="wc-values"><div><dt>Allocated</dt><dd>{bytes(inspection.cache.total_bytes)}</dd></div>
            <div><dt>Page Metadata</dt><dd>{bytes(inspection.cache.metadata_bytes)}</dd></div>
            <div><dt>Page Tail Waste</dt><dd>{bytes(inspection.cache.tail_waste_bytes)}</dd></div></dl>
        </details>
      </>}
    </section>}
    {rank && <><MemorySummary memory={rank} />
      <RawDetails title="GPU Weight Ledger">{() => rank.weights.map(s =>
        <div className="wc-shard" key={s.storage_id}><strong>{s.module_id}</strong>
          <span>{s.local_shape.join(" × ")} · {bytes(s.bytes)}</span><code>{s.storage_id}</code></div>)}</RawDetails>
    </>}
    {!inspection && !rank && !pending && <p className="muted">运行方案后，点击结构中的算子，查看 GPU 分片和计算解释。</p>}
  </aside>;
}
