import { useEffect, useRef, useState } from "react";
import { Play } from "lucide-react";
import type { AnalysisRequest } from "./queries";
import { useHardware } from "../queries";
import { fmtFlop, fmtInt } from "./format";

interface Cost { flops: number; traffic_bytes: number; total_memory_bytes: number;
  memory_fits: boolean; latency_bound_ms: number; calibrated: boolean;
  formula_version?: string;
  rows: { group_id: string; operation: string; flops: number; traffic_bytes: number; latency_bound_ms: number }[];
  assumptions: string[]; }

export function CostView({ analysisId, analysisRequest }: { analysisId: string; analysisRequest: AnalysisRequest }) {
  const hardware = useHardware();
  const [hardwareId, setHardwareId] = useState("");
  const selectedHardware = hardwareId || hardware.data?.[0]?.id || "";
  const [phase, setPhase] = useState("decode");
  const [batch, setBatch] = useState(1);
  const [length, setLength] = useState(128);
  const [dtype, setDtype] = useState("bf16");
  const [result, setResult] = useState<Cost | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  function invalidate() {
    request.current?.abort(); setResult(null); setError(""); setLoading(false);
  }
  return <main className="full-view" tabIndex={0} aria-label="算子成本内容">
    <h2>算子成本 · 理论估算 / 未实测校准</h2>
    <form className="cost-form" onSubmit={event => {
      event.preventDefault(); request.current?.abort();
      const controller = new AbortController(); request.current = controller;
      setResult(null); setError(""); setLoading(true);
      void analysisRequest<Cost>(analysisId, "cost", controller.signal, {
        phase, batch_size: batch, sequence_length: length, context_length: length, dtype,
        hardware_id: selectedHardware,
      }).then(value => { if (!controller.signal.aborted) setResult(value); }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); })
        .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    }}>
      <label>阶段<select aria-label="阶段" value={phase} onChange={event => { invalidate(); setPhase(event.target.value); }}>
        <option value="decode">Decode</option><option value="prefill">Prefill</option></select></label>
      <label>Batch<input type="number" min={1} max={4096} value={batch} onChange={event => { invalidate(); setBatch(Number(event.target.value)); }} /></label>
      <label>Token 长度<input type="number" min={1} max={16777216} value={length} onChange={event => { invalidate(); setLength(Number(event.target.value)); }} /></label>
      <label>计算 / KV dtype<select value={dtype} onChange={event => { invalidate(); setDtype(event.target.value); }}>
        <option>bf16</option><option>fp32</option></select></label>
      <label>Hardware<select aria-label="成本硬件" value={selectedHardware}
        onChange={event => { invalidate(); setHardwareId(event.target.value); }}>
        {hardware.data?.filter(gpu => !gpu.legacy).map(gpu => <option key={gpu.id} value={gpu.id}
          disabled={!gpu.planner_precisions.includes(dtype)}>{gpu.name}{!gpu.planner_precisions.includes(dtype) ? " (unverified)" : ""}</option>)}
      </select></label>
      <button disabled={loading || !selectedHardware} className="primary"><Play size={15} />计算</button>
    </form>
    {hardware.isError && <p role="alert">{hardware.error.message}</p>}
    {error && <p role="alert" className="error-banner">{error}</p>}
    {result && <>
      <div className="cost-summary"><span>{fmtFlop(result.flops)}</span>
        <span>{fmtInt(result.total_memory_bytes)} B · {result.memory_fits ? "容量内" : "OOM"}</span>
        <span>Roofline估算 {result.latency_bound_ms < 0.0001 ? result.latency_bound_ms.toExponential(2) : result.latency_bound_ms.toFixed(4)} ms</span>
        {result.formula_version && <span>{result.formula_version}</span>}</div>
      <div className="table-scroll"><table><thead><tr><th>组 / 算子</th><th>FLOPs</th><th>估算流量 B</th><th>Roofline ms</th></tr></thead>
        <tbody>{result.rows.map((row, index) => <tr key={index}><td>{row.group_id} / {row.operation}</td>
          <td>{fmtInt(row.flops)}</td><td>{fmtInt(row.traffic_bytes)}</td><td>{row.latency_bound_ms.toFixed(5)}</td></tr>)}</tbody></table></div>
      <section className="byte-ledger"><h2>假设与边界</h2><ul>{result.assumptions.map(text => <li key={text}>{text}</li>)}</ul></section>
    </>}
  </main>;
}
