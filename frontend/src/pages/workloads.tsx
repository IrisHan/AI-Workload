import { useDeferredValue, useState } from "react";
import { Link, Navigate, useParams, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { PageTitle, ErrorState, Loading, Empty } from "../components/common";
import { Designer } from "../features/workload-designer/Designer";
import { useComparison, useDesigns, active } from "../features/workload-designer/queries";
import { useModels } from "../features/queries";
import { workloadApi } from "../api/workloads";
import type { Design, PlanRef } from "../types/workload";
import { bytes, number } from "../features/workload-designer/state";

export function WorkloadsPage() {
  const [q, setQ] = useState("");
  const query = useDeferredValue(q);
  const designs = useDesigns(query), models = useModels(), client = useQueryClient();
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState("");
  async function remove(d: Design) {
    setDeleting(d.id); setError("");
    try { await workloadApi.remove(d.id, d.etag); await client.invalidateQueries({ queryKey: ["workload-designs"] }); }
    catch (e) { setError(String(e)); } finally { setDeleting(""); }
  }
  const items = designs.data?.pages.flatMap(p => p.items) ?? [];
  return <div className="wc-list"><PageTitle title="Workload Center">
    <Link className="button primary" to="/workloads/inference/new">New Inference Design</Link></PageTitle>
    <div className="tabs"><button className="selected">Inference</button><button disabled title="Not implemented">Training</button>
      <button disabled title="Not implemented">Fine-tuning</button></div>
    <label className="wc-search">Search Designs<input type="search" aria-label="Search Designs" value={q}
      placeholder="Design, model or hardware" onChange={e => setQ(e.target.value)} /></label>
    {error && <p role="alert" className="bad">{error}</p>}
    {designs.isPending ? <Loading /> : designs.isError ? <ErrorState error={designs.error} /> :
      !items.length ? <Empty>{q ? "No matching designs." : "No inference designs."} <Link to="/workloads/inference/new">Create a design</Link></Empty> :
        <div className="table-scroll"><table><thead><tr><th>Design</th><th>Model</th><th>Hardware</th><th>Mode</th><th>Revision</th><th>Latest Run</th><th>Updated</th><th /></tr></thead>
          <tbody>{items.map(d => <tr key={d.id}><td><Link to={`/workloads/inference/${d.id}`}>{d.name}</Link></td>
            <td>{d.model_name ?? models.data?.find(m => m.id === d.revision.input.model_id)?.name ?? d.revision.input.model_id}</td>
            <td>{d.hardware_name ?? d.revision.input.hardware_id}</td>
            <td>{d.revision.input.mode}</td><td>{d.head_revision}</td><td>{d.runs[0]?.status ?? "Not run"}</td>
            <td>{new Date(d.updated_at).toLocaleString()}</td><td><button disabled={!!deleting || d.runs.some(r => active(r.status))}
              onClick={() => void remove(d)}>Delete</button></td></tr>)}</tbody></table></div>}
    {designs.hasNextPage && <button onClick={() => void designs.fetchNextPage()}>Load More</button>}
  </div>;
}
export function NewWorkloadPage() {
  const [params] = useSearchParams();
  const next = new URLSearchParams();
  for (const key of ["model", "hardware"]) if (params.get(key)) next.set(key, params.get(key)!);
  return <Navigate replace to={`/workloads/inference/new?${next}`} />;
}
export function InferenceDesignPage() {
  const { designId = "" } = useParams();
  return <Designer key={designId} id={designId} />;
}
export function WorkloadDetailPage() {
  return <><PageTitle title="Legacy Workload Removed" /><p>旧版 Workload 数据已清理，请创建新的 Inference Design。</p>
    <Link className="button primary" to="/workloads/inference/new">New Inference Design</Link></>;
}
export function WorkloadComparePage() {
  const [params] = useSearchParams();
  let refs: PlanRef[] = [];
  try {
    const value: unknown = JSON.parse(params.get("plans") ?? "[]");
    if (Array.isArray(value) && value.length >= 2 && value.length <= 4 && value.every(r =>
      r && typeof r.run_id === "string" && typeof r.plan_id === "string")) refs = value;
  } catch { /* invalid URL is rendered below */ }
  const query = useComparison(refs);
  return <><PageTitle title="Compare Inference Plans"><Link to="/workloads">Workloads</Link></PageTitle>
    {!refs.length ? <p role="alert">请选择 2–4 个候选进行比较。</p> : query.isPending ? <Loading /> : query.isError ?
      <ErrorState error={query.error} /> : <div className="wc-compare">
        <table><thead><tr><th>Metric</th>{query.data.plans.map((p, i) => <th key={i}>
          <Link to={`/workloads/inference/new?run=${p.run_id}&plan=${p.id}`}>{p.candidate_key}</Link></th>)}</tr></thead>
          <tbody><tr><th>Peak / GPU</th>{query.data.plans.map((p, i) => <td key={i}>{bytes(p.summary.memory.peak_bytes)}</td>)}</tr>
            {["capacity_status", "execution_status", "target_status"] .map(key => <tr key={key}><th>{key}</th>
              {query.data.plans.map((p, i) => <td key={i}>{String(p.summary[key as "capacity_status"])}</td>)}</tr>)}
            {Object.keys(query.data.plans[0].summary.metric_values).map(key => <tr key={key}><th>{key}</th>
              {query.data.plans.map((p, i) => { const m = p.summary.metric_values[key]; return <td key={i}>
                {number(m?.value)} {m?.unit}<br />{m?.kind} · {m?.unavailable_reason}</td>; })}</tr>)}</tbody></table>
        <h2>Input Differences</h2>{query.data.input_differences.length ? <table><tbody>{query.data.input_differences.map(d =>
          <tr key={d.path}><th>{d.path}</th>{d.values.map((v, i) => <td key={i}>{JSON.stringify(v)}</td>)}</tr>)}</tbody></table>
          : <p>相同输入。</p>}
        <h2>Versions & Assumptions</h2>{query.data.versions.map((v, i) => <details key={i}><summary>Plan {i + 1}</summary>
          <pre>{JSON.stringify({ versions: v, assumptions: query.data.assumptions[i] }, null, 2)}</pre>
          <a href={workloadApi.exportUrl(refs[i].run_id)} download>Export JSON</a></details>)}
      </div>}</>;
}
