import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { PanelLeftClose, PanelLeft, Play, Save } from "lucide-react";
import { useModels } from "../queries";
import { hardwareApi } from "../../api/hardware";
import { structureApi } from "../../api/structure";
import { api as plannerApi } from "../../api/planner";
import { workloadApi as api } from "../../api/workloads";
import type { Analysis } from "../spec-studio/analysis";
import type { Graph, GraphNode, Plan, PlanRef, Preview, View } from "../../types/workload";
import { active, useCapabilities, useDesign, useRevision, usePlans, useRanks, useRun, useGraph, usePlacementRanks } from "./queries";
import { defaultSearch, initialInput, reducer, validate } from "./state";
import { InputPanel } from "./InputPanel";
import { CommunicationCanvas } from "./CommunicationCanvas";
import { PlacementCanvas } from "./PlacementCanvas";
import { Inspector } from "./Inspector";
import { MemoryTimeline } from "./MemoryTimeline";
import { CandidateTable } from "./CandidateTable";
import { SourceCanvas } from "./SourceCanvas";
import { ParamsLedger } from "./ParamsLedger";
import { LiveResults } from "./LiveResults";
import { EvidenceProvider, RawDetails } from "./Evidence";
import "./workload.css";

export function Designer({ id = "" }: { id?: string }) {
  const [params, setParams] = useSearchParams();
  const latestParams = useRef(params);
  useEffect(() => { latestParams.current = params; }, [params]);
  const navigate = useNavigate(), client = useQueryClient();
  const design = useDesign(id);
  const revisionNumber = params.get("revision") ? Number(params.get("revision")) : null;
  const revision = useRevision(id, revisionNumber);
  const restoredRevision = useRef("");
  const [state, dispatch] = useReducer(reducer, {
    draft: initialInput(params.get("model") ?? "", params.get("hardware") ?? ""),
    base: null, dirty: false, clientRevision: 0,
  });
  const [name, setName] = useState("Inference Design");
  const [search, setSearch] = useState(defaultSearch);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [inputsOpen, setInputsOpen] = useState(true);
  const [mobilePane, setMobilePane] = useState("canvas");
  const [resultTab, setResultTab] = useState<"results" | "details">("results");
  const [preview, setPreview] = useState<Preview>();
  const [compare, setCompare] = useState<PlanRef[]>([]);
  const [phase, setPhase] = useState("prefill");
  const [operator, setOperator] = useState("");
  const [timelineOpen, setTimelineOpen] = useState(false);
  const loaded = useRef("");
  const operationKey = useRef({ signature: "", key: "" });
  const models = useModels();
  const hardware = useQuery({ queryKey: ["hardware-records"], queryFn: hardwareApi.list });
  const caps = useCapabilities(state.draft.model_id, state.draft.hardware_id);
  const runId = params.has("run") ? params.get("run")! : revisionNumber ? "" : design.data?.runs[0]?.id ?? "";
  const run = useRun(runId);
  const plans = usePlans(runId, !!run.data && !active(run.data.status));
  const selectedPlan = plans.data?.items.find(p => p.id === params.get("plan")) ?? plans.data?.items[0];
  const planId = selectedPlan?.id ?? "";
  const layer = Math.max(0, Number(params.get("layer") ?? 0) || 0);
  const rank = Math.max(0, Number(params.get("rank") ?? selectedPlan?.summary.memory.worst_rank ?? 0) || 0);
  const view: View | "params" = ["logical", "placement", "communication"].includes(params.get("view") ?? "")
    ? params.get("view") as View : "params";
  const ranks = useRanks(runId, planId, Math.floor(rank / 100) * 100);
  const rankDetail = ranks.data?.items.find(r => r.rank === rank);
  const commGraph = useGraph(view === "communication" ? runId : "", planId, "communication");
  const placementRanks = usePlacementRanks(view === "placement" ? runId : "", planId);
  const inspection = useQuery({
    queryKey: ["workload-inspection", runId, planId, layer, rank, phase],
    queryFn: () => api.inspection(runId, planId, layer, rank, phase), enabled: !!planId, retry: false,
  });
  const sourceModel = selectedPlan && run.data ? run.data.design_input.model_id : state.draft.model_id;
  const modelDetail = useQuery({ queryKey: ["models", sourceModel], queryFn: () => plannerApi.model(sourceModel), enabled: !!sourceModel });
  const snapshotId = selectedPlan ? run.data?.structure_snapshot_id : undefined;
  const structure = useQuery({
    queryKey: ["workload-source-structure", sourceModel, snapshotId],
    queryFn: ({ signal }) => snapshotId ? structureApi.snapshot(sourceModel, snapshotId, signal)
      : structureApi.create(sourceModel, { kv_dtype: "bf16", include_mtp: false }, signal),
    enabled: !!sourceModel, staleTime: Infinity, retry: false,
  });
  const analysis = structure.data?.analysis as Analysis | undefined;
  const template = useMemo(() => {
    const segment = analysis?.execution.find(s => layer >= s.start && layer < s.stop_exclusive);
    const group = analysis?.groups.find(g => g.id === segment?.group_id);
    return analysis?.templates.find(t => t.id === group?.template_id);
  }, [analysis, layer]);
  const graph = useMemo<Graph | undefined>(() => {
    if (!template || !structure.data) return;
    const c = selectedPlan?.summary.candidate;
    const stage = c?.boundaries.filter(b => layer >= b).length;
    const refs = selectedPlan?.summary.placement.filter(r => r.p === stage).map(r => r.rank) ?? [];
    return {
      nodes: template.nodes.map(n => ({ ...n, layer_index: layer, source_node_id: n.id,
        module_id: n.module_id ?? undefined, rank_refs: refs, stage })),
      edges: template.edges.map((e, i) => ({ ...e, id: `e${i}` })),
      source_structure_id: structure.data.id, plan_id: planId,
      result_hash: selectedPlan?.result_hash ?? "", truncated: false,
    };
  }, [template, structure.data, selectedPlan, layer, planId]);
  const selectedNode = graph?.nodes.find(n => n.id === operator);
  const stale = !!run.data && JSON.stringify(run.data.design_input) !== JSON.stringify(state.draft);
  const errors = validate(state.draft);
  const history = [...(design.data?.runs ?? state.base?.runs ?? [])]
    .sort((a, b) => b.queued_at.localeCompare(a.queued_at) || b.id.localeCompare(a.id)).slice(0, 10);
  const running = history.some(r => active(r.status)) || active(run.data?.status);
  const evaluateReason = busy ? "正在保存并提交…" : running ? "已有运行正在计算，请等待完成或取消后重算。"
    : errors.length ? errors.join("；") : caps.isPending ? "正在检查模型与 GPU 支持情况…"
      : !caps.data?.supported ? "当前模型或 GPU 暂不支持，请查看 Constraints & Support。" : "";
  const candidate = state.draft.parallelism.manual_candidate;
  const requiredGpu = candidate ? candidate.tp * candidate.pp * candidate.dp
    * (state.draft.parallelism.layout === "orthogonal-tp-ep-v1" ? candidate.ep : 1) : 0;
  const budget = state.draft.resources.node_count * state.draft.resources.gpus_per_node;
  const referenceGpus = hardware.data?.find(h => h.id === state.draft.hardware_id)?.node.accelerators ?? 8;
  const budgetNodes = Math.max(state.draft.resources.node_count, Math.ceil(requiredGpu / referenceGpus));
  const budgetGpus = Math.ceil(requiredGpu / budgetNodes);
  const canAdjustBudget = Number.isSafeInteger(requiredGpu) && requiredGpu > budget && requiredGpu <= 256
    && budgetNodes * budgetGpus <= 256;
  const updateUrl = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(latestParams.current);
    Object.entries(patch).forEach(([key, value]) => value == null ? next.delete(key) : next.set(key, value));
    latestParams.current = next;
    setParams(next, { replace: true });
  };

  useEffect(() => {
    if (design.data && loaded.current !== design.data.id) {
      loaded.current = design.data.id;
      dispatch({ type: "saved", design: design.data }); setName(design.data.name);
    }
  }, [design.data]);
  useEffect(() => {
    if (run.data && run.data.design_id !== id) {
      navigate(`/workloads/inference/${run.data.design_id}?${params}`, { replace: true });
    }
  }, [run.data, id, navigate, params]);
  useEffect(() => {
    if (view !== "placement" && view !== "communication") return;
    const current = selectedPlan?.summary.placement.find(r => r.rank === rank);
    if (current && current.d !== 0) {
      const first = selectedPlan?.summary.placement.find(r => r.d === 0
        && r.p === current.p && r.e === current.e && r.t === current.t);
      updateUrl({ rank: String(first?.rank ?? 0) });
    }
  }, [view, rank, selectedPlan]);
  useEffect(() => {
    if (!revisionNumber) { restoredRevision.current = ""; return; }
    if (revision.data && state.base && restoredRevision.current !== revision.data.id) {
      restoredRevision.current = revision.data.id;
      dispatch({ type: "edit", input: revision.data.input });
    }
  }, [revision.data, revisionNumber, state.base]);
  useEffect(() => {
    if (id && run.data && !active(run.data.status)) {
      void client.invalidateQueries({ queryKey: ["workload-design", id] });
    }
  }, [id, run.data?.status, client]);
  useEffect(() => {
    if (!state.dirty) return;
    const guard = (e: BeforeUnloadEvent) => { e.preventDefault(); };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [state.dirty]);
  useEffect(() => {
    setPreview(undefined);
    if (validate(state.draft).length) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void api.preview(state.draft, state.clientRevision, controller.signal).then(result => {
        if (!controller.signal.aborted && result.client_revision === state.clientRevision) setPreview(result);
      }).catch(e => {
        if (!controller.signal.aborted) setPreview({ client_revision: state.clientRevision, coverage: "constraints_only",
          capacity_status: "unknown", needs_full_run: true, lower_bound_bytes: null,
          diagnostics: [{ code: "PREVIEW_UNAVAILABLE", message: String(e) }] });
      });
    }, 300);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [state.draft, state.clientRevision]);

  async function saveAndRun(execute: boolean, fork = false) {
    if (errors.length) { setError(errors.join("；")); return; }
    setBusy(true); setError("");
    try {
      const signature = JSON.stringify({ name, input: state.draft, fork, execute, search });
      if (operationKey.current.signature !== signature) operationKey.current = { signature, key: crypto.randomUUID() };
      const key = operationKey.current.key;
      const saved = state.base && !fork
        ? state.dirty ? await api.save(state.base.id, state.draft, state.base.etag) : state.base
        : await api.create(name, state.draft, `design-${key}`);
      dispatch({ type: "saved", design: saved });
      const result = execute ? await api.submit(saved.revision.id, search, `run-${key}`) : null;
      await client.invalidateQueries({ queryKey: ["workload-designs"] });
      await client.invalidateQueries({ queryKey: ["workload-design", saved.id] });
      if (saved.id !== id) navigate(`/workloads/inference/${saved.id}${result ? `?run=${result.id}` : ""}`);
      else updateUrl({ revision: null, ...(result ? { run: result.id, plan: null, rank: null } : {}) });
      operationKey.current = { signature: "", key: "" };
      if (result) { setResultTab("results"); setMobilePane("details"); }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  }
  async function runAction(action: "cancel" | "retry") {
    setBusy(true); setError("");
    try {
      const result = action === "cancel" ? await api.cancel(runId) : await api.retry(runId, crypto.randomUUID());
      updateUrl({ run: result.id, plan: null, rank: null });
      await client.invalidateQueries({ queryKey: ["workload-run", result.id] });
      await client.invalidateQueries({ queryKey: ["workload-design", id] });
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  async function reloadHead() {
    setBusy(true);
    try {
      const latest = await api.design(id);
      client.setQueryData(["workload-design", id], latest);
      dispatch({ type: "saved", design: latest });
      updateUrl({ revision: null }); setError("");
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }
  function selectPlan(p: Plan) {
    updateUrl({ plan: p.id, rank: String(p.summary.memory.worst_rank) });
  }
  function selectNode(n: GraphNode) {
    if (view === "communication" && n.rank != null) updateUrl({ rank: String(n.rank) });
    else setOperator(n.id);
    setResultTab("details");
    setMobilePane("details");
  }
  if (id && design.isPending) return <p role="status">加载设计…</p>;
  if (design.isError) return <p role="alert">{design.error.message}</p>;
  return <EvidenceProvider plan={selectedPlan} rank={rankDetail} analysis={analysis} run={run.data}>
    <div className={`wc-designer ${inputsOpen ? "" : "wc-inputs-closed"}`} data-pane={mobilePane}>
    <header className="wc-header">
      <Link to="/workloads">Workloads</Link><span>/ Inference</span>
      {state.base ? <strong className="wc-design-name">{name}</strong>
        : <input aria-label="Design Name" value={name} onChange={e => setName(e.target.value)} />}
      <span className="muted">{state.base ? `Revision ${state.base.head_revision}` : "New"}{state.dirty ? " · Unsaved" : ""}</span>
      <div className="wc-actions"><button disabled={busy} onClick={() => void saveAndRun(false)}><Save size={14} /> Save</button>
        {state.base && <button disabled={busy} onClick={() => void saveAndRun(false, true)}>Save as New Design</button>}
        <button className="primary" disabled={!!evaluateReason} title={evaluateReason || undefined}
          onClick={() => void saveAndRun(true)}><Play size={14} />{state.draft.parallelism.mode === "manual" ? "Evaluate" : "Auto Design"}</button></div>
      {canAdjustBudget && <div className="wc-evaluate-feedback">
        <span>需要 {requiredGpu} GPU，当前预算 {budget} GPU。</span>
          <button disabled={busy} onClick={() => dispatch({ type: "edit", input: { ...state.draft,
            resources: { ...state.draft.resources, node_count: budgetNodes, gpus_per_node: budgetGpus } } })}>
            调整 GPU 预算至 {budgetNodes * budgetGpus}</button>
      </div>}
    </header>
    <div className="wc-zone-title"><h2>Model & Hardware</h2></div>
    <fieldset className="wc-source-row" disabled={busy} aria-label="Model and hardware sources">
      <label>Model<select aria-label="Model" value={state.draft.model_id} onChange={e => {
        dispatch({ type: "edit", input: { ...state.draft, model_id: e.target.value,
          parallelism: { ...state.draft.parallelism, expert_parallel: false, allowed_tp: null, allowed_pp: null,
            allowed_ep: null, layer_locks: [], pp_boundaries: null, manual_candidate: null, mode: "auto" } } });
        setOperator(""); updateUrl({ layer: "0", plan: null, run: "", rank: null });
      }}><option value="">Select Model</option>{models.data?.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select></label>
      <label>Hardware<select aria-label="Hardware" value={state.draft.hardware_id} onChange={e =>
        dispatch({ type: "edit", input: { ...state.draft, hardware_id: e.target.value } })}>
        <option value="">Select GPU</option>{hardware.data?.map(h => <option key={h.id} value={h.id}>{h.name}</option>)}</select></label>
      <label>Run History<select aria-label="Run History" value={runId} onChange={e => updateUrl({ run: e.target.value, plan: null, rank: null })}>
        <option value="">Current Draft</option>{history.map(r =>
          <option key={r.id} value={r.id}>{r.status} · {new Date(r.queued_at).toLocaleString()}</option>)}
        {runId && !history.some(r => r.id === runId) && <option value={runId}>Earlier run · {run.data?.status ?? "Loading"}
          {run.data && ` · ${new Date(run.data.queued_at).toLocaleString()}`}</option>}
      </select></label>
    </fieldset>
    {(error || caps.isError || structure.isError || revision.isError || run.isError || plans.isError) && <div className="wc-error" role="alert">
      {error || caps.error?.message || structure.error?.message || revision.error?.message || run.error?.message || plans.error?.message}
      {error.includes("REVISION_CONFLICT") && <button disabled={busy} onClick={() => void reloadHead()}>Reload Head</button>}
    </div>}
    {state.base && <details className="wc-issues wc-history"><summary>Revision History & Sources</summary>
      <label>Revision <select aria-label="Revision" value={revisionNumber ?? ""} onChange={e => updateUrl({ revision: e.target.value || null })}>
        <option value="">Current Draft</option>{Array.from({ length: state.base.head_revision }, (_, i) =>
          <option value={i + 1} key={i}>{i + 1}</option>)}</select></label>
      <button disabled={busy} onClick={() => void reloadHead()}>Reload Head</button>
      {run.data && <><button onClick={() => {
        dispatch({ type: "edit", input: run.data!.design_input }); updateUrl({ revision: null });
      }}>Use Run Input</button><RawDetails title="Frozen Source Records"><p>Frozen Input · {run.data.input_hash}</p>
        {run.data.source_updates.map(s => <p key={s.source}>{s.source} · {s.pinned_version} · {s.change_kind}</p>)}
        <pre>{JSON.stringify(run.data.versions, null, 2)}</pre></RawDetails></>}
      {revisionNumber && <p>Revision {revisionNumber} loaded as draft. Save creates a new revision.</p>}
    </details>}
    {(errors.length > 0 || caps.data?.issues.length || preview?.diagnostics.length) ? <details className="wc-issues" open>
      <summary>Constraints & Support</summary>{errors.map(e => <p key={e}>{e}</p>)}
      {(caps.data?.issues ?? []).map((issue, i) => <p key={i}><strong>{issue.code}</strong> · {issue.message}</p>)}
      {preview?.diagnostics.map((issue, i) => <p key={i}>{issue.code} · {issue.message}</p>)}
    </details> : null}
    {stale && <p className="wc-notice" role="status">Stale Result · 画布和诊断仍对应所选运行的冻结输入，当前修改需重新运行。</p>}
    <nav className="wc-mobile-tabs">{["inputs", "canvas", "details"].map(p => <button key={p} aria-pressed={mobilePane === p}
      onClick={() => setMobilePane(p)}>{p === "inputs" ? "Parameters" : p === "canvas" ? "Workspace" : "Results"}</button>)}</nav>
    <div className="wc-zone-title wc-configuration-title"><h2>Workload Configuration</h2>
      <button aria-label="Toggle Parameters" aria-expanded={inputsOpen} onClick={() => setInputsOpen(!inputsOpen)}>
        {inputsOpen ? "Collapse parameters" : "Expand parameters"}</button></div>
      <fieldset className="wc-input-panel" disabled={busy}>
        <InputPanel key={state.base?.id ?? "new"} input={state.draft} change={input => dispatch({ type: "edit", input })}
          capabilities={caps.data} search={search} setSearch={setSearch} selectedLayer={layer}
          memory={rankDetail ?? selectedPlan?.summary.memory} stale={stale} />
      </fieldset>
    <div className="wc-zone-title"><h2>Analysis Workspace</h2></div>
    <div className="wc-workbench">
      <section className="wc-canvas-panel">
        <div className="wc-toolbar wc-tabs"><button aria-label="Toggle Parameters from workspace" onClick={() => setInputsOpen(!inputsOpen)}>
          {inputsOpen ? <PanelLeftClose size={16} /> : <PanelLeft size={16} />}</button>
          {(["params", "logical", "placement", "communication"] as const).map(v => <button key={v} aria-pressed={view === v}
            disabled={["placement", "communication"].includes(v) && !selectedPlan} onClick={() => updateUrl({ view: v })}>
            {v === "params" ? "Params" : v === "logical" ? "Structure" : v === "placement" ? "Placement" : "Communication"}</button>)}</div>
        <div className="wc-toolbar wc-scope-row"><label>Layer<select aria-label="Layer" value={layer} onChange={e => {
          updateUrl({ layer: e.target.value }); setOperator("");
        }}>{Array.from({ length: analysis?.backbone.num_layers ?? 1 }, (_, i) =>
          <option value={i} key={i}>Layer {i}</option>)}</select></label>
          <label>GPU<select aria-label="GPU" value={rank} disabled={!selectedPlan} onChange={e => updateUrl({ rank: e.target.value })}>
            {(selectedPlan?.summary.placement.filter(r => view !== "placement" && view !== "communication" || r.d === 0)
              ?? [{ rank: 0, p: 0 }]).map(r =>
              <option key={r.rank} value={r.rank}>GPU {r.rank} · PP {r.p}</option>)}</select></label>
          <label>Phase<select aria-label="Phase" value={phase} onChange={e => setPhase(e.target.value)}><option value="prefill">Prefill</option><option value="decode">Decode</option></select></label>
        </div>
        {view === "params" ? <ParamsLedger analysis={analysis} model={modelDetail.data} plan={selectedPlan} rank={rankDetail}
          select={(node, layer) => { setOperator(node); updateUrl({ layer: String(layer), view: "logical" }); setResultTab("details"); setMobilePane("details"); }} />
          : view === "logical" ? template ? <SourceCanvas template={template} selectedId={operator} inspection={inspection.data}
          onSelect={value => { setOperator(value); setMobilePane("details"); setResultTab("details"); }} />
          : <p className="wc-empty">{structure.isFetching ? "加载模型结构…" : "选择 Model，查看计算结构。"}</p>
          : view === "placement" && selectedPlan ? <>
            {placementRanks.isError && <p role="alert">{placementRanks.error.message}</p>}
            <PlacementCanvas plan={selectedPlan} ranks={placementRanks.data ?? []}
              selectedRank={rank} onSelect={r => { updateUrl({ rank: String(r) }); setMobilePane("details"); setResultTab("details"); }} /></>
          : commGraph.data ? <CommunicationCanvas key={planId} graph={commGraph.data} selectedRank={rank} phase={phase} onSelect={selectNode} />
            : <p role={commGraph.isError ? "alert" : "status"}>{commGraph.error?.message ?? "Loading network…"}</p>}
      </section>
      <section className="wc-results-panel">
        <nav className="wc-toolbar wc-tabs">{(["results", "details"] as const).map(tab =>
          <button key={tab} aria-pressed={resultTab === tab} onClick={() => setResultTab(tab)}>{tab === "results" ? "Live Results" : "Details"}</button>)}</nav>
        {resultTab === "results" ? <LiveResults plan={selectedPlan} rank={rankDetail} ranks={ranks.data?.items} run={run.data} stale={stale} />
          : <Inspector node={selectedNode} inspection={inspection.data} rank={rankDetail} pending={inspection.isFetching}
            error={inspection.error?.message} stale={stale} candidate={selectedPlan?.summary.candidate} />}
      </section>
    </div>
    <div className="wc-run-status" role="status">
      <span>{run.data ? `Run ${run.data.status} · ${run.data.summary?.outcome ?? ""}` : "Ready"}
        {run.data?.progress.evaluated != null ? ` · ${run.data.progress.evaluated} evaluated` : ""}</span>
      {run.data?.error && <span className="bad">{run.data.error.code} · {run.data.error.message}</span>}
      {run.data && <>{active(run.data.status) ? <button disabled={busy} onClick={() => void runAction("cancel")}>Cancel</button>
        : <button disabled={busy} onClick={() => void runAction("retry")}>Retry Frozen Input</button>}
        <a href={api.exportUrl(runId)} download>Export JSON</a></>}
      {rankDetail && <button onClick={() => setTimelineOpen(!timelineOpen)}>Memory Timeline</button>}
      {compare.length >= 2 && <Link to={`/workloads/compare?plans=${encodeURIComponent(JSON.stringify(compare))}`}>Compare ({compare.length})</Link>}
    </div>
    {run.data?.summary?.rejections?.length ? <details className="wc-issues"><summary>Rejected Candidates</summary>
      {run.data.summary.rejections.map((r, i) => <p key={i}>{r.code} · {r.message}</p>)}</details> : null}
    <CandidateTable data={plans.data} selected={planId} select={selectPlan} compare={compare} toggle={p =>
      setCompare(current => current.some(r => r.plan_id === p.id) ? current.filter(r => r.plan_id !== p.id)
        : current.length < 4 ? [...current, { run_id: p.run_id, plan_id: p.id }] : current)} />
    {timelineOpen && rankDetail && <MemoryTimeline key={`${planId}-${rank}`} rank={rankDetail}
      coverage={selectedPlan?.summary.event_memory.coverage ?? "prefix"} />}
  </div></EvidenceProvider>;
}
