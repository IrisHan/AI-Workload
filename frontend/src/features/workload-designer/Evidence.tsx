import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { X } from "lucide-react";
import type { Plan, RankDetail, Run, Trace } from "../../types/workload";
import type { Analysis } from "../spec-studio/analysis";
import { CalculationTrace, DialogFacts, Facts, factValue } from "./CalculationTrace";
import { groupLayers, layerShards, metricNames, relevantInputs } from "./calculationEvidence";
import { useLayers } from "./queries";
import { bytes, number } from "./state";

interface EvidenceState { open: (trace: Trace) => void }
const EvidenceContext = createContext<EvidenceState>({ open: () => undefined });
export const useEvidence = () => useContext(EvidenceContext);
export function RawDetails({ children, title }: { children: ReactNode | (() => ReactNode); title: string }) {
  const [expanded, setExpanded] = useState(false);
  return <details onToggle={e => setExpanded(e.currentTarget.open)}><summary>{title}</summary>
    {expanded && (typeof children === "function" ? children() : children)}</details>;
}

function LayerTemplates({ plan, rank, trace, analysis }: { plan: Plan; rank?: RankDetail; trace: Trace; analysis?: Analysis }) {
  const selectedRank = Number(trace.scope.rank ?? rank?.rank ?? plan.summary.memory.worst_rank);
  const query = useLayers(plan.run_id, plan.id, selectedRank);
  const pages = query.data?.pages ?? [];
  if (query.isPending) return <p role="status">Loading layer templates…</p>;
  if (query.error) return <p role="alert">{query.error.message}</p>;
  if (pages.some(p => p.result_hash !== plan.result_hash)) return <p role="alert">Calculation Inconsistency · Layer result binding.</p>;
  // Admission cache and resident weights use one phase only; never sum prefill + decode copies.
  const rows = pages.flatMap(p => p.items).filter(l => l.phase === "prefill");
  const groups = groupLayers(rows);
  const kv = trace.id === "kv";
  const complete = !query.hasNextPage;
  const cacheTotal = rows.reduce((sum, row) => sum + row.cache.total_bytes, 0);
  const expectedLayers = Number(plan.summary.explanation?.template_layer_counts
    ? Object.values(plan.summary.explanation.template_layer_counts).reduce((a, b) => a + b, 0) : 0);
  return <section className="wc-layer-templates" aria-label="Transformer Layer Templates">
    <h3>Transformer Layer Template</h3>
    <p className="muted">GPU {selectedRank} · {rows.length} resident layers{complete ? "" : " · Partial page — load remaining layers"}
      {expectedLayers > 0 ? ` / ${expectedLayers} model layers` : ""}</p>
    {groups.map((group, index) => {
      const row = group.representative, count = group.layers.length;
      const perLayer = kv ? row.cache.total_bytes : group.weightBytes;
      const template = analysis?.templates.find(t => t.id === row.template_id);
      return <section className="wc-template" key={index}>
        <strong>Stage {row.stage} · Layers {group.layers.map(l => l.layer_index).length === 1 ? row.layer_index
          : `${group.layers[0].layer_index}–${group.layers.at(-1)!.layer_index}`}</strong>
        <p>{bytes(perLayer)} / layer × {count} identical layers = {bytes(perLayer == null ? null : perLayer * count)}
          {kv ? " KV allocation" : " weights"}</p>
        {!kv && template && <p>{number(template.modules.reduce((sum, m) => sum + m.params, 0))} source parameters / Transformer layer</p>}
        <details><summary>{kv ? "Show representative layer" : "Show weight calculation"}</summary>
          {kv ? <>
            <dl className="wc-values">{Object.entries(row.cache).map(([key, v]) =>
              <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{bytes(v)}</dd></div>)}
              {template && [["KV Heads · Model Source", template.backbone?.num_kv_heads ?? analysis?.backbone.num_kv_heads],
                ["Head Size", template.backbone?.head_dim ?? analysis?.backbone.head_dim]].map(([k, v]) =>
                <div key={String(k)}><dt>{k}</dt><dd>{String(v ?? "Unavailable")}</dd></div>)}</dl>
            <p className="muted">Payload includes page tail waste; metadata is added once.</p>
          </> : ["Attention", "MLP", "Norm", "Other"].map(category => {
            const shards = layerShards(row).filter(s => {
              const c = /norm/.test(s.module_id) ? "Norm" : /[qkvo]_proj|attn|attention/.test(s.module_id)
                ? "Attention" : /gate|up_proj|down_proj|expert|ffn/.test(s.module_id) ? "MLP" : "Other";
              return c === category;
            });
            return shards.length > 0 && <section key={category}><h4>{category}</h4><dl className="wc-values">
              {shards.map((s, i) => <div key={i}><dt>{s.module_id}</dt><dd>
                {s.local_shape.join(" × ")}{s.bias ? ` + ${s.bias} bias` : ""} · {s.count} copies
                <br />{bytes(s.bytes)} · {s.partition}
                {template?.modules.find(m => m.id === s.module_id) && <small>
                  Source: {number(template.modules.find(m => m.id === s.module_id)!.params)} params</small>}
              </dd></div>)}</dl></section>;
          })}
        </details>
        <RawDetails title={`All layer details · ${count}`}>{() => <pre>{JSON.stringify(group.layers, null, 2)}</pre>}</RawDetails>
      </section>;
    })}
    {kv && complete && cacheTotal !== trace.result.value && <p className="wc-notice">
      Raw allocation {bytes(cacheTotal)} → effective reserve {factValue(trace.result.value, trace.result.unit)}.
      See Formula and Assumptions for KV efficiency and reserve rounding.</p>}
    {!kv && rank?.rank === selectedRank && <section><h4>Shared Modules · Embedding / LM Head / Norm</h4>
      <dl className="wc-values">{rank.weights.filter(s => s.storage_id.includes("/shared/")).map(s =>
        <div key={s.storage_id}><dt>{s.module_id}</dt><dd>{bytes(s.bytes)}</dd></div>)}</dl></section>}
    <RawDetails title="Raw layer records · All phases">{() => <>
      <pre>{JSON.stringify(pages.flatMap(p => p.items), null, 2)}</pre>
      {!complete && <p>Partial results · load remaining layers below.</p>}
    </>}</RawDetails>
    {query.hasNextPage && <button disabled={query.isFetchingNextPage} onClick={() => void query.fetchNextPage()}>Load remaining layers</button>}
  </section>;
}

function EvidenceBody({ trace, plan, rank, analysis, run }: {
  trace: Trace; plan?: Plan; rank?: RankDetail; analysis?: Analysis; run?: Run;
}) {
  const inputs = relevantInputs(trace);
  const [formula, setFormula] = useState(false);
  const sourceParameter = ["parameters", "source_weight"].includes(trace.id);
  return <>
    {trace.status !== "available" && <p className="wc-notice">{trace.unavailable_reason ?? "Unavailable"}</p>}
    {inputs.length > 0 && <><h3>Uses</h3><Facts items={inputs} /></>}
    {trace.id === "peak" && <div className="wc-component-links">
      {Object.entries((rank?.rank === trace.scope.rank ? rank : plan?.summary.memory)?.explanation?.traces ?? {})
        .filter(([key]) => ["weight", "kv", "activation", "communication", "runtime", "fragmentation", "safety"].includes(key))
        .map(([key, t]) => <CalculationTrace key={key} trace={t} />)}
    </div>}
    {trace.id === "itl_ms" && <p className="wc-notice">
      Previous token → wait while other waves run → this request’s wave service → next token.
      ITL averages request intervals; Iteration Time averages completed wave service.
      Sum of Decode-wave Service already includes other requests’ work, not a second copy of this request’s wait.</p>}
    {trace.id === "output_tokens_s" && plan?.summary.explanation && <p>
      {plan.summary.explanation.performance.completed_full_outputs} completed-request tokens +{" "}
      {plan.summary.explanation.performance.unfinished_outputs_in_window} unfinished-request tokens −{" "}
      {plan.summary.explanation.performance.completed_outputs_before_window} before-window tokens.
    </p>}
    {plan && ["kv", "weight"].includes(trace.id) && <LayerTemplates plan={plan} rank={rank} trace={trace} analysis={analysis} />}
    {sourceParameter && analysis && <section><h3>Source Templates</h3>
      {analysis.templates.map(t => <p key={t.id}>{t.label} · {number(t.modules.reduce((a, m) => a + m.params, 0))} params / layer
        {" × "}{analysis.groups.filter(g => g.template_id === t.id).reduce((a, g) => a + g.count, 0)} layers</p>)}
      <details><summary>参数矩阵</summary>{analysis.templates.map(t => <section key={t.id}><h4>{t.label}</h4>
        <dl className="wc-values">{t.modules.map(m => <div key={m.id}><dt>{m.label}</dt>
          <dd>{m.shape.join(" × ")} · {number(m.params)} params · {m.dtype}</dd></div>)}</dl></section>)}</details>
      <h4>Shared Modules</h4><Facts items={analysis.shared_modules.map(m => ({
        key: m.id, label: m.label, value: m.params, unit: "parameters", source: "Model Source" }))} />
    </section>}
    <button className="wc-evidence-link" aria-expanded={formula} onClick={() => setFormula(!formula)}>Show Formula</button>
    {formula && <section aria-label="Formula">
      {trace.id === "kv" && analysis?.templates.every(t => ["GQA", "MHA", "MQA"].includes(t.attn_kind.toUpperCase())) &&
        <p>Per-layer K/V payload: ceil(context / block_size) × block_size × resident_batch
          × 2 (K/V) × local_kv_heads × head_size × dtype_bytes.
          Local KV heads follow the frozen TP partition. Page metadata is added separately.</p>}
      <pre className="wc-formula">{trace.formula}</pre>
      {trace.steps.map((s, i) => <p key={i}>{s.label}: {s.formula} = {factValue(s.value, s.unit)}</p>)}</section>}
    <details><summary>Show dependency</summary>
      {inputs.length ? <Facts items={inputs} /> : <p>No compact dependency mapping for this historical metric. See Raw Calculation for full saved evidence.</p>}
      <p className="muted">Shared window and hardware values appear once in Calculation Context. Full saved inputs remain in Raw Calculation.</p>
    </details>
    <details><summary>Assumptions</summary><Facts items={trace.assumptions} />{trace.notes.map((n, i) => <p key={i}>{n}</p>)}</details>
    <details><summary>Source</summary><p>Frozen model and hardware sources · not a GPU benchmark.</p>
      {[...new Set(trace.sources.map(s => s.type))].map(t => <p key={t}>{t}</p>)}
      <RawDetails title="Model Revision & Source Paths">{() => <pre>{JSON.stringify({
        sources: trace.sources, scope: trace.scope, versions: run?.versions }, null, 2)}</pre>}</RawDetails></details>
    <RawDetails title="Raw Calculation">{() => <><h3>Calculation Inputs</h3><Facts items={trace.inputs} />
      <pre>{JSON.stringify({ trace, model_revision: trace.scope.model_revision, versions: run?.versions }, null, 2)}</pre></>}</RawDetails>
  </>;
}

export function EvidenceProvider({ children, plan, rank, analysis, run }: {
  children: ReactNode;
  plan?: Plan; rank?: RankDetail; analysis?: Analysis; run?: Run;
}) {
  const [trace, setTrace] = useState<Trace>();
  const dialog = useRef<HTMLDialogElement>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const close = () => { dialog.current?.close(); setTrace(undefined); trigger.current?.focus(); };
  useEffect(() => {
    dialog.current?.close();
    setTrace(undefined);
    trigger.current = null;
  }, [run?.id, plan?.id, rank?.rank]);
  useEffect(() => { if (trace) dialog.current?.showModal(); }, [trace]);
  return <EvidenceContext.Provider value={{ open: t => {
    if (!dialog.current?.open) trigger.current = document.activeElement as HTMLElement;
    setTrace(t);
  } }}>
    {children}
    <dialog className="wc-evidence-dialog" ref={dialog} aria-labelledby="wc-evidence-title" onCancel={e => { e.preventDefault(); close(); }}>
      {trace && <>
        <header className="wc-evidence-head">
          <p className="wc-evidence-headline">
            <span id="wc-evidence-title">{metricNames[trace.id] ?? trace.title}</span>
            <strong>{factValue(trace.result.value, trace.id === "parameters" ? "" : trace.result.unit)}</strong>
          </p>
          <button type="button" className="wc-dialog-close" onClick={close} aria-label="Close evidence">
            <X size={14} aria-hidden="true" /></button>
        </header>
        <DialogFacts.Provider value={true}>
          <EvidenceBody key={trace.id} trace={trace} plan={plan} rank={rank} analysis={analysis} run={run} />
        </DialogFacts.Provider>
      </>}
    </dialog>
  </EvidenceContext.Provider>;
}
