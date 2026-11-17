import { Fragment, type ReactNode } from "react";
import type { Analysis, ModuleSpec } from "../spec-studio/analysis";
import type { Model } from "../../types";
import type { Plan, RankDetail, Trace } from "../../types/workload";
import { number, bytes } from "./state";
import { CalculationTrace } from "./CalculationTrace";

export function ParamsLedger({ analysis, model, plan, rank, select }: {
  analysis?: Analysis; model?: Model; plan?: Plan; rank?: RankDetail;
  select: (node: string, layer: number) => void;
}) {
  if (!analysis) return <p className="wc-empty">选择 Model，查看参数账本。</p>;
  const b = analysis.backbone;
  const raw = model?.config_sha256 === analysis.source.config_sha256 ? model.raw_config : undefined;
  const config = (raw?.text_config ?? raw) as Record<string, unknown> | undefined;
  const value = (v: unknown) => v == null ? "Unavailable" : typeof v === "number" ? number(v) : typeof v === "string" ? v : JSON.stringify(v);
  const cell = (text: string, trace?: Trace, label?: string) => <span className="wc-inline-value">{text}
    <CalculationTrace trace={trace} label={label} inline /></span>;
  type Pair = { id: string; label: string; value?: ReactNode };
  const modules = (items: ModuleSpec[], templateId?: string) => items.map(m => {
    const t = analysis.templates.find(t => t.id === templateId);
    const node = t?.nodes.find(n => n.module_id === m.id);
    const group = analysis.groups.find(g => g.template_id === templateId);
    const layer = analysis.execution.find(s => s.group_id === group?.id)?.start ?? 0;
    return <tr key={`${templateId ?? "shared"}-${m.id}`}><td>{node
      ? <button className="wc-module-link" onClick={() => select(node.id, layer)}>{m.label}</button> : m.label}
      {m.shared_with && <span className="muted"> · shared with {m.shared_with}</span>}</td>
      <td>{m.shape.join(" × ")} × {m.count ?? 1}</td><td>{number(m.params)}</td><td>{number(m.active_params)}</td><td>{m.dtype}</td></tr>;
  });
  const memory = rank ?? plan?.summary.memory;
  const traces = plan?.summary.explanation?.traces;
  const architecture: Pair[] = [
    { id: "model_type", label: "Architecture", value: value(analysis.identity.model_type) },
    { id: "layers", label: "Layers", value: value(b.num_layers) },
    { id: "hidden", label: "Hidden Size", value: value(b.hidden_size) },
    { id: "vocab", label: "Vocab Size", value: value(b.vocab_size) },
    { id: "position", label: "Max Position", value: value(b.context_length) },
    { id: "total", label: "Total Parameters", value: cell(value(analysis.totals.total), traces?.parameters, "Total Parameters") },
    { id: "active", label: "Active Parameters", value: value(analysis.totals.active) },
    { id: "source_weight", label: "Source Weight Memory", value: cell(bytes(analysis.serving.weight_bytes), traces?.source_weight, "Source Weight") },
    { id: "runtime_weight", label: "Runtime Weight / GPU", value: cell(memory ? bytes(memory.components.weight) : "Unavailable",
      memory?.explanation?.traces.weight, "Weights / GPU") },
    { id: "dtype", label: "Source Dtype", value: value(config?.torch_dtype ?? config?.dtype) },
  ];
  const templates = analysis.templates.map(t => {
    const prefix = t.id;
    const rows: Pair[] = [
      { id: `${prefix}-heads`, label: "Attention Heads", value: value(t.backbone?.num_heads ?? b.num_heads) },
      { id: `${prefix}-kv`, label: "KV Heads", value: value(t.backbone?.num_kv_heads ?? b.num_kv_heads) },
      { id: `${prefix}-dim`, label: "Head Dim", value: value(t.backbone?.head_dim ?? b.head_dim) },
      { id: `${prefix}-rope`, label: "RoPE", value: value(config?.rope_parameters ?? config?.rope_scaling ?? config?.rope_theta) },
      { id: `${prefix}-intermediate`, label: "Intermediate Size", value: value(t.moe?.expert_dim ?? b.intermediate_size) },
      { id: `${prefix}-activation`, label: "Activation", value: value(config?.hidden_act) },
      { id: `${prefix}-moe`, label: "Is MoE", value: t.moe ? "Yes" : "No" },
      { id: `${prefix}-routed`, label: "Routed Experts", value: value(t.moe?.n_routed ?? 0) },
      { id: `${prefix}-expert-dim`, label: "Expert Dim", value: value(t.moe?.expert_dim) },
      { id: `${prefix}-topk`, label: "Top K", value: value(t.moe?.top_k) },
      { id: `${prefix}-shared`, label: "Shared Experts", value: value(t.moe?.n_shared ?? 0) },
      { id: `${prefix}-count`, label: "Layer Count", value: value(plan?.summary.explanation?.template_layer_counts?.[t.id]
        ?? analysis.groups.filter(g => g.template_id === t.id).map(g => g.count).join(" + ")) },
    ];
    return { id: prefix, title: `${t.label} · ${t.attn_kind} / ${t.ffn_kind}`, rows };
  });
  const quad = (left: Pair[], right: Pair[]) => Array.from({ length: Math.max(left.length, right.length) }, (_, i) => {
    const l = left[i], r = right[i];
    return <tr key={l?.id ?? r?.id ?? i}>
      <th>{l?.label ?? ""}</th><td>{l?.value}</td><th>{r?.label ?? ""}</th><td>{r?.value}</td>
    </tr>;
  });
  return <div className="wc-params" aria-label="Model parameter ledger">
    <h3>{analysis.identity.name}</h3>
    <section><h3>Architecture</h3>
      <table className="wc-param-summary wc-param-quad">
        <colgroup><col /><col /><col /><col /></colgroup>
        <tbody>
        {templates.length === 0 ? quad(architecture, []) : templates.map((t, index) => <Fragment key={t.id}>
          <tr className="wc-param-group"><td colSpan={2} /><th colSpan={2}>{t.title}</th></tr>
          {quad(index === 0 ? architecture : [], t.rows)}
        </Fragment>)}
      </tbody></table>
    </section>
    <section><h3>Module Ledger</h3>
      <div className="wc-param-scroll"><table>
        <thead><tr><th>Module</th><th>Shape × Count</th><th>Parameters</th><th>Active</th><th>Dtype</th></tr></thead>
        <tbody>{analysis.templates.map(t => modules(t.modules, t.id))}</tbody>
      </table></div>
    </section>
    <section><h3>Shared Modules</h3>
      <div className="wc-param-scroll"><table>
        <thead><tr><th>Module</th><th>Shape × Count</th><th>Parameters</th><th>Active</th><th>Dtype</th></tr></thead>
        <tbody>{modules(analysis.shared_modules)}</tbody>
      </table></div>
    </section>
    <p className="wc-model-source">Model Source · {analysis.source.resolved_sha ?? analysis.source.config_sha256}</p>
  </div>;
}
