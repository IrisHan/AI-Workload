import { useEffect, useState } from "react";
import { ArrowRight, Braces, Check, ChevronLeft, ChevronRight, Download, ExternalLink,
  Layers3, ListTree, Search, ShieldCheck, SquareArrowOutUpRight, X, Activity } from "lucide-react";
import { LayerFlow } from "./LayerFlow";
import { categoryTone, ffnRecipe, isMoeFfn, moeLayout, rangeLabel, type Analysis, type GraphScope, type ModuleSpec, type Template } from "./analysis";
import { useSearchParams, useLocation } from "react-router-dom";
import { useAnalysisRequest, useStudioGlossary } from "./queries";
import type { StructureSnapshot } from "../../types";
import { fmtFlop, fmtInt, fmtParams } from "./format";
import { Instances } from "./Instances";
import { CostView } from "./CostView";
import { NodeDetails } from "./NodeDetails";

type View = "structure" | "ledger" | "source" | "cost";
type Pane = "hierarchy" | "canvas" | "details";

export default function Studio({ snapshot }: { snapshot: StructureSnapshot }) {
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const focusPage = /\/studio\/?$/.test(location.pathname);
  const popHref = `/models/${snapshot.model_id}/studio?${params.toString()}`;
  const [initial] = useState(() => new URLSearchParams(params));
  const result = snapshot.analysis as unknown as Analysis;
  const analysisId = snapshot.id;
  const analysisRequest = useAnalysisRequest(snapshot.model_id);
  const [view, setView] = useState<View>(
    ["structure", "ledger", "source", "cost"].includes(initial.get("studio_view") || "")
      ? initial.get("studio_view") as View : "structure");
  const [groupId, setGroupId] = useState(() => {
    const fromUrl = result.groups.find(g => g.id === initial.get("group"))?.id;
    if (fromUrl) return fromUrl;
    const moeGroup = result.groups.find(g => isMoeFfn(result.templates.find(t => t.id === g.template_id)?.ffn_kind));
    return moeGroup?.id || result.groups[0]?.id || "";
  });
  const [branchId, setBranchId] = useState(initial.get("branch") || "");
  const [selectedId, setSelectedId] = useState(initial.get("module") || "");
  const [sharedId, setSharedId] = useState(initial.get("shared") || "");
  const [pane, setPane] = useState<Pane>(initial.get("module") ? "details" : "canvas");
  const [scope, setScope] = useState<GraphScope>("all");
  const [aggregate, setAggregate] = useState(false);
  const [search, setSearch] = useState("");
  const [instance, setInstance] = useState("");
  const [locatedInstance, setLocatedInstance] = useState<number | null>(initial.has("layer") ? Number(initial.get("layer")) : null);
  const [instanceError, setInstanceError] = useState("");
  const [sequencePage, setSequencePage] = useState(0);
  const group = result?.groups.find(item => item.id === groupId);
  const branch = result?.branches?.find(item => item.id === branchId);
  const template = branch ?? result?.templates.find(item => item.id === group?.template_id);
  const node = template?.nodes.find(item => item.id === selectedId);
  const selected = sharedId ? result?.shared_modules.find(item => item.id === sharedId)
    : template?.modules.find(item => item.id === (node?.module_id ?? selectedId));
  const count = branch?.count ?? group?.count ?? 1;
  const multiplier = aggregate && !sharedId ? count : 1;

  useEffect(() => {
    const next = new URLSearchParams(initial);
    next.set("tab", "structure"); next.set("studio_view", view); next.set("snapshot", snapshot.id);
    next.set("kv_dtype", result.serving.kv_layout.dtype_assumed);
    next.set("mtp", String(result.serving.include_mtp));
    for (const key of ["repo", "revision", "group", "branch", "module", "shared", "layer"]) next.delete(key);
    if (groupId) next.set("group", groupId);
    if (branchId) next.set("branch", branchId);
    if (selectedId) next.set("module", selectedId);
    if (sharedId) next.set("shared", sharedId);
    if (locatedInstance !== null) next.set("layer", String(locatedInstance));
    if (next.toString() !== params.toString()) setParams(next, { replace: true });
  }, [view, groupId, branchId, selectedId, sharedId, locatedInstance, snapshot.id, initial, params, setParams, result]);

  function selectGroup(id: string) {
    setBranchId("");
    setGroupId(id); setSharedId(""); setSelectedId(""); setScope("all");
    setPane("canvas"); setSearch("");
    setLocatedInstance(null);
  }
  function selectModule(id: string) {
    setSharedId(""); setSelectedId(id);
    setPane("details");
  }
  function selectShared(id: string) {
    if (result?.branches?.some(item => item.id === id)) {
      setBranchId(id); setSharedId(""); setSelectedId(""); setScope("all"); setPane("canvas");
      return;
    }
    setBranchId("");
    setSharedId(id); setSelectedId(""); setPane("details");
  }
  function locateInstance() {
    const index = Number(instance);
    if (!instance.trim() || !Number.isInteger(index)) { setInstanceError("请输入整数层号"); return; }
    const target = result?.groups.find(item => item.ranges.some(([a, b]) => index >= a && index < b));
    if (!target) { setInstanceError("层号不在模型范围内"); return; }
    selectGroup(target.id); setInstanceError("");
    setLocatedInstance(index);
  }
  function exportJSON() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `${result.identity.name.split("/").pop()}-analysis-v2.json`;
    anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="studio-content">
      <div className="studio-meta">
        <div className="studio-meta-main">
          <div className="identity-meta"><span>{result.identity.model_type}</span><span className="support-status"><ShieldCheck size={14} />{result.status}</span>
            <span className="mono" title={result.source.resolved_sha ?? "local config"}>{result.source.resolved_sha?.slice(0, 12) || "local config"}</span></div>
          {result.morphology && <div className="morphology-tags" data-testid="morphology-tags" aria-label="形态标签">
            {result.morphology.attn.map(kind => <span key={`attn-${kind}`}>{kind}</span>)}
            {result.morphology.ffn.map(kind => <span key={`ffn-${kind}`}>{kind}</span>)}
            {result.morphology.vision !== "none" && <span>{result.morphology.vision}</span>}
            {result.morphology.quant !== "none" && <span>{result.morphology.quant}</span>}
          </div>}
        </div>
        <div className="studio-meta-actions">
          {focusPage ? null : <a className="studio-popout" href={popHref} target="planner-studio" rel="noreferrer"
            aria-label="在独立页面打开结构图">独立页面<SquareArrowOutUpRight size={14} /></a>}
          <button className="icon-button" title="导出分析 JSON" aria-label="导出分析 JSON" onClick={exportJSON}><Download size={18} /></button>
        </div>
      </div>
      <section className="metrics-strip" aria-label="模型指标">
        <Metric label="总参数" value={fmtParams(result.totals.total)} exact={fmtInt(result.totals.total)} />
        <Metric label="参与计算的参数" value={fmtParams(result.totals.active)} exact={fmtInt(result.totals.active)} />
        <Metric label="权重字节" value={`${(result.serving.weight_bytes / 1024 ** 3).toFixed(3)} GiB`} exact={result.serving.weight_bytes_source} />
        <Metric label={result.serving.kv_layout.bytes_per_token == null ? `Cache @ ${result.serving.kv_layout.reference_context ?? "S"}` : "KV / token"}
          value={`${fmtInt(result.serving.kv_layout.bytes_per_token ?? result.serving.kv_layout.reference_bytes ?? 0)} B`}
          exact={result.serving.kv_layout.dtype_assumed} />
        <Metric label="Decode GEMM / token" value={fmtFlop(result.serving.flop_gemm_decode)} exact="静态矩阵计算量" />
      </section>
      <nav className="tabs" aria-label="分析视图">
        {([["structure", "结构", ListTree], ["ledger", "参数账本", Layers3], ["source", "来源与审计", Braces], ["cost", "算子成本", Activity]] as const).map(([key, label, Icon]) =>
          <button key={key} aria-current={view === key ? "page" : undefined} onClick={() => setView(key)}><Icon size={16} />{label}</button>)}
        <span className="structure-count">{result.backbone.num_layers} 层 / {result.groups.length} 组</span>
      </nav>
      {view === "structure" && <>
        <nav className="pane-switch" aria-label="工作区视图">
          {([["hierarchy", "层级"], ["canvas", "画布"], ["details", "详情"]] as const).map(([key, label]) =>
            <button key={key} aria-pressed={pane === key} onClick={() => setPane(key)}>{label}</button>)}
        </nav>
        <div className={`workspace pane-${pane}`}>
          <aside className="hierarchy">
            <h2>模型层级</h2>
            <div className="dimensions"><span>hidden <b>{fmtInt(result.backbone.hidden_size)}</b></span>
              <span>Q / KV heads <b>{result.backbone.num_heads} / {result.backbone.num_kv_heads}</b></span></div>
            {result.shared_modules.filter(item => /embed/i.test(item.id)).map(item =>
              <SharedButton key={item.id} module={item} active={sharedId === item.id} onSelect={selectShared} />)}
            <div className="group-list">
              {result.groups.map(item => {
                const spec = result.templates.find(candidate => candidate.id === item.template_id)!;
                return <button className={`group-button ${groupId === item.id && !sharedId ? "active" : ""}`}
                  data-testid="layer-group" key={item.id} onClick={() => selectGroup(item.id)}>
                  <span className="group-title"><span>{spec.attn_kind} + {spec.ffn_kind}</span><b>×{item.count}</b></span>
                  <span className="group-ffn">{isMoeFfn(spec.ffn_kind) ? ffnRecipe(spec) : (result.groups.some(g => isMoeFfn(result.templates.find(t => t.id === g.template_id)?.ffn_kind)) ? "Dense SwiGLU · prefix" : "Dense SwiGLU")}</span>
                  <span className="group-range">{rangeLabel(item.ranges)}</span>
                  <span className="group-total mono">{fmtParams(item.metrics.params_total)} params</span>
                </button>;
              })}
            </div>
            {result.shared_modules.filter(item => !/embed/i.test(item.id)).map(item =>
              <SharedButton key={item.id} module={item} active={sharedId === item.id} onSelect={selectShared} />)}
            <form className="instance-form" onSubmit={event => { event.preventDefault(); locateInstance(); }}>
              <label htmlFor="layer-index">层实例</label>
              <div><input id="layer-index" type="number" min="0" max={result.backbone.num_layers - 1}
                value={instance} onChange={event => setInstance(event.target.value)} placeholder="0" />
                <button title="定位层实例" aria-label="定位层实例"><ArrowRight size={16} /></button></div>
              {instanceError && <span role="alert">{instanceError}</span>}
            </form>
            <details className="execution"><summary>执行顺序 · {result.execution.length} 段</summary>
              {result.execution.slice(sequencePage * 20, sequencePage * 20 + 20).map((segment, index) =>
                <button key={`${segment.start}-${index}`} onClick={() => selectGroup(segment.group_id)}>
                  <span>L{segment.start}–{segment.stop_exclusive - 1}</span><span>{segment.group_id}</span>
                </button>)}
              {result.execution.length > 20 && <div className="pagination">
                <button aria-label="上一页执行段" disabled={sequencePage === 0} onClick={() => setSequencePage(sequencePage - 1)}><ChevronLeft size={16} /></button>
                <span>{sequencePage + 1}/{Math.ceil(result.execution.length / 20)}</span>
                <button aria-label="下一页执行段" disabled={(sequencePage + 1) * 20 >= result.execution.length} onClick={() => setSequencePage(sequencePage + 1)}><ChevronRight size={16} /></button>
              </div>}
            </details>
            {analysisId && <Instances key={analysisId} analysisId={analysisId} analysisRequest={analysisRequest}
              onSelect={(id, index) => { selectGroup(id); setLocatedInstance(index); setInstance(String(index)); }} />}
          </aside>
          <main className="canvas-pane">
            <div className="canvas-heading"><div><span className="breadcrumb">模型 / {branch ? `${branch.label} (${branch.included ? "计入" : "未计入总量"})` : group?.id ?? "共享模块"}{!branch && locatedInstance !== null ? ` / L${locatedInstance}` : ""}</span>
              <h2>{template?.label ?? "模型结构"}</h2></div>
              <div className="segmented" aria-label="核算范围">
                <button aria-pressed={!aggregate} onClick={() => setAggregate(false)}>单层</button>
                <button aria-pressed={aggregate} onClick={() => setAggregate(true)}>本组 ×{count}</button>
              </div>
            </div>
            <div className="scope-bar" aria-label="算子范围">
              {([
                ["all", "完整层"],
                ["attention", "Attention"],
                ["ffn", "FFN"],
                ...(template?.role === "vision" || template?.attn_kind === "VISION" ? [["vision", "Vision"]] : []),
              ] as [GraphScope, string][]).map(([key, text]) =>
                <button key={key} aria-pressed={scope === key} onClick={() => setScope(key)}>{text}</button>)}
              <span>{template?.nodes.length ?? 0} 个算子</span>
            </div>
            {template && <p className="ffn-recipe" data-testid="ffn-recipe">{template.attn_kind} · {ffnRecipe(template)}</p>}
            {template && <LayerFlow template={template} scope={scope} selectedId={selectedId}
              multiplier={aggregate ? count : 1} onSelect={selectModule} />}
            {template && <section className="module-section">
              <div className="section-toolbar"><h2>{scope === "all" ? "模块账本" : scope === "attention" ? "Attention 参数账本" : "FFN / MLP 参数账本"}</h2><label><Search size={14} /><input aria-label="搜索模块"
                value={search} onChange={event => setSearch(event.target.value)} placeholder="q_proj / expert" /></label></div>
              <ModuleTable modules={template.modules.filter(item => (scope === "all" || categoryTone(item.category) === scope)
                && `${item.id} ${item.label}`.toLowerCase().includes(search.toLowerCase()))}
                multiplier={aggregate ? count : 1} selectedId={selected?.id || ""} onSelect={selectModule}
                moe={moeLayout(template)} />
            </section>}
          </main>
          <aside className="detail-pane" aria-label="模块详情">
            <div className="detail-heading"><h2>详情</h2><button className="detail-close icon-button" aria-label="关闭详情" onClick={() => setPane("canvas")}><X size={18} /></button></div>
            {selected ? <ModuleDetails module={selected} multiplier={multiplier} />
              : node && template ? <NodeDetails node={node} template={template} onSelect={selectModule} />
              : template && <TemplateDetails template={template} count={count} />}
            <div className="kv-details"><h3>KV 布局</h3><p className="formula">{result.serving.kv_layout.formula}</p>
              <p>{result.serving.kv_layout.bytes_per_token == null
                ? `${fmtInt(result.serving.kv_layout.reference_bytes ?? 0)} B @ ${result.serving.kv_layout.reference_context} tokens`
                : `${fmtInt(result.serving.kv_layout.bytes_per_token)} B/token`} · {result.serving.kv_layout.dtype_assumed}</p>
              <p>{result.serving.kv_layout.notes}</p></div>
          </aside>
        </div>
      </>}
      {view === "ledger" && <main className="full-view">
        <h2>模型参数账本</h2>
        <div className="table-scroll"><table><thead><tr><th>组成</th><th>数量</th><th>总参数</th><th>参与计算的参数</th></tr></thead><tbody>
          {result.shared_modules.map(item => <tr key={item.id}><td><button className="text-button" onClick={() => {
            selectShared(item.id); setView("structure");
          }}>{item.label}</button>{item.shared_with && <span className="shared-note">共享 {item.shared_with}</span>}</td>
            <td>{item.count ?? 1}</td><td className="mono">{fmtInt(item.params)}</td><td className="mono">{fmtInt(item.active_params)}</td></tr>)}
          {result.groups.map(item => <tr key={item.id}><td><button className="text-button" onClick={() => {
            selectGroup(item.id); setView("structure");
          }}>{result.templates.find(template => template.id === item.template_id)?.label}</button></td>
            <td>{item.count}</td><td className="mono">{fmtInt(item.metrics.params_total)}</td><td className="mono">{fmtInt(item.metrics.params_active)}</td></tr>)}
        </tbody><tfoot><tr><th>模型合计</th><td /><td className="mono">{fmtInt(result.totals.total)}</td><td className="mono">{fmtInt(result.totals.active)}</td></tr></tfoot></table></div>
        <section className="byte-ledger"><h2>权重口径</h2><dl>
          <dt>Physical / estimated bytes</dt><dd className="mono">{fmtInt(result.serving.weight_bytes)} B</dd>
          <dt>来源</dt><dd>{result.serving.weight_bytes_source}</dd>
          <dt>Logical bytes</dt><dd className="mono">{fmtInt(result.serving.weight_bytes_logical)} B</dd>
        </dl></section>
      </main>}
      {view === "source" && <SourceView result={result} />}
      {view === "cost" && <CostView key={analysisId} analysisId={analysisId} analysisRequest={analysisRequest} />}
      <footer><span>Analysis V{result.schema_version} · formula {result.formula_version}</span>
        <span><Check size={13} />{result.source.kind}</span></footer>
  </div>;
}

function Metric({ label, value, exact }: { label: string; value: string; exact: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><span className="mono">{exact}</span></div>;
}

function SharedButton({ module, active, onSelect }: { module: ModuleSpec; active: boolean; onSelect: (id: string) => void }) {
  return <button className={`shared-button ${active ? "active" : ""}`} onClick={() => onSelect(module.id)}>
    <span>{module.label}</span><span className="mono">{module.shared_with ? "shared · 0" : fmtParams(module.params)}</span>
  </button>;
}

function moduleCopies(module: ModuleSpec, moe: ReturnType<typeof moeLayout>): string {
  const count = module.count ?? 1;
  if (count > 1) return `×${count}`;
  if (module.id.includes("shared_expert.") && moe?.n_shared)
    return moe.n_shared > 1 ? `1 · ${moe.n_shared}×W` : "×1";
  return "1";
}

function ModuleTable({ modules, multiplier, selectedId, onSelect, moe }: {
  modules: ModuleSpec[]; multiplier: number; selectedId: string; onSelect: (id: string) => void;
  moe: ReturnType<typeof moeLayout>;
}) {
  return <div className="table-scroll"><table className="module-table"><thead><tr>
    <th>模块</th><th>份数</th><th>单份形状</th><th>参数</th></tr></thead><tbody>
    {modules.map(module => {
      const count = module.count ?? 1;
      return <tr key={module.id} className={selectedId === module.id ? "selected" : ""}>
        <td><button className="text-button" onClick={() => onSelect(module.id)}>{module.label}</button>
          {count > 1 && <span className="module-count">×{count}</span>}</td>
        <td className="mono">{moduleCopies(module, moe)}</td>
        <td className="mono">{module.shape.join(" × ") || "—"}</td>
        <td className="mono">{count > 1
          ? `${fmtInt(module.params / count)} × ${count} = ${fmtInt(module.params * multiplier)}`
          : fmtInt(module.params * multiplier)}</td>
      </tr>;
    })}
    {!modules.length && <tr><td colSpan={4}>无匹配模块</td></tr>}
  </tbody></table></div>;
}

function ModuleDetails({ module, multiplier }: { module: ModuleSpec; multiplier: number }) {
  return <div className="detail-content" data-testid="module-detail">
    <span className={`category-label ${categoryTone(module.category)}`}>{module.category}</span><h3>{module.label}</h3>
    <dl><dt>权重形状 [out, in]</dt><dd className="mono">{module.shape.join(" × ") || "—"}</dd>
      <dt>输入</dt><dd className="mono">{module.input_shape}</dd><dt>输出</dt><dd className="mono">{module.output_shape}</dd>
      <dt>份数</dt><dd className="mono">{module.count ?? 1}{(module.count ?? 1) > 1 ? "（routed experts）" : ""}</dd>
      <dt>参数 {multiplier > 1 ? `(×${multiplier} 层)` : ""}</dt><dd className="mono">{
        (module.count ?? 1) > 1
          ? `${fmtInt(module.params / (module.count ?? 1))} × ${module.count} = ${fmtInt(module.params * multiplier)}`
          : fmtInt(module.params * multiplier)
      }</dd>
      <dt>参与计算的参数（非激活张量）</dt><dd className="mono">{fmtInt(module.active_params * multiplier)}</dd>
      <dt>矩阵参数</dt><dd className="mono">{fmtInt(module.matrix_params * multiplier)}</dd>
      <dt>dtype</dt><dd>{module.dtype}</dd>
      {module.shared_with && <><dt>共享权重</dt><dd>{module.shared_with}；新增唯一参数为 0，不代表没有使用权重。</dd></>}
    </dl><h3>公式</h3><p className="formula">{module.formula}</p>
    <h3>依据</h3><p>{module.source}</p>{module.notes && <p className="module-notes">{module.notes}</p>}
  </div>;
}

function TemplateDetails({ template, count }: { template: Template; count: number }) {
  const moe = moeLayout(template);
  return <div className="detail-content"><span className="category-label">Layer template</span><h3>{template.label}</h3>
    <p className="ffn-recipe-inline">{ffnRecipe(template)}</p>
    <dl><dt>实例数量</dt><dd>{count}</dd><dt>单层参数</dt><dd className="mono">{fmtInt(template.metrics.params_total)}</dd>
      <dt>单层参与计算的参数</dt><dd className="mono">{fmtInt(template.metrics.params_active)}</dd>
      <dt>模块数</dt><dd>{template.modules.length}</dd>
      {moe && <>
        <dt>Routed experts</dt><dd className="mono">{moe.n_routed} · top-k {moe.top_k ?? "—"} · W={moe.expert_dim}</dd>
        <dt>Shared expert</dt><dd className="mono">{moe.n_shared
          ? `${moe.n_shared}×W = ${moe.shared_width}${moe.has_shared_expert_gate ? " · gate d→1" : ""}`
          : "无"}</dd>
      </>}
    </dl>
    <h3>结构签名</h3><p className="mono signature">{template.signature}</p></div>;
}

function SourceView({ result }: { result: Analysis }) {
  const query = useStudioGlossary();
  const glossary = query.data ?? {};
  const [cap, setCap] = useState(result.capabilities[0]?.id ?? "");
  const entry = glossary[cap];
  return <main className="full-view source-view"><section><h2>不可变来源</h2><dl>
    <dt>Repository</dt><dd>{result.source.repo_id || result.identity.name}</dd>
    <dt>Requested revision</dt><dd className="mono">{result.source.requested_revision || "—"}</dd>
    <dt>Resolved SHA</dt><dd className="mono">{result.source.resolved_sha || "local config"}</dd>
    <dt>Config SHA256</dt><dd className="mono">{result.source.config_sha256}</dd>
    <dt>获取时间</dt><dd>{result.source.fetched_at || "—"}</dd>
    <dt>Engine / Formula</dt><dd>{result.engine_version} / {result.formula_version}</dd>
  </dl>{result.source.repo_id && result.source.resolved_sha &&
    <a className="source-link" target="_blank" rel="noreferrer"
      href={`https://huggingface.co/${result.source.repo_id}/blob/${result.source.resolved_sha}/config.json`}>
      config.json<ExternalLink size={15} /></a>}</section>
    <section><h2>配置审计</h2>
      {result.diagnostics.length ? result.diagnostics.map((item, index) =>
        <div key={`${item.code}-${index}`} className={`diagnostic ${item.severity}`}><strong>{item.code}</strong><p>{item.message}</p></div>)
        : <p className="audit-clean"><ShieldCheck size={16} />无诊断项</p>}
    </section><section><h2>架构能力</h2>
      <div className="cap-list">{result.capabilities.map(item =>
        <button key={item.id} aria-pressed={cap === item.id} onClick={() => setCap(item.id)}>{item.label}</button>)}</div>
      {entry && <div className="glossary"><h3>{entry.title}</h3><p>{entry.summary}</p>{entry.formula && <p className="formula">{entry.formula}</p>}</div>}
      <pre>{JSON.stringify(result.capabilities.find(item => item.id === cap)?.evidence ?? [], null, 2)}</pre>
    </section></main>;
}
