import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Capabilities, DesignInput, Search, RankDetail, Memory } from "../../types/workload";
import { componentNames } from "./Inspector";
import { number } from "./state";
import { FieldName, Readout } from "./FieldName";

export function Numeric({ label, value, onChange, min = 1, max, step = 1 }: {
  label: string; value: number | null; onChange: (n: number) => void; min?: number; max?: number; step?: number;
}) {
  const display = (n: number | null) => n == null || !Number.isFinite(n) ? "" : String(n);
  const [text, setText] = useState(display(value));
  const written = useRef(value);
  useEffect(() => {
    if (!Object.is(value, written.current)) { setText(display(value)); written.current = value; }
  }, [value]);
  return <label><FieldName name={label} /><input aria-label={label} type="number" value={text}
    min={min} max={max} step={step} onChange={e => {
      setText(e.target.value);
      written.current = e.target.value === "" ? NaN : Number(e.target.value);
      onChange(written.current);
    }} /></label>;
}
function IntList({ label, value, defaults, onChange }: { label: string; value: number[] | null; defaults?: number[]; onChange: (v: number[] | null) => void }) {
  const [text, setText] = useState(value?.join(",") ?? defaults?.join(",") ?? "");
  const [error, setError] = useState("");
  const written = useRef(JSON.stringify(value));
  useEffect(() => {
    const incoming = JSON.stringify(value);
    if (incoming !== written.current) {
      setText(value?.join(",") ?? defaults?.join(",") ?? ""); setError(""); written.current = incoming;
    }
    if (value == null && defaults) setText(defaults.join(","));
  }, [value, defaults?.join(",")]);
  return <label><FieldName name={label} /><input aria-label={label} value={text} placeholder="自动；或 1,2,4" onChange={e => {
    const raw = e.target.value; setText(raw || defaults?.join(",") || "");
    const valid = !raw || /^\s*\d+(\s*,\s*\d+)*\s*$/.test(raw);
    setError(valid ? "" : "请输入逗号分隔的正整数");
    const next = valid ? raw ? raw.split(",").map(Number) : null : [NaN];
    written.current = JSON.stringify(next);
    onChange(next);
  }} />{defaults && <span className="muted">{value == null ? "Auto search range" : "Override · clear to reset"}</span>}
    {error && <span role="alert" className="bad">{error}</span>}</label>;
}
export function InputPanel({ input, change, capabilities, search, setSearch, memory, stale }: {
  input: DesignInput; change: (input: DesignInput) => void; capabilities?: Capabilities;
  search: Search; setSearch: (s: Search) => void; selectedLayer?: number;
  memory?: RankDetail | Memory; stale?: boolean;
}) {
  const [online, setOnline] = useState({ rate: input.load.request_rate ?? 1, arrival: input.load.arrival ?? "deterministic" });
  const [offline, setOffline] = useState(input.load.request_count ?? 100);
  const load = (patch: Partial<DesignInput["load"]>) => change({ ...input, load: { ...input.load, ...patch } });
  const runtime = (patch: Partial<DesignInput["runtime"]>) => change({ ...input, runtime: { ...input.runtime, ...patch } });
  const parallel = (patch: Partial<DesignInput["parallelism"]>) => change({ ...input, parallelism: { ...input.parallelism, ...patch } });
  const resource = (patch: Partial<DesignInput["resources"]>) => change({ ...input, resources: { ...input.resources, ...patch } });
  const manual = input.parallelism.manual_candidate;
  const layout = input.parallelism.layout;
  const legacy = layout === "orthogonal-tp-ep-v1";
  const vllm = layout === "vllm-v0.10.2";
  const spec = input.runtime.speculative;
  const k = spec.draft_tokens ?? 5, acceptance = spec.acceptance_rate ?? .75;
  const accepted = Array.from({ length: Math.max(0, Math.min(64, k || 0)) }, (_, i) => acceptance ** (i + 1)).reduce((a, b) => a + b, 0);
  const speculative = (patch: Partial<typeof spec>) => runtime({ speculative: { ...spec, ...patch } });
  const domains = capabilities?.parallel_domains_by_layout?.[layout] ?? capabilities?.parallel_domains;
  const layers = domains?.pp.length ?? 1;
  const isMoe = (domains?.ep ?? [1]).some(n => n > 1);
  const defaultBoundaries = (pp: number) => {
    if (!Number.isInteger(pp) || pp < 1 || pp > layers) return [];
    const sizes = Array.from({ length: pp }, () => Math.floor(layers / pp));
    for (let i = 0; i < layers % pp; i++) sizes[vllm ? pp - 2 - i : pp - 1]++;
    return sizes.slice(0, -1).map((_, i) => sizes.slice(0, i + 1).reduce((a, b) => a + b, 0));
  };
  const changeManual = (patch: Partial<NonNullable<typeof manual>>) => {
    if (!manual) return;
    const next = { ...manual, ...patch };
    if (patch.pp != null) next.boundaries = defaultBoundaries(patch.pp);
    if (vllm) next.ep = input.parallelism.expert_parallel ? next.tp * next.dp : 1;
    parallel({ manual_candidate: next });
  };
  const hw = capabilities?.hardware;
  const budget = input.resources.node_count * input.resources.gpus_per_node;
  const topology = capabilities?.topology_defaults[input.resources.topology_template] ?? {};
  const memoryPaths = ["memory_efficiency", "activation_efficiency", "kv_efficiency", "safety_fraction"];
  const commonPaths = ["compute_efficiency", "hbm_efficiency"];
  const precisionLabels = { weight_dtype: "Weight Precision", activation_dtype: "Activation Precision", kv_dtype: "KV Precision" };
  const assumption = (path: string) => {
    const spec = capabilities?.overrides[path];
    if (!spec) return null;
    const override = input.overrides.find(o => o.path === path);
    const label = ({ memory_efficiency: "Memory Efficiency", activation_efficiency: "Activation Efficiency",
      kv_efficiency: "KV Cache Efficiency", safety_fraction: "Safety Margin",
      compute_efficiency: "Compute Efficiency", hbm_efficiency: "HBM Efficiency" } as Record<string, string>)[path] ?? path;
    return <div className="wc-assumption" key={path}>
      <Numeric label={label} value={override?.value ?? spec.default} min={spec.minimum} max={spec.maximum}
        step={0.01} onChange={value => change({ ...input, overrides: [
          ...input.overrides.filter(o => o.path !== path), { path, value, reason: "用户指定情景" },
        ] })} />
      {override ? <button onClick={() => change({ ...input, overrides: input.overrides.filter(o => o.path !== path) })}>Override · Reset</button>
        : <span className="muted">Engine Default</span>}
    </div>;
  };
  const topologyField = (path: string, label: string, scale = 1) => {
    const override = input.resources.topology_overrides[path];
    const value = override ?? topology[path];
    return <div key={path} className="wc-assumption"><Numeric label={label} value={value == null ? null : value / scale}
      min={0} step={0.01} onChange={n => {
        const next = { ...input.resources.topology_overrides };
        if (Number.isFinite(n)) next[path] = n * scale; else delete next[path];
        resource({ topology_overrides: next });
      }} /><span className="muted">{override != null ? "Override" : value == null ? "Unavailable" : "Source Default"}</span></div>;
  };
  return <div className="wc-input">
    <details open><summary>Workload</summary>
      <div className="wc-fields">
      <label><FieldName name="Mode" /><select aria-label="Mode" value={input.mode} onChange={e => {
        const mode = e.target.value as DesignInput["mode"];
        if (input.mode === "online") setOnline({ rate: input.load.request_rate!, arrival: input.load.arrival! });
        else setOffline(input.load.request_count!);
        change({ ...input, mode, load: { ...input.load,
          request_rate: mode === "online" ? online.rate : null,
          arrival: mode === "online" ? online.arrival : null, request_count: mode === "offline" ? offline : null } });
      }}><option value="online">Online · 请求流</option><option value="offline">Offline · 批量任务</option></select></label>
      <Numeric label="Prompt Tokens" value={input.load.prompt_tokens} onChange={n => load({ prompt_tokens: n })} />
      <Numeric label="Output Tokens" value={input.load.output_tokens} onChange={n => load({ output_tokens: n })} />
      <Numeric label="Batch Size" value={input.runtime.max_batch_sequences} onChange={n => runtime({ max_batch_sequences: n })} />
      <Numeric label="Concurrency" value={input.load.concurrency} max={4096} onChange={n => load({ concurrency: n })} />
      {input.mode === "online" ? <>
        <Numeric label="Request Rate (req/s)" value={input.load.request_rate} step={0.1} min={0.001} onChange={n => load({ request_rate: n })} />
        <label><FieldName name="Arrival" /><select aria-label="Arrival" value={input.load.arrival!} onChange={e => load({ arrival: e.target.value as "poisson" | "deterministic" })}>
          <option value="deterministic">均匀</option><option value="poisson">Poisson</option></select></label>
      </> : <Numeric label="Request Count" value={input.load.request_count} onChange={n => load({ request_count: n })} />}
      <Numeric label="Prefill Chunk" value={input.runtime.prefill_chunk_tokens} onChange={n => runtime({ prefill_chunk_tokens: n })} />
      <label><FieldName name="Continuous Batching" /><select aria-label="Continuous Batching" value={String(input.runtime.continuous_batching)} onChange={e => runtime({
        continuous_batching: e.target.value === "true",
        profile: e.target.value === "true" ? "analytical-continuous-v1" : "analytical-sequential-v1",
      })}><option value="true">Enabled</option><option value="false">Disabled</option></select></label>
      </div>
      <div className="wc-spec-toggle"><label className="wc-check"><input type="checkbox" checked={spec.enabled}
        onChange={e => speculative({ enabled: e.target.checked })} />Speculative Decoding</label>
        <span className="muted">Independent cycle analysis</span></div>
      {spec.enabled && <div className="wc-spec-fields"><div className="wc-fields">
        <Numeric label="Draft Tokens" value={k} max={64} onChange={n => speculative({ draft_tokens: n })} />
        <Numeric label="Acceptance Rate" value={acceptance} min={0} max={1} step={.01} onChange={n => speculative({ acceptance_rate: n })} />
        <Readout label="Expected Accepted Tokens" value={number(accepted)} />
        <Readout label="Expected Emitted Tokens" value={number(1 + accepted)} />
        <Numeric label="Draft Latency (ms)" value={spec.draft_latency_ms ?? null} min={0} step={.01}
          onChange={n => speculative({ draft_latency_ms: Number.isFinite(n) ? n : null })} />
        <Numeric label="Draft Memory (GB)" value={spec.draft_memory_bytes == null ? null : spec.draft_memory_bytes / 1e9} min={0} step={.01}
          onChange={n => speculative({ draft_memory_bytes: Number.isFinite(n) ? Math.round(n * 1e9) : null })} />
      </div><label className="wc-check"><input type="checkbox" checked={spec.include_prefill ?? false}
        onChange={e => speculative({ include_prefill: e.target.checked })} />Include one Prefill Chunk in cycle</label>
        <p className="muted">期望接受值由 K / a 推导。Draft 开销留空表示未知；填 0 为乐观假设。完整请求指标仍显示非投机基线。</p>
      </div>}
      <details className="wc-advanced"><summary>Runtime Policy & Measurement Window</summary><div className="wc-fields">
      <Readout label="KV Cache Policy" value="Paged · 16 tokens" hint="当前分析策略固定使用16-token页，已计入尾页和页索引，不是可修改的来源参数。" />
      <Numeric label="Duration (s)" value={input.load.duration_seconds} max={3600} onChange={n => load({ duration_seconds: n })} />
      <Numeric label="Warmup (s)" value={input.load.warmup_seconds} min={0} onChange={n => load({ warmup_seconds: n })} />
      <Numeric label="Seed" value={input.load.seed} min={0} onChange={n => load({ seed: n })} />
      </div></details>
    </details>
    <details open><summary>Resources</summary>
      {hw && <div className="wc-hardware-facts">
        {[["Dense TFLOPS", hw.compute[input.runtime.activation_dtype]?.dense],
          ["HBM (GB)", hw.memory_bytes / 1e9], ["Memory BW (GB/s)", hw.memory_bandwidth_bytes_s / 1e9],
          ["NVLink (GB/s, spec)", hw.nvlink_gb_s], ["NIC (GB/s/GPU)", hw.node.nic_gb_s_per_gpu],
          ["PCIe (GB/s, spec)", hw.interconnect.find(l => /pcie|pci express/i.test(l.name))?.bandwidth_gb_s],
          ["CNY / day / chip", hw.cost.cny_per_day]].map(([label, value]) =>
          <div key={String(label)}><span>{label}</span><strong>{number(value as number | null | undefined)}</strong></div>)}
      <Link to={`/hardware/${hw.id}`}>Roofline & Sources ↗</Link>
      </div>}
      <div className="wc-fields wc-resource-inputs">
        {topologyField("intra_bandwidth_bytes_s", "Intra Injection (GB/s)", 1e9)}
        {topologyField("inter_bandwidth_bytes_s", "Inter / GPU (GB/s)", 1e9)}
      </div>
      <details className="wc-advanced"><summary>Topology Overrides & Source Notes</summary><div className="wc-fields">
        <label><FieldName name="Topology" /><select aria-label="Topology" value={input.resources.topology_template} onChange={e => resource({ topology_template: e.target.value })}>
          {(capabilities?.topology_templates ?? [input.resources.topology_template]).map(t => <option key={t}>{t}</option>)}</select></label>
        {topologyField("nic_count", "NIC Count")}
        {topologyField("nic_bandwidth_bytes_s", "NIC Port (GB/s)", 1e9)}
        {["oversubscription", "link_efficiency", "inter_link_efficiency", "intra_latency_us", "inter_latency_us"].map(path => topologyField(path, path))}
      </div><div className="wc-form-actions"><button onClick={() => resource({ topology_overrides: {} })}>恢复拓扑默认值</button></div>
      <p className="muted">SI units: 1 GB/s = 10⁹ bytes/s. 双向链路规格转换为单向注入上限。NIC 端口数量和带宽需成对覆盖。
        价格日期 {hw?.cost.as_of ?? "Unavailable"}；参考报价不直接用于 token 成本。
        {hw && <Link to={`/hardware/${hw.id}`}> Roofline & Sources ↗</Link>}</p></details>
    </details>
    <details open><summary>Parallelism</summary>
      <div className="wc-fields">
      <label><FieldName name="Framework" hint="按固定版本转换框架参数、设备分组和默认PP分层。Legacy保留旧运行语义。" />
        <select aria-label="Framework" value={layout} onChange={e => parallel({
          layout: e.target.value as typeof layout, expert_parallel: false, allowed_ep: null,
          allowed_tp: null, allowed_pp: null, layer_locks: [], pp_boundaries: null,
          ...(manual ? { manual_candidate: { tp: 1, pp: 1, ep: 1, dp: 1, boundaries: [] } } : {}),
        })}><option value="vllm-v0.10.2">vLLM · v0.10.2</option><option value="sglang-v0.5.3">SGLang · v0.5.3</option>
          <option value="orthogonal-tp-ep-v1">Legacy · Orthogonal</option></select></label>
      <label><FieldName name="Search Mode" /><select aria-label="Search Mode" value={input.parallelism.mode} onChange={e => parallel(e.target.value === "auto"
        ? { mode: "auto", manual_candidate: null } : { mode: "manual", manual_candidate: { tp: 1, pp: 1, ep: 1, dp: 1, boundaries: [] } })}>
        <option value="auto">Auto Design</option><option value="manual">Manual</option></select></label>
      {manual ? <>
        {(["tp", "pp", "dp"] as const).map(key =>
          <Numeric label={key.toUpperCase()} key={key} value={manual[key]} max={256}
            onChange={n => changeManual({ [key]: n })} />)}
        {!vllm && <Numeric label="EP" value={manual.ep} max={256} onChange={n => changeManual({ ep: n })} />}
        {manual.pp > 1 && <IntList key={`manual-${input.model_id}`} label="PP Boundaries" value={manual.boundaries}
          onChange={v => parallel({ manual_candidate: { ...manual, boundaries: v ?? defaultBoundaries(manual.pp) } })} />}
      </> : <>
        {(vllm ? ["tp", "pp"] as const : ["tp", "pp", "ep"] as const).map(key => <IntList key={`${input.model_id}-${layout}-${key}`}
          label={`Allowed ${key.toUpperCase()}`} value={input.parallelism[`allowed_${key}`]}
          defaults={(domains?.[key] ?? [1]).filter(n => n <= budget)}
          onChange={v => parallel({ [`allowed_${key}`]: v })} />)}
        <label><FieldName name="DP" /><select aria-label="DP" value={String(input.parallelism.dp_scaling)} onChange={e => parallel({ dp_scaling: e.target.value === "true" })}>
          <option value="false">1</option><option value="true">Auto · up to {budget}</option></select></label>
      </>}
      <Readout label="GPU Budget" value={manual ? `${manual.tp * manual.pp * manual.dp * (legacy ? manual.ep : 1)} / ${budget}` : budget}
        hint={legacy ? "Legacy使用TP×PP×EP×DP。" : "框架模式使用TP×PP×DP，EP复用这些GPU。"} />
      <Numeric label="Nodes" value={input.resources.node_count} max={256} onChange={n => resource({ node_count: n })} />
      <Numeric label="GPUs / Node" value={input.resources.gpus_per_node} max={256} onChange={n => resource({ gpus_per_node: n })} />
      </div>
      {vllm && isMoe && <div className="wc-spec-toggle"><label className="wc-check"><input type="checkbox"
        checked={input.parallelism.expert_parallel ?? (manual ? manual.ep > 1 : false)}
        onChange={e => parallel({ expert_parallel: e.target.checked, ...(manual ? {
          manual_candidate: { ...manual, ep: e.target.checked ? manual.tp * manual.dp : 1 },
        } : {}) })} />Expert Parallel</label>
        <span className="muted">EP = TP × DP when enabled · Expert TP = 1</span></div>}
      <p className="wc-parallel-rule">{legacy ? "GPU = TP × PP × EP × DP · Expert TP = TP"
        : vllm ? "GPU = TP × PP × DP · Attention TP = TP · EP shares TP × DP devices"
          : "GPU = TP × PP × DP · Expert TP = TP ÷ EP"}</p>
      {manual && <p className="muted">Pipeline layers: {[0, ...manual.boundaries].map((start, i) =>
        `Stage ${i}: ${start}–${(manual.boundaries[i] ?? layers) - 1}`).join(" · ")}</p>}
      <details className="wc-advanced"><summary>Pipeline Overrides & Legacy Constraints</summary>
        {!manual && <div className="wc-fields"><IntList key={`auto-${input.model_id}`} label="PP Boundaries"
          value={input.parallelism.pp_boundaries} onChange={v => parallel({ pp_boundaries: v })} /></div>}
        <p className="muted">TP作用于所有层的张量；PP才分配连续层区间。旧Layer锁仅限制全局TP集合，不表示每层不同TP。</p>
        {input.parallelism.layer_locks.map(lock => <div key={lock.layer_index} className="wc-form-actions">
          <span>Layer {lock.layer_index} · Allowed TP {lock.allowed_tp.join(",")}</span>
          <button onClick={() => parallel({ layer_locks: input.parallelism.layer_locks.filter(l => l !== lock) })}>Remove constraint</button>
        </div>)}
      </details>
    </details>
    <details open><summary>Memory {memory && `· GPU ${"rank" in memory ? memory.rank : memory.worst_rank}`}{stale ? " · Stale Result" : ""}</summary>
      <div className="wc-fields">{memoryPaths.map(assumption)}</div>
      <details className="wc-advanced"><summary>Result Reference · Memory</summary>
      <div className="wc-hardware-facts wc-memory-facts">
        {Object.entries(componentNames).map(([key, label]) => <div key={key}><span>{label}</span>
          <strong>{memory ? `${number(memory.components[key as keyof typeof componentNames] / 1e9)} GB` : "—"}</strong></div>)}
        <div><span>Total / HBM</span><strong>{memory ? `${number(memory.peak_bytes / 1e9)} / ${number(memory.capacity_bytes / 1e9)} GB` : "—"}</strong></div>
        <div><span>Utilization</span><strong>{memory?.explanation ? `${number(memory.explanation.utilization * 100)}%` : "—"}</strong></div>
      </div></details>
    </details>
    <details open><summary>Performance</summary><div className="wc-fields">
      <label><FieldName name="Scheduler" /><select aria-label="Scheduler" value={input.runtime.profile} onChange={e => runtime({
        profile: e.target.value, continuous_batching: e.target.value === "analytical-continuous-v1",
      })}>{(capabilities?.runtime_profiles ?? [input.runtime.profile]).map(p => <option key={p}>{p}</option>)}</select></label>
      {(["weight_dtype", "activation_dtype", "kv_dtype"] as const).map(key =>
        <label key={key}><FieldName name={precisionLabels[key]} /><select aria-label={precisionLabels[key]} value={input.runtime[key]} onChange={e => runtime({ [key]: e.target.value })}>
          <option value="bf16">BF16</option><option value="fp32">FP32</option></select></label>)}
      {commonPaths.map(assumption)}
      </div><details className="wc-advanced"><summary>Runtime & Precision</summary><div className="wc-fields">
      <Numeric label="Batch tokens" value={input.runtime.max_batched_tokens} onChange={n => runtime({ max_batched_tokens: n })} />
      <Numeric label="Fairness Interval" value={input.runtime.fairness_interval} max={100} onChange={n => runtime({ fairness_interval: n })} />
      </div></details>
    <details className="wc-advanced"><summary>SLA & Search Budget</summary><div className="wc-fields">
      {(["ttft_ms", "itl_ms"] as const).map(key => <Numeric key={key} label={key} value={input.targets[key]}
        onChange={n => change({ ...input, targets: { ...input.targets, [key]: Number.isFinite(n) ? n : null } })} />)}
      <label><FieldName name="SLA Policy" /><select aria-label="SLA Policy" value={input.targets.policy} onChange={e =>
        change({ ...input, targets: { ...input.targets, policy: e.target.value as "soft" | "hard" } })}>
        <option value="soft">Soft · 排序参考</option><option value="hard">Hard · 必须满足</option></select></label>
      {(["top_k", "max_candidates", "max_seconds", "event_budget"] as const).map(key =>
        <Numeric key={key} label={key} value={search[key]} onChange={n => setSearch({ ...search, [key]: n })} />)}
    </div></details>
    <details className="wc-advanced"><summary>Runtime Assumptions</summary><div className="wc-fields">
      {Object.entries(capabilities?.overrides ?? {}).filter(([path]) => ![...memoryPaths, ...commonPaths].includes(path)).map(([path, spec]) => {
        const override = input.overrides.find(o => o.path === path);
        const update = (value: number, reason = override?.reason ?? "用户指定情景") =>
          change({ ...input, overrides: [...input.overrides.filter(o => o.path !== path), { path, value, reason }] });
        return <div className="wc-override" key={path}>
          <Numeric label={path} value={override?.value ?? spec.default} min={spec.minimum} max={spec.maximum}
            step={path.endsWith("_bytes") ? 1 : 0.01} onChange={n => update(n)} />
          <span className="muted">默认 {spec.default} · {override ? "已覆盖" : "默认生效"}</span>
          {override && <><label>Override Reason<input value={override.reason} onChange={e => update(override.value, e.target.value)} /></label>
            <button onClick={() => change({ ...input, overrides: input.overrides.filter(o => o.path !== path) })}>恢复 {path}</button></>}
        </div>;
      })}
    </div></details></details>
  </div>;
}
