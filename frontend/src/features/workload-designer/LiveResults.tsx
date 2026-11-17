import { Fragment } from "react";
import type { Plan, RankDetail, Run, Trace } from "../../types/workload";
import { componentNames } from "./MemorySummary";
import { CalculationTrace, Facts, factValue } from "./CalculationTrace";
import { RawDetails } from "./Evidence";
import { calculationContext, metricNames, validateCalculation } from "./calculationEvidence";
import { bytes, number } from "./state";

export function LiveResults({ plan, rank, ranks = [], run, stale }: {
  plan?: Plan; rank?: RankDetail; ranks?: RankDetail[]; run?: Run; stale: boolean;
}) {
  if (!plan) return <div className="wc-results" aria-label="Live Results"><p>保存并运行后显示计算结果。</p></div>;
  const metrics = plan.summary.metric_values, raw = plan.summary.performance?.metrics;
  const memory = rank ?? plan.summary.memory, e = plan.summary.explanation, performance = e?.performance;
  const cycle = plan.summary.speculative_cycle, cycleMemory = cycle?.memory.find(r => r.rank === (rank?.rank ?? 0));
  const context = (key: string) => factValue(e?.context.find(f => f.key === key)?.value);
  const metric = (key: string) => metrics[key]?.value == null ? "Unavailable" : factValue(metrics[key].value, metrics[key].unit);
  const checks = validateCalculation(plan, ranks, run), failures = checks.filter(c => c.status === "fail");
  const memoryTrace = (key: string) => memory.explanation?.traces[key];
  const button = (label: string, value: string, trace?: Trace) =>
    <div className="wc-result-value"><span>{label}</span>
      <span className="wc-inline-value">{value}<CalculationTrace trace={trace} label={label} inline /></span>
    </div>;
  const rows = (entries: [string, string][]) => <dl className="wc-values">{entries.map(([label, key]) =>
    <div key={key}><dt>{label}</dt><dd><span className="wc-inline-value">{metric(key)}
      <CalculationTrace trace={e?.traces[key]} label={label} inline /></span></dd></div>)}</dl>;
  const iteration = metrics.iteration_time_ms?.value;
  const componentValues = performance?.breakdown.map(t => t.result.value)
    .filter((v): v is number => typeof v === "number" && Number.isFinite(v)) ?? [];
  const componentMax = iteration != null && componentValues.length ? Math.max(...componentValues) : null;
  const largestComponent = componentMax == null ? undefined : performance?.breakdown.find(t => t.result.value === componentMax);
  const partition = plan.summary.candidate;
  return <div className="wc-results" aria-label="Live Results">
    {failures.length > 0 && <section role="alert" className="wc-inconsistency"><h3>Calculation Inconsistency</h3>
      <p>结论已暂停：保存的结果存在内部矛盾。</p>{failures.map((f, i) => <p key={i}>{f.name} · {f.detail}</p>)}</section>}
    <section className="wc-execution-summary" aria-label="Decision Summary">
      <h3>Decision Summary</h3>
      <strong>{e ? context("model") : "Saved Workload"}</strong>
      <p>{e ? context("hardware") : "Saved hardware"} · <strong>{plan.summary.gpu_count} GPU</strong></p>
      <p>TP {partition.tp} / PP {partition.pp} / EP {partition.ep} / DP {partition.dp}</p>
      <p className={failures.length ? "bad" : ""}><strong>Feasibility · {failures.length ? "Calculation Inconsistency"
        : `Memory ${plan.summary.capacity_status} · Execution ${plan.summary.execution_status}`}</strong>
        {" · "}Targets {plan.summary.target_status}</p>
      <div className="wc-primary-metrics">
        {button("TTFT", metric("ttft_ms"), e?.traces.ttft_ms)}
        {button("ITL", metric("itl_ms"), e?.traces.itl_ms)}
        {button("Output Throughput", metric("output_tokens_s"), e?.traces.output_tokens_s)}
      </div>
      <div className="wc-memory-summary" aria-label="GPU Memory">
        {button(`Peak Memory · GPU ${rank?.rank ?? plan.summary.memory.worst_rank}`, `${bytes(memory.peak_bytes)} / ${bytes(memory.capacity_bytes)}`, memoryTrace("peak"))}
        <meter min={0} max={1} value={memory.explanation?.utilization ?? memory.peak_bytes / memory.capacity_bytes} aria-label="GPU Memory Utilization" />
        <p className="muted">Free {bytes(memory.free_bytes)} · Safety included once · Admission bound</p>
      </div>
      <p><strong>Current Bottleneck · {performance?.bottleneck ?? "Unavailable"}</strong></p>
    </section>
    <section aria-label="Why this plan"><h3>Why this plan?</h3>
      <div className="wc-plan-reasons">
        {button(largestComponent?.title ?? "Decode", componentMax == null ? "Unavailable" : `${number(componentMax)} ms / iteration`,
          largestComponent)}
        {(["weight", "kv", "activation", "communication", "runtime", "fragmentation", "safety"] as const)
          .map(key => <Fragment key={key}>{button(componentNames[key], bytes(memory.components[key]), memoryTrace(key))}</Fragment>)}
        {(["peak", "free", "utilization"] as const).map(key => {
          const trace = memoryTrace(key);
          const fallback = key === "peak" ? bytes(memory.peak_bytes) : key === "free" ? bytes(memory.free_bytes)
            : memory.explanation ? factValue(memory.explanation.utilization, "fraction") : "Unavailable";
          return <Fragment key={key}>{button(trace?.title ?? (key === "peak" ? "Admission Peak" : key === "free" ? "Free Memory" : "Memory Utilization"),
            trace ? factValue(trace.result.value, trace.result.unit) : fallback, trace)}</Fragment>;
        })}
      </div>
      <p className="muted">Context {run ? run.design_input.load.prompt_tokens + run.design_input.load.output_tokens : "—"} tokens
        {" · "}Batch {rank?.resident_request_quota ?? context("max_batch_sequences")}</p>
      {plan.summary.gpu_count === 1 && <p className="muted">No inter-GPU communication · TP=1 / PP=1 / EP=1 / DP=1</p>}
    </section>
    <section aria-label="Performance Breakdown"><h3>Decode Iteration · {metric("iteration_time_ms")}</h3>
      <div className="wc-performance-bars">{performance?.breakdown.map(t => {
        const value = typeof t.result.value === "number" && Number.isFinite(t.result.value) ? t.result.value : null;
        return <div className={`wc-performance-bar ${value === componentMax && componentMax != null && componentMax > 0 ? "is-bottleneck" : ""}`}
          key={t.id}>
          <span>{t.title}</span><span className="wc-bar-track"><span style={{ width: `${value != null && iteration && iteration > 0 ? Math.max(0, Math.min(100, value / iteration * 100)) : 0}%` }} /></span>
          <strong className="wc-inline-value">{factValue(t.result.value, t.result.unit)}<CalculationTrace trace={t} inline /></strong>
        </div>;
      })}</div>
      {!performance && <p>Unavailable in this historical result.</p>}
      <p className="muted">Selected compute or memory path per operator; runtime and communication are separate.
        Communication Bound combines TP, PP and EP. Click a component for evidence.</p>
    </section>
    <details><summary>Secondary Metrics</summary>
      {rows([["Prefill Token Rate · Window", "prefill_tokens_s"], ["Achieved QPS", "achieved_qps"],
        ["Window MFU", "mfu"], ["Modeled GPU Activity", "gpu_activity_fraction"],
        ["Output Tokens/s/GPU", "tokens_s_per_gpu"], ["Iteration Time", "iteration_time_ms"]])}
      <p className="muted">Window MFU uses matrix FLOPs. Modeled GPU Activity uses the union of compute and communication intervals.</p>
      <details><summary>Additional Metrics & Samples</summary>
        {rows(Object.keys(metrics).filter(k => !["ttft_ms", "itl_ms", "output_tokens_s", "prefill_tokens_s", "achieved_qps",
          "mfu", "gpu_activity_fraction", "tokens_s_per_gpu", "iteration_time_ms",
          "cost_per_million_output_tokens", "energy_per_million_output_tokens"].includes(k))
          .map(k => [metricNames[k] ?? k.replaceAll("_", " "), k]))}
        <dl className="wc-values">{Object.entries(raw?.percentiles ?? {}).map(([key, value]) =>
          <div key={key}><dt>End-to-End {key}</dt><dd>{factValue(value, "ms")}</dd></div>)}</dl>
        <p>Samples {raw?.percentile_sample_count ?? "—"} · {raw?.percentile_reason}</p>
        {run && <p>Concurrency {run.design_input.load.concurrency} · Seed {run.design_input.load.seed}
          {" · "}{run.design_input.load.arrival} · Warmup {run.design_input.load.warmup_seconds}s
          {" · "}Censored {String(raw?.censored_count ?? "—")}</p>}
      </details>
    </details>
    <details aria-label="Calculation Context"><summary>Calculation Context</summary><Facts items={calculationContext(plan)} />
      <p>Prompt {context("prompt_tokens")} · Output {context("output_tokens")} · Batch {context("max_batch_sequences")}
        {" · "}Concurrency {context("concurrency")}</p>
    </details>
    <details><summary>Advanced · Cost, Energy & Speculative</summary>
      {rows([["Cost / 1M Tokens", "cost_per_million_output_tokens"], ["Energy / 1M Tokens", "energy_per_million_output_tokens"]])}
      {cycle && <section aria-label="Speculative Cycle"><h3>Speculative Cycle · Fixed Context · {cycle.status}</h3>
        <p>Decode requests {cycle.decode_requests} · Context {cycle.context_tokens} · Prefill {cycle.prefill_tokens}</p>
        <dl className="wc-values">
          <div><dt>Draft Latency</dt><dd>{factValue(cycle.draft_seconds, "s")}
            {cycle.draft_seconds === 0 && <span className="wc-source-tag">Optimistic assumption</span>}</dd></div>
          <div><dt>Draft Memory</dt><dd>{bytes(cycleMemory?.draft_bytes)}
            {cycleMemory?.draft_bytes === 0 && <span className="wc-source-tag">Optimistic assumption</span>}</dd></div>
          <div><dt>Expected Accepted / Emitted</dt><dd>{number(cycle.expected_accepted_tokens)} / {number(cycle.expected_emitted_tokens)}</dd></div>
          <div><dt>Target Verify</dt><dd>{factValue(cycle.target_seconds, "s")}</dd></div>
          <div><dt>Cycle Time</dt><dd>{factValue(cycle.cycle_seconds, "s")}</dd></div>
          <div><dt>Throughput</dt><dd>{number(cycle.output_tokens_s)} tokens/s</dd></div>
          <div><dt>Throughput / GPU</dt><dd>{number(cycle.tokens_s_per_gpu)} tokens/s</dd></div>
          <div><dt>TPOT</dt><dd>{number(cycle.tpot_ms)} ms</dd></div>
          <div><dt>Cycle MFU</dt><dd>{factValue(cycle.mfu, "fraction")}</dd></div>
          <div><dt>Target Memory / device</dt><dd>{bytes(cycleMemory?.target_peak_bytes)}</dd></div>
          <div><dt>Target + Draft / device</dt><dd>{bytes(cycleMemory?.peak_bytes)} · {cycle.capacity_status}</dd></div>
        </dl>
        <p className="muted">Single fixed-context scenario; full request queueing and rollback are not modeled by this cycle.</p>
        {cycle.unknown.map((n, i) => <p key={i}>{n}</p>)}
        {Object.entries(e?.traces ?? {}).filter(([k]) => k.startsWith("spec_")).map(([k, t]) => <CalculationTrace key={k} trace={t} />)}
      </section>}
    </details>
    {plan.summary.framework_config && <details className="wc-framework-config"><summary>Framework Parameters · {plan.summary.framework_config.layout}</summary>
      <p>Expert TP = {plan.summary.etp ?? partition.tp} · Stage layers: {plan.summary.framework_config.stage_layer_counts?.join(" / ")}</p>
      <pre>{plan.summary.framework_config.args.join(" ")}{"\n"}
        {Object.entries(plan.summary.framework_config.env).map(([k, v]) => `${k}=${v}`).join("\n")}</pre>
    </details>}
    <details className="wc-assumptions"><summary>Assumptions</summary>{e && <Facts items={e.assumptions} />}
      <details><summary>Hardware Spec & Effective Model</summary>{e && <Facts items={e.topology} />}</details>
      {plan.summary.notes.map((n, i) => <p key={i}>{n}</p>)}
    </details>
    <details><summary>Consistency checks · {failures.length ? `${failures.length} failed` : `${checks.filter(c => c.status === "pass").length} verified`}
      {` · ${checks.filter(c => c.status === "unverifiable").length} not verifiable`}</summary>
      <p className="muted">Read-only UI validation of saved results; abs 1e-6 / relative 1e-8 tolerance, exact integer bytes.</p>
      {checks.map((c, i) => <p key={i}>{c.name} · {c.status} · {c.detail}</p>)}
    </details>
    <RawDetails title="All metric calculations">{() => <>{Object.values(e?.traces ?? {}).map(t => <CalculationTrace key={t.id} trace={t} />)}
      {Object.values(memory.explanation?.traces ?? {}).map(t => <CalculationTrace key={t.id} trace={t} />)}</>}</RawDetails>
    <RawDetails title="Frozen inputs, sources, hashes & IDs">
      {() => <pre>{JSON.stringify({ run, plan, selected_rank: rank }, null, 2)}</pre>}</RawDetails>
  </div>;
}
