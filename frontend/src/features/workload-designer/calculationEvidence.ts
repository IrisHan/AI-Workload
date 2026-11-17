import type { Fact, LayerDetail, Memory, Plan, RankDetail, Run, Shard, Trace } from "../../types/workload";

export const metricNames: Record<string, string> = {
  ttft_ms: "TTFT", itl_ms: "ITL", output_tokens_s: "Output Throughput",
  iteration_time_ms: "Iteration Time", prefill_tokens_s: "Prefill Token Rate · Window",
  prefill_tokens_s_per_gpu: "Prefill Token Rate / GPU · Window", mfu: "Window MFU",
  gpu_activity_fraction: "Modeled GPU Activity", modeled_busy_fraction: "Compute Busy · Window",
  achieved_qps: "Achieved QPS", offered_qps: "Offered QPS", tokens_s_per_gpu: "Output Tokens/s/GPU",
};
const dependencies: Record<string, string[]> = {
  ttft_ms: ["ttft_admission_wait_ms", "service_ttft_ms"],
  itl_ms: ["itl_wave_service_ms", "itl_between_wave_wait_ms", "itl_samples"],
  iteration_time_ms: ["decode_wall_total_ms", "decode_wave_count"],
  output_tokens_s: ["emitted_tokens", "window_seconds"],
  achieved_qps: ["completed_requests", "window_seconds"],
  mfu: ["model_flops_in_window", "dense_peak", "gpu_count", "window_seconds"],
  gpu_activity_fraction: ["gpu_activity_seconds", "gpu_window_seconds"],
  tokens_s_per_gpu: ["emitted_tokens", "window_seconds", "gpu_count"],
  effective_tokens: ["decode_output_tokens", "decode_wave_count"],
  kv: ["context", "quota", "tp", "dtype", "resident_layers", "reserved_tokens"],
  weight: ["runtime_dtype"],
  parameters: [], source_weight: [],
  spec_cycle_seconds: ["target_seconds", "draft_seconds"],
  spec_expected_accepted_tokens: ["draft_tokens", "acceptance_rate"],
  spec_expected_emitted_tokens: ["draft_tokens", "acceptance_rate"],
  spec_output_tokens_s: ["output_tokens", "cycle_seconds"],
  spec_tpot_ms: ["cycle_seconds", "expected_emitted_tokens"],
  spec_mfu: ["flops", "dense_peak", "gpu_count", "cycle_seconds"],
};
export function displayFact(f: Fact): Fact {
  return { ...f, label: f.key === "decode_wall_total_ms" ? "Sum of Decode-wave Service"
    : f.key === "quota" ? "Resident Batch" : f.label,
    source: f.source === "Model" ? "Model Source" : f.source };
}
export function relevantInputs(trace: Trace): Fact[] {
  const keys = dependencies[trace.id];
  // A shared performance envelope is not a dependency declaration.
  const sharedEnvelope = trace.inputs.some(f => f.key === "decode_wall_total_ms");
  return trace.inputs.filter(f => keys ? keys.includes(f.key)
    : sharedEnvelope ? false : !f.key.includes("/") && !/^layer_\d+$/.test(f.key)).map(displayFact);
}
export function calculationContext(plan: Plan): Fact[] {
  const inputs = plan.summary.explanation?.traces.iteration_time_ms?.inputs ?? [];
  return ["window_seconds", "gpu_count", "dense_peak", "decode_wave_count", "emitted_tokens"]
    .flatMap(key => inputs.filter(f => f.key === key).map(displayFact));
}
const identities = new Set(["layer_index", "instance_id", "storage_id", "weight_storage_refs"]);
function semantic(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(semantic);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !identities.has(key)).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, v]) => [key, semantic(v)]));
  return value;
}
export interface LayerGroup { representative: LayerDetail; layers: LayerDetail[]; weightBytes: number | null }
export function layerShards(layer: LayerDetail): Shard[] {
  return layer.local_shapes.filter((s): s is Shard => !!s && typeof s === "object"
    && "bytes" in s && "local_shape" in s && "module_id" in s);
}
export function groupLayers(rows: LayerDetail[]): LayerGroup[] {
  const groups = new Map<string, LayerGroup>();
  for (const row of rows) {
    const key = JSON.stringify(semantic({ rank: row.rank, stage: row.stage, phase: row.phase,
      template: row.template_id, shapes: row.local_shapes, cache: row.cache,
      activation: row.activation_peak_bytes, flops: row.flops, hbm: row.hbm_traffic_bytes, operators: row.operators }));
    const found = groups.get(key);
    if (found) found.layers.push(row);
    else {
      const shards = layerShards(row);
      groups.set(key, { representative: row, layers: [row],
        weightBytes: shards.length === row.local_shapes.length && shards.every(s => Number.isFinite(s.bytes))
          ? shards.reduce((total, s) => total + s.bytes, 0) : null });
    }
  }
  return [...groups.values()];
}

export interface Invariant { name: string; status: "pass" | "fail" | "unverifiable"; detail: string }
const numeric = (v: unknown): v is number => typeof v === "number";
export function validateCalculation(plan: Plan, ranks: RankDetail[] = [], run?: Run): Invariant[] {
  const checks: Invariant[] = [];
  const check = (name: string, actual: unknown, parts: unknown[], expected: () => number, exact = false) => {
    if ([actual, ...parts].some(v => numeric(v) && !Number.isFinite(v))) {
      checks.push({ name, status: "fail", detail: "Non-finite value in calculation." }); return;
    }
    if (actual == null || parts.some(v => v == null)) {
      checks.push({ name, status: "unverifiable", detail: "Missing data or no complete samples." }); return;
    }
    const valid = [actual, ...parts].every(v => numeric(v) && Number.isFinite(v));
    const value = valid ? expected() : NaN;
    const pass = numeric(actual) && Number.isFinite(value)
      && Math.abs(actual - value) <= (exact ? 0 : Math.max(1e-6, Math.abs(value) * 1e-8));
    checks.push({ name, status: pass ? "pass" : "fail", detail: `${String(actual)} ↔ ${String(value)}` });
  };
  const memory = (m: Memory | RankDetail, label: string) => {
    const values = ["weight", "kv", "activation", "communication", "runtime", "fragmentation", "safety"]
      .map(k => m.components[k as keyof typeof m.components]);
    check(`${label} · Peak components`, m.peak_bytes, values, () => values.reduce((a, b) => a + b, 0), true);
    check(`${label} · Free memory`, m.free_bytes, [m.capacity_bytes, m.peak_bytes],
      () => m.capacity_bytes - m.peak_bytes, true);
    Object.entries(m.explanation?.traces ?? {}).forEach(([key, trace]) => {
      const actual = key === "peak" ? m.peak_bytes : key === "free" ? m.free_bytes : m.components[key as keyof typeof m.components];
      if (actual != null) check(`${label} · ${key} evidence`, actual, [trace.result.value], () => Number(trace.result.value), true);
    });
  };
  memory(plan.summary.memory, "Admission");
  ranks.forEach(r => memory(r, `GPU ${r.rank}`));
  const raw = plan.summary.performance?.metrics ?? {}, e = plan.summary.explanation;
  const value = (key: string) => raw[key];
  const input = (trace: string, key: string) => e?.traces[trace]?.inputs.find(f => f.key === key)?.value;
  const n = (v: unknown) => Number(v);
  const window = value("window_seconds");
  const ratio = (name: string, actual: unknown, numerator: unknown, denominator: unknown) =>
    check(name, actual, [numerator, denominator], () => n(denominator) > 0 ? n(numerator) / n(denominator) : NaN);
  ratio("Output Throughput", value("output_tokens_s"), value("emitted_tokens"), window);
  ratio("Achieved QPS", value("achieved_qps"), value("completed_requests"), window);
  const wall = input("iteration_time_ms", "decode_wall_total_ms"), waves = input("iteration_time_ms", "decode_wave_count");
  check("Decode-wave service total", wall, [waves, value("iteration_time_ms")],
    () => n(waves) * n(value("iteration_time_ms")));
  const service = input("itl_ms", "itl_wave_service_ms"), wait = input("itl_ms", "itl_between_wave_wait_ms");
  check("ITL · service + between-wave wait", value("itl_ms"), [service, wait],
    () => n(service) >= 0 && n(wait) >= 0 ? n(service) + n(wait) : NaN);
  const peak = input("mfu", "dense_peak"), gpus = plan.summary.gpu_count;
  check("Window MFU", value("mfu"), [value("model_flops_in_window"), peak, gpus, window],
    () => n(window) > 0 && n(peak) > 0 && gpus > 0
      ? n(value("model_flops_in_window")) / (n(peak) * 1e12 * gpus * n(window)) : NaN);
  ratio("Modeled GPU Activity", value("gpu_activity_fraction"),
    input("gpu_activity_fraction", "gpu_activity_seconds"), input("gpu_activity_fraction", "gpu_window_seconds"));
  if (e) {
    const parts = e.performance.breakdown.map(t => t.result.value);
    check("Decode breakdown", value("iteration_time_ms"), parts.length ? parts : [null],
      () => parts.reduce<number>((sum, v) => sum + n(v), 0));
    const p = e.performance;
    check("Emitted token reconciliation", value("emitted_tokens"),
      [p.completed_full_outputs, p.unfinished_outputs_in_window, p.completed_outputs_before_window],
      () => p.completed_full_outputs + p.unfinished_outputs_in_window - p.completed_outputs_before_window, true);
  }
  for (const [key, metric] of Object.entries(plan.summary.metric_values)) {
    if (key in raw) check(`${metricNames[key] ?? key} · metric envelope`, metric.value, [raw[key]], () => n(raw[key]));
    const t = e?.traces[key];
    if (t) check(`${metricNames[key] ?? key} · trace result`, metric.value, [t.result.value], () => n(t.result.value));
  }
  for (const trace of Object.values(e?.traces ?? {})) {
    // Speculative cycles deliberately have a different window and denominator.
    if (trace.id.startsWith("spec_")) continue;
    for (const f of relevantInputs(trace)) {
      if (f.key in raw) check(`${metricNames[trace.id] ?? trace.id} · ${f.label} binding`,
        f.value, [raw[f.key]], () => n(raw[f.key]));
      if (f.key === "gpu_count") check(`${trace.id} · GPU count binding`, f.value, [gpus], () => gpus, true);
    }
  }
  const scopes = [e?.scope, plan.summary.memory.explanation?.scope, ...ranks.map(r => r.explanation?.scope),
    ...Object.values(e?.traces ?? {}).map(t => t.scope),
    ...ranks.flatMap(r => Object.values(r.explanation?.traces ?? {}).map(t => t.scope))];
  for (const [index, scope] of scopes.entries()) {
    for (const [key, expected] of Object.entries({ plan_id: plan.id, run_id: plan.run_id,
      ...(run ? { input_hash: run.input_hash, structure_id: run.structure_snapshot_id } : {}) })) {
      checks.push({ name: `Scope ${index + 1} · ${key}`,
        status: scope?.[key] == null ? "unverifiable" : scope[key] === expected ? "pass" : "fail",
        detail: scope?.[key] == null ? "Historical evidence has no binding." : scope[key] === expected ? "Bound to selected result." : "Evidence belongs to a different result." });
    }
  }
  for (const r of ranks) {
    const scope = r.explanation?.scope;
    if (scope?.rank != null) check(`GPU ${r.rank} · scope`, scope.rank, [r.rank], () => r.rank, true);
  }
  return checks;
}
