import type { DesignInput, Design, Search } from "../../types/workload";

export const defaultSearch: Search = { top_k: 5, max_candidates: 2000, max_seconds: 10, event_budget: 100000 };
export function initialInput(model = "", hardware = ""): DesignInput {
  return {
    schema_version: "workload-design-1.0", model_id: model, hardware_id: hardware, mode: "online",
    load: { prompt_tokens: 2048, output_tokens: 128, concurrency: 8, request_rate: 1, request_count: null,
      arrival: "deterministic", seed: 1, duration_seconds: 30, warmup_seconds: 0 },
    resources: { node_count: 1, gpus_per_node: 8, topology_template: "switched_sxm_reference-v1", topology_overrides: {} },
    runtime: { profile: "analytical-continuous-v1", weight_dtype: "bf16", activation_dtype: "bf16", kv_dtype: "bf16",
      max_batch_sequences: 8, max_batched_tokens: 2048, prefill_chunk_tokens: 512,
      continuous_batching: true, kv_page_tokens: 16, speculative: { enabled: false }, fairness_interval: 8 },
    parallelism: { layout: "vllm-v0.10.2", expert_parallel: false, mode: "auto", allowed_tp: null, allowed_pp: null, allowed_ep: null,
      dp_scaling: false, layer_locks: [], pp_boundaries: null, manual_candidate: null },
    targets: { ttft_ms: null, itl_ms: null, policy: "soft" }, overrides: [],
  };
}
export interface EditorState { draft: DesignInput; dirty: boolean; clientRevision: number; base: Design | null }
export type EditorAction = { type: "edit"; input: DesignInput } | { type: "saved"; design: Design };
export function reducer(state: EditorState, action: EditorAction): EditorState {
  if (action.type === "saved") return { ...state, base: action.design, draft: action.design.revision.input, dirty: false };
  return { ...state, draft: action.input, dirty: true, clientRevision: state.clientRevision + 1 };
}
export function validate(input: DesignInput): string[] {
  const errors: string[] = [];
  if (!input.model_id || !input.hardware_id) errors.push("请选择模型和 GPU 来源");
  const positive: [string, number][] = [
    ["Prompt tokens", input.load.prompt_tokens], ["Output tokens", input.load.output_tokens],
    ["并发", input.load.concurrency], ["节点数", input.resources.node_count], ["每节点卡数", input.resources.gpus_per_node],
    ["Batch sequences", input.runtime.max_batch_sequences], ["Batch tokens", input.runtime.max_batched_tokens],
    ["Prefill chunk", input.runtime.prefill_chunk_tokens],
  ];
  for (const [name, value] of positive) if (!Number.isSafeInteger(value) || value < 1) errors.push(`${name} 必须是正整数`);
  if (input.runtime.max_batch_sequences > input.load.concurrency) errors.push("Batch sequences 不能超过并发");
  if (input.runtime.prefill_chunk_tokens > input.runtime.max_batched_tokens) errors.push("Prefill chunk 不能超过 Batch tokens");
  if (input.resources.node_count * input.resources.gpus_per_node > 256) errors.push("GPU 总预算不能超过 256");
  if (input.load.warmup_seconds >= input.load.duration_seconds) errors.push("预热时间必须小于测量窗口");
  const lists = [input.parallelism.allowed_tp, input.parallelism.allowed_pp, input.parallelism.allowed_ep,
    input.parallelism.pp_boundaries, ...input.parallelism.layer_locks.map(l => l.allowed_tp),
    input.parallelism.manual_candidate?.boundaries];
  if (lists.some(list => list?.some(n => !Number.isSafeInteger(n) || n < 1))) errors.push("Parallelism 列表必须是正整数");
  if (input.parallelism.manual_candidate) {
    const c = input.parallelism.manual_candidate;
    if ([c.tp, c.pp, c.ep, c.dp].some(n => !Number.isSafeInteger(n) || n < 1)) errors.push("TP / PP / EP / DP 必须是正整数");
    if (c.boundaries.length !== c.pp - 1) errors.push("PP 边界数量必须等于 PP − 1");
    const used = c.tp * c.pp * c.dp * (input.parallelism.layout === "orthogonal-tp-ep-v1" ? c.ep : 1);
    if (used > input.resources.node_count * input.resources.gpus_per_node) errors.push("手动并行方案超过 GPU 预算");
    if (input.parallelism.layout === "sglang-v0.5.3" && c.tp % c.ep !== 0) errors.push("SGLang EP 必须整除 TP");
    if (input.parallelism.layout === "sglang-v0.5.3" && c.pp > 1 && input.runtime.speculative.enabled)
      errors.push("SGLang v0.5.3 不支持 PP > 1 与投机解码同时使用");
    if (input.parallelism.layout === "sglang-v0.5.3" && c.dp > 1 && input.resources.node_count > 1)
      errors.push("SGLang v0.5.3 的普通 DP 模式仅支持单节点");
  }
  const spec = input.runtime.speculative;
  if (spec.enabled) {
    const k = spec.draft_tokens ?? 5, a = spec.acceptance_rate ?? .75;
    if (!Number.isInteger(k) || k < 1 || k > 64) errors.push("Draft Tokens 必须是 1–64 的整数");
    if (!Number.isFinite(a) || a < 0 || a > 1) errors.push("Acceptance Rate 必须在 0–1 之间");
  }
  return errors;
}
export const bytes = (value: number | null | undefined) => {
  if (value == null) return "—";
  const unit = Math.abs(value) >= 2 ** 30 ? 3 : Math.abs(value) >= 2 ** 20 ? 2 : Math.abs(value) >= 1024 ? 1 : 0;
  return `${(value / 1024 ** unit).toLocaleString(undefined, { maximumFractionDigits: unit ? 3 : 0 })} ${["B", "KiB", "MiB", "GiB"][unit]}`;
};
export const number = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: 3 });
