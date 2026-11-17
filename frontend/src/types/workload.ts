import type { Model } from "./index";
import type { HardwareRecord } from "../api/hardware";

export type Precision = "bf16" | "fp32";
export type View = "logical" | "placement" | "communication";
export type Phase = "all" | "prefill" | "decode" | "mixed";
export type ParallelLayout = "orthogonal-tp-ep-v1" | "vllm-v0.10.2" | "sglang-v0.5.3";
export interface Candidate { tp: number; pp: number; ep: number; dp: number; boundaries: number[]; layout?: ParallelLayout; moe?: boolean }
export interface DesignInput {
  schema_version: "workload-design-1.0";
  model_id: string; hardware_id: string; mode: "online" | "offline";
  load: { prompt_tokens: number; output_tokens: number; concurrency: number;
    request_rate: number | null; request_count: number | null; arrival: "deterministic" | "poisson" | null;
    seed: number; duration_seconds: number; warmup_seconds: number };
  resources: { node_count: number; gpus_per_node: number; topology_template: string;
    topology_overrides: Record<string, number | null> };
  runtime: { profile: string; weight_dtype: Precision; activation_dtype: Precision; kv_dtype: Precision;
    max_batch_sequences: number; max_batched_tokens: number; prefill_chunk_tokens: number;
    continuous_batching: boolean; kv_page_tokens: 16; speculative: {
      enabled: boolean; draft_tokens?: number; acceptance_rate?: number;
      draft_latency_ms?: number | null; draft_memory_bytes?: number | null; include_prefill?: boolean };
    fairness_interval: number };
  parallelism: { layout: ParallelLayout; expert_parallel?: boolean; mode: "auto" | "manual"; allowed_tp: number[] | null;
    allowed_pp: number[] | null; allowed_ep: number[] | null; dp_scaling: boolean;
    layer_locks: { layer_index: number; allowed_tp: number[] }[]; pp_boundaries: number[] | null;
    manual_candidate: Candidate | null };
  targets: { ttft_ms: number | null; itl_ms: number | null; policy: "soft" | "hard" };
  overrides: { path: string; value: number; reason: string }[];
}
export interface Search { top_k: number; max_candidates: number; max_seconds: number; event_budget: number }
export interface Revision { id: string; design_id: string; revision_no: number; input: DesignInput;
  input_hash: string; created_at: string }
export interface Design { id: string; name: string; head_revision: number; etag: string;
  model_name?: string; hardware_name?: string;
  created_at: string; updated_at: string; revision: Revision; runs: Run[] }
export interface Issue { code: string; message?: string; field_path?: string; details?: Record<string, unknown>;
  candidate?: Candidate; status?: string }
export interface Coverage { generated?: number; evaluated?: number; rejected?: number; events?: number;
  minimum_proven?: boolean; reasons?: Record<string, number>; all_partitions?: boolean }
export interface Run { id: string; design_id: string; revision_id: string;
  design_input: DesignInput; structure_snapshot_id: string;
  status: "queued" | "running" | "cancel_requested" | "completed" | "partial" | "cancelled" | "failed" | "interrupted";
  input_hash: string; versions: Record<string, string>; queued_at: string; finished_at: string | null;
  progress: Coverage; summary: { outcome?: string; stop_reason?: string; coverage?: Coverage; rejections?: Issue[] } | null;
  error: Issue | null; source_updates: { source: string; pinned_version: string; change_kind: string }[] }
export interface Page<T> { items: T[]; next_cursor: string | null }
export interface DetailPage<T> { items: T[]; total: number; next_offset: number | null; result_hash: string }
export type Components = Record<"weight" | "kv" | "activation" | "communication" | "runtime" | "fragmentation" | "safety", number>;
export interface Memory { peak_bytes: number; capacity_bytes: number; free_bytes: number;
  components: Components; worst_rank: number; kind: string; explanation?: MemoryExplanation }
export interface Metric { value: number | null; unit: string; kind: string; scope: string;
  unavailable_reason?: string | null; confidence_basis: string; scenario_id: string }
export interface Assumption { default: number; effective: number; reason?: string; source?: string }
export interface Placement { rank: number; node: number; gpu: number; p: number; t: number; e: number; d: number }
export interface SpeculativeCycle {
  status: string; scope: string; draft_tokens: number; acceptance_rate: number;
  expected_accepted_tokens: number; expected_emitted_tokens: number;
  decode_requests: number; context_tokens: number; prefill_tokens: number;
  target_seconds: number | null; draft_seconds: number | null; cycle_seconds: number | null;
  output_tokens_s: number | null; tokens_s_per_gpu: number | null; tpot_ms: number | null; mfu: number | null;
  capacity_status: string; unknown: string[]; notes: string[];
  memory: { rank: number; target_peak_bytes: number; draft_bytes: number | null; peak_bytes: number | null }[];
}
export interface Plan { id: string; run_id: string; rank: number; candidate_key: string; result_hash: string;
  summary: { candidate: Candidate; gpu_count: number; unused_gpu_count: number; node_count: number;
    explanation?: PlanExplanation;
    capacity_status: string; execution_status: string; target_status: string; recommended: boolean;
    recommendation_reasons: string[]; memory: Memory; placement: Placement[];
    metric_values: Record<string, Metric>; assumptions: Record<string, Assumption>; notes: string[];
    event_memory: { coverage: string; until: number; assumption: string };
    performance?: { metrics: { percentiles?: Record<string, number | null>; percentile_sample_count?: number;
      percentile_policy?: string; percentile_reason?: string | null; [key: string]: unknown };
      notes?: string[] };
    speculative_cycle?: SpeculativeCycle | null;
    etp?: number;
    framework_config?: { layout: string; status: string; args: string[]; env: Record<string, string>;
      stage_layer_counts?: number[]; notes?: string[] };
    sensitivity: unknown; diagnostic_replay: { status: string; trace_complete: boolean; output_truncated?: boolean } } }
export interface Plans extends Page<Plan> { search_complete: boolean; minimum_proven: boolean; coverage: Coverage }
export interface TimelinePoint { at: number; total_bytes: number; components: Components }
export interface RankDetail extends Placement { peak_bytes: number; capacity_bytes: number; free_bytes: number;
  explanation?: MemoryExplanation;
  components: Components; weights: Shard[];
  resident_request_quota: number; event_memory: { peak_bytes: number; peak_at: number;
    component_at_peak: Components; timeline: TimelinePoint[]; timeline_sampled: boolean } }
export interface LayerDetail { layer_index: number; instance_id: string; rank: number; stage: number;
  phase: string; template_id: string; weight_storage_refs: string[]; local_shapes: unknown[];
  flops: number | null; hbm_traffic_bytes: number | null; scope: string; operators: unknown[];
  cache: { total_bytes: number }; activation_peak_bytes: number }
export interface GraphNode { id: string; label: string; layer_index?: number; rank_refs: number[];
  rank?: number; node?: number; stage?: number; attention?: string; ffn?: string; source_node_id?: string;
  gpu?: number; p?: number; t?: number; e?: number; d?: number;
  category?: string; module_id?: string; weight_refs?: string[] }
export interface NetworkStat { bytes: number | null; operation_seconds: number | null; operation_count: number;
  wave_ids: number[]; sample_paths: { src: number; dst: number; bytes: number; path: string[] }[] }
export interface NetworkEdge { id: string; source: string; target: string; kind?: string;
  group_id?: string; rank_refs?: number[]; distance?: string; medium?: string;
  bandwidth_bytes_s?: number | null; effective_bandwidth_bytes_s?: number | null; efficiency?: number;
  latency_seconds?: number; path?: string[]; statistics?: Record<string, NetworkStat> }
export interface NetworkInfo { version: string; replica: number; replica_count: number; gpu_count: number;
  local_medium: string; intra_bandwidth_bytes_s: number | null; inter_bandwidth_bytes_s: number | null;
  node_network_bytes_s: number | null;
  nic_per_gpu_bytes_s: number | null; fabric: string; hardware_sha256: string;
  iteration: Metric | null; diagnostic_coverage: string; timing_scope: string }
export interface Graph { nodes: GraphNode[];
  edges: NetworkEdge[]; network?: NetworkInfo | null;
  source_structure_id: string; plan_id: string; result_hash: string; truncated: boolean }
export interface Capabilities { supported: boolean; issues: Issue[]; model: Model | null; precisions: string[];
  hardware: HardwareRecord | null;
  topology_defaults: Record<string, Record<string, number | null>>;
  parallel_domains: Record<string, number[]> | null;
  parallel_domains_by_layout?: Partial<Record<ParallelLayout, Record<string, number[]>>>;
  overrides: Record<string, { default: number; minimum: number; maximum: number }>;
  runtime_profiles: string[]; topology_templates: string[] }
export interface Preview { client_revision: number; coverage: string; needs_full_run: boolean;
  capacity_status: string; lower_bound_bytes: number | null; diagnostics: Issue[] }
export interface PlanRef { run_id: string; plan_id: string }
export interface Comparison { plans: Plan[]; input_differences: { path: string; values: unknown[] }[];
  versions: Record<string, string>[]; assumptions: Record<string, Assumption>[] }
export interface Shard { storage_id: string; module_id: string; local_shape: number[]; count: number;
  bias: number; bytes: number; partition: string; expert_start: number | null; expert_stop: number | null }
export interface Inspection {
  explanation?: { version: string; scope: Record<string, unknown>; nodes: Record<string, NodeExplanation>;
    steps: { step: number; bytes: number; allocation_ids: string[] }[]; lifetime: Trace };
  result_hash: string; source_structure_id: string; resident: boolean;
  scope: { layer: number; rank: number; phase: string; kind: string };
  placement: Placement; shards: Shard[]; local_tokens: number; resident_request_quota: number;
  cache: { total_bytes: number; payload_bytes: number; metadata_bytes: number; tail_waste_bytes: number };
  allocations: { id: string; component: string; bytes: number; start: number; stop: number; source: string }[];
  activation_peak_bytes: number;
  timing?: { reason: string | null; scope: string; context_tokens?: number; tokens_per_group?: number[];
    operators: { id: string; flops: number; scalar_equivalent_ops: number; hbm_bytes: number;
      seconds: number | null; compute_seconds?: number; hbm_seconds?: number; launch_seconds?: number;
      peak_tflops?: number; hbm_bandwidth_bytes_s?: number; compute_efficiency?: number;
      hbm_efficiency?: number; bottleneck?: string; unknown?: string | null }[] };
}

export interface Fact { key: string; label: string; value: unknown; unit: string; source: string;
  source_ref?: string | null; formula?: string | null }
export interface Trace { id: string; title: string; status: string; scope: Record<string, unknown>;
  inputs: Fact[]; steps: { label: string; formula: string; value: unknown; unit: string }[];
  formula: string; assumptions: Fact[]; result: Fact; notes: string[]; unavailable_reason?: string | null;
  sources: { path: string; type: string }[] }
export interface MemoryExplanation { version: string; scope: Record<string, unknown>; traces: Record<string, Trace>;
  utilization: number; capacity_gb: number; capacity_pass: boolean }
export interface PlanExplanation { version: string; scope: Record<string, unknown>; context: Fact[];
  template_layer_counts?: Record<string, number>;
  traces: Record<string, Trace>; assumptions: Fact[]; topology: Fact[];
  performance: { status: string; reason?: string | null; breakdown: Trace[]; bottleneck: string;
    decode_wave_count: number; itl_samples: number; effective_tokens_per_iteration: number | null;
    itl_wave_service_ms: number | null; itl_between_wave_wait_ms: number | null;
    ttft_admission_wait_ms: number | null; completed_full_outputs: number;
    completed_outputs_before_window: number; unfinished_outputs_in_window: number } }
export interface NodeExplanation { parameters: number; weight_bytes: number; full_shape: number[] | null;
  local_shape: number[] | null; shapes: Fact[]; traces: Trace[]; attribution: string;
  expert_ranges: number[][]; operators: NonNullable<Inspection["timing"]>["operators"] }
