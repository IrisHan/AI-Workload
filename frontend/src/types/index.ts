export interface Model {
  id: string;
  name: string;
  provider: string;
  repo_id: string;
  revision: string;
  architecture: string;
  parameters: number;
  layers: number;
  hidden_size: number;
  attention_heads: number;
  kv_heads: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  tie_word_embeddings: boolean;
  dtype: string;
  is_moe: boolean;
  source_url: string;
  retrieved_at: string;
  config_sha256: string;
  raw_config: Record<string, unknown>;
  version: string;
  active_parameters?: number | null;
  parameter_scope?: string;
  analysis_formula?: string;
  workload_supported?: boolean;
  support_notes?: string[];
  source_metadata?: Record<string, unknown>;
}
export interface ModelMemoryInput {
  batch_size: number;
  context_length: number;
  dtype: "bf16" | "fp32";
  include_training_state: boolean;
}
export interface ModelMemory {
  formula_version: string;
  input_snapshot: Record<string, unknown>;
  weight_bytes: number;
  kv_cache_bytes: number;
  gradient_bytes: number;
  optimizer_bytes: number;
  static_total_bytes: number;
  integer_buffer_bytes?: number;
  activation_bytes: null;
  peak_memory_bytes: null;
  assumptions: string[];
}
export interface ModelAnalysis {
  model_id: string;
  parameters: number;
  parameter_breakdown: Record<string, number>;
  layers: {
    name: string;
    parameters: number;
    input_shape: string;
    output_shape: string;
  }[];
  engine_version: string;
  formula_version: string;
  assumptions: string[];
}
export interface StructureModule {
  id: string; label: string; category: string; shape: number[];
  params: number; active_params: number; formula: string; source: string;
  input_shape: string; output_shape: string; shared_with?: string | null;
}
export interface StructureTemplate {
  id: string; label: string; modules: StructureModule[];
  nodes: { id: string; label: string; module_id?: string | null; category: string; detail?: string }[];
  edges: { source: string; target: string; label?: string | null }[];
}
export interface StructureSnapshot {
  id: string; model_id: string; model_version: string; engine_version: string;
  formula_version: string; created_at: string;
  analysis: {
    templates: StructureTemplate[];
    groups: { id: string; template_id: string; count: number; ranges: [number, number][] }[];
    shared_modules: StructureModule[];
    totals: { total: number; active: number };
  };
  comparison: { matched: boolean; parameter_delta: number; category_deltas: Record<string, number>;
    baseline_engine: string; baseline_formula: string };
}
export interface Hardware {
  id: string;
  name: string;
  vendor: string;
  memory_bytes: number;
  memory_bandwidth_bytes_s: number;
  bf16_tflops: number | null;
  fp32_tflops: number | null;
  nvlink_gb_s: number | null;
  power_watts: number | null;
  planner_precisions: string[];
  legacy: boolean;
  source_url: string;
  retrieved_at: string;
  version: string;
  notes: string;
}
