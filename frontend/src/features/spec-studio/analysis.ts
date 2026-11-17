export interface Metrics {
  params_total: number;
  params_active: number;
  matrix_params: number;
}
export interface ModuleSpec {
  id: string;
  label: string;
  category: string;
  shape: number[];
  input_shape: string;
  output_shape: string;
  params: number;
  active_params: number;
  matrix_params: number;
  dtype: string;
  formula: string;
  source: string;
  notes?: string | null;
  count?: number;
  shared_with?: string | null;
}
export interface MoeLayout {
  n_routed: number;
  top_k: number | null;
  n_shared: number;
  expert_dim: number;
  hidden_size?: number | null;
  shared_width: number;
  has_shared_expert_gate: boolean;
  has_router: boolean;
  latent_dim?: number | null;
}
export interface OpNode {
  id: string;
  label: string;
  category: string;
  module_id?: string | null;
  detail?: string | null;
  cluster?: string | null;
}
export interface Template {
  id: string;
  backbone?: Pick<Analysis["backbone"], "num_heads" | "num_kv_heads" | "head_dim">;
  label: string;
  signature: string;
  role: string;
  attn_kind: string;
  ffn_kind: string;
  modules: ModuleSpec[];
  nodes: OpNode[];
  edges: { source: string; target: string; label?: string | null }[];
  metrics: Metrics;
  moe?: MoeLayout | null;
}
export interface LayerGroup {
  id: string;
  template_id: string;
  count: number;
  ranges: [number, number][];
  metrics: Metrics;
}
export interface Analysis {
  analysis_id: string;
  branches?: (Template & { count: number; included: boolean })[];
  schema_version: string;
  engine_version: string;
  formula_version: string;
  status: string;
  source: {
    repo_id: string | null;
    requested_revision: string | null;
    resolved_sha: string | null;
    config_sha256: string;
    fetched_at: string | null;
    kind: string;
  };
  identity: { name: string; repo_id: string | null; model_type: string };
  backbone: {
    num_layers: number; hidden_size: number; num_heads: number;
    num_kv_heads: number; head_dim: number; vocab_size: number;
    context_length: number; intermediate_size: number;
  };
  totals: { total: number; active: number; [key: string]: unknown };
  serving: {
    weight_bytes: number; weight_bytes_logical: number; weight_bytes_source: string;
    flop_gemm_decode: number; include_mtp: boolean;
    kv_layout: { bytes_per_token: number | null; dtype_assumed: string; formula: string; notes?: string;
      reference_context?: number; reference_bytes?: number };
  };
  capabilities: { id: string; label: string; supported: boolean; evidence: unknown[] }[];
  diagnostics: { severity: string; code: string; message: string }[];
  morphology?: {
    attn: string[];
    ffn: string[];
    vision: string;
    quant: string;
    cost_supported: boolean;
  };
  templates: Template[];
  groups: LayerGroup[];
  execution: { group_id: string; start: number; stop_exclusive: number }[];
  shared_modules: ModuleSpec[];
}

export type GraphScope = "all" | "attention" | "ffn" | "vision";

export function categoryTone(category: string): string {
  if (/attn|attention|rope|position/i.test(category)) return "attention";
  if (/ffn|mlp|expert|moe|router/i.test(category)) return "ffn";
  if (/vision|connector/i.test(category)) return "vision";
  if (/norm/i.test(category)) return "norm";
  return "other";
}

export function rangeLabel(ranges: [number, number][]): string {
  return ranges.map(([start, stop]) => stop === start + 1 ? `L${start}` : `L${start}–${stop - 1}`).join(", ");
}

export function isMoeFfn(kind?: string | null): boolean {
  return kind === "MOE" || kind === "LATENT_MOE";
}

export function moeLayout(template: Template): MoeLayout | null {
  if (template.moe?.n_routed) return template.moe;
  if (!isMoeFfn(template.ffn_kind)) return null;
  const routed = template.modules.find(item => item.id.endsWith("routed_expert.gate_proj"));
  if (!routed || routed.shape.length < 2) return null;
  const nRouted = routed.count ?? 1;
  const expertDim = routed.shape[0];
  const per = nRouted ? routed.params / nRouted : 0;
  const shared = template.modules.find(item => item.id.endsWith("shared_expert.gate_proj"));
  let nShared = 0;
  let sharedWidth = 0;
  if (shared) {
    sharedWidth = shared.shape[0];
    const counted = shared.count ?? 1;
    nShared = counted > 1 ? counted : (expertDim ? Math.floor(sharedWidth / expertDim) : 1);
    if (counted > 1) sharedWidth = expertDim * nShared;
  }
  return {
    n_routed: nRouted,
    top_k: per ? Math.round(routed.active_params / per) : null,
    n_shared: nShared,
    expert_dim: expertDim,
    hidden_size: routed.shape[1],
    shared_width: sharedWidth,
    has_shared_expert_gate: template.modules.some(item => item.id.endsWith("shared_expert_gate")),
    has_router: template.modules.some(item => item.id === "router" || item.id.endsWith(".router")),
    latent_dim: template.modules.find(item => item.id === "routed_expert_down_proj")?.shape[0] ?? null,
  };
}

export function ffnRecipe(template: Template): string {
  if (!isMoeFfn(template.ffn_kind) && !template.moe?.n_routed) {
    const gate = template.modules.find(item => item.id === "gate_proj");
    const width = gate?.shape[0];
    return width
      ? `Dense SwiGLU · gate + up + down · d→${width}`
      : "Dense SwiGLU · gate + up + down";
  }
  const moe = moeLayout(template);
  if (!moe) return "MoE";
  const parts = [
    moe.has_router ? `router d→${moe.n_routed}` : "router",
  ];
  if (moe.latent_dim) parts.push(`latent d→${moe.latent_dim}`);
  parts.push(
    `routed ×${moe.n_routed}${moe.top_k != null ? ` (top-k ${moe.top_k})` : ""} SwiGLU d→${moe.expert_dim}`,
  );
  if (moe.n_shared) {
    parts.push(
      moe.n_shared > 1
        ? `shared ${moe.n_shared}×W (${moe.shared_width})`
        : `shared ×1 (W=${moe.shared_width})`,
    );
  } else {
    parts.push("no shared expert");
  }
  if (moe.has_shared_expert_gate) parts.push("shared_expert_gate d→1");
  return `MoE · ${parts.join(" · ")}`;
}

export type ClusterId = "q_path" | "kv_path" | "score" | "out" | "router" | "latent" | "routed" | "shared" | "merge" | "patch" | "encoder" | "projector";

export function nodeCluster(node: OpNode): ClusterId | null {
  const known: ClusterId[] = [
    "q_path", "kv_path", "score", "out", "router", "latent", "routed", "shared", "merge",
    "patch", "encoder", "projector",
  ];
  if (node.cluster && known.includes(node.cluster as ClusterId)) return node.cluster as ClusterId;
  const name = node.id.startsWith("block.") ? node.id.slice(6) : node.id;
  if (name === "routed_expert_down_proj" || name === "routed_expert_up_proj" || name === "routed_expert_norm.weight")
    return "latent";
  if (name.startsWith("routed_expert")) return "routed";
  if (name.startsWith("shared_expert") || name === "shared_expert_gate") return "shared";
  if (["router", "router_correction", "router_scoring", "top_k", "expert_selection", "hash_lookup"].includes(name))
    return "router";
  if (["expert_sum", "weighted_expert_sum"].includes(name)) return "merge";
  return null;
}

export function readableError(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(row => readableError(row)).join("; ");
  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    return String(record.message ?? record.msg ?? JSON.stringify(detail));
  }
  return "分析请求失败";
}
