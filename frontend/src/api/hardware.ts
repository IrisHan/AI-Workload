import { request } from "./client";

export interface Evidence { confidence: string; source_ids: string[]; note: string }
export interface Compute { dense: number | null; sparse: number | null; unit: "TFLOPS" | "TOPS"; status: string; evidence: Evidence }
export interface HardwareRecord {
  id: string; sku: string; name: string; vendor: string; family: string; architecture: string;
  form_factor: string; memory_type: string; memory_bytes: number; memory_bandwidth_bytes_s: number;
  bf16_tflops: number | null; fp32_tflops: number | null; nvlink_gb_s: number | null; power_watts: number | null;
  source_url: string; retrieved_at: string; version: string; notes: string; content_sha256: string;
  compute: Record<string, Compute>;
  interconnect: { name: string; bandwidth_gb_s: number | null; direction: string; scope: string; evidence: Evidence }[];
  node: { name: string; accelerators: number | null; nic_gb_s_per_gpu: number | null; nic_direction: string; full_mesh: boolean | null; evidence: Evidence };
  cost: { cny_per_day: number | null; basis: string; as_of: string | null; evidence: Evidence };
  field_evidence: Record<string, Evidence>;
  education: { title: string; text: string; source_ids: string[] }[];
  sources: { id: string; title: string; url: string | null; retrieved_at: string; sha256: string | null; artifact: string }[];
  caveats: string[]; planner_precisions: string[]; legacy: boolean;
}
export interface RooflinePoint { intensity: number; performance_tops: number }
export interface ArchitectureFact {
  key: string; label: string; value: string; scope: string; status: string; source_ids: string[]; note: string;
}
export interface ArchitectureBlock { kind: "text" | "table"; text: string; headers: string[]; rows: string[][] }
export interface ArchitectureDocument {
  hardware_id: string; hardware_version: string; hardware_sha256: string;
  architecture: { id: string; name: string; version: string; content_sha256: string; specs: ArchitectureFact[]; explanations: string[] };
  variant: { id: string; architecture_id: string; source_family: string | null; facts: ArchitectureFact[]; notes: Record<string, string[]> };
  document_version: string; content_sha256: string; headline_metrics: ArchitectureFact[]; memory_hierarchy: ArchitectureFact[];
  sections: { number: number; title: string; notes: string[]; blocks: ArchitectureBlock[]; reference_blocks: ArchitectureBlock[]; reference_text: string }[];
  sources: HardwareRecord["sources"]; related_skus: { id: string; name: string }[];
  comparison: ArchitectureBlock[]; caveats: string[];
}
export interface RooflineResult {
  hardware_id: string; hardware_version: string; hardware_sha256: string; formula_version: string;
  precision: string; sparse: boolean; unit: string; peak_tops: number; bandwidth_bytes_s: number;
  ridge_point: number; operating_point: RooflinePoint; bottleneck: string; curve: RooflinePoint[]; assumptions: string[];
}
export const hardwareApi = {
  list: () => request<HardwareRecord[]>("/hardware"),
  detail: (id: string) => request<HardwareRecord>(`/hardware/${encodeURIComponent(id)}`),
  roofline: (id: string, precision: string, sparse: boolean, intensity: number) =>
    request<RooflineResult>(`/hardware/${encodeURIComponent(id)}/roofline?${new URLSearchParams({
      precision, sparse: String(sparse), intensity: String(intensity),
    })}`),
  export: (id: string) => request<HardwareRecord>(`/hardware/${encodeURIComponent(id)}/export`),
  architecture: (id: string) => request<ArchitectureDocument>(`/hardware/${encodeURIComponent(id)}/architecture`),
};
