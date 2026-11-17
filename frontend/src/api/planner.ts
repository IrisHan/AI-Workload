import { post, request } from "./client";
import type {
  Hardware,
  Model,
  ModelAnalysis,
  ModelMemory,
  ModelMemoryInput,
  StructureSnapshot,
} from "../types";

export const api = {
  health: () =>
    request<{ status: string; version: string; database: string }>("/health"),
  models: () => request<Model[]>("/models"),
  model: (id: string) => request<Model>(`/models/${encodeURIComponent(id)}`),
  importModel: (input: { repo_id: string; revision?: string }) =>
    post<Model>("/models/import", input),
  analyze: (id: string) =>
    post<ModelAnalysis>(`/models/${encodeURIComponent(id)}/analyze`),
  modelMemory: (id: string, input: ModelMemoryInput) =>
    post<ModelMemory>(`/models/${encodeURIComponent(id)}/memory`, input),
  structure: (id: string) =>
    post<StructureSnapshot>(`/models/${encodeURIComponent(id)}/structure`),
  structures: (id: string) =>
    request<StructureSnapshot[]>(`/models/${encodeURIComponent(id)}/structures`),
  hardware: () => request<Hardware[]>("/hardware?include_legacy=true"),
};
