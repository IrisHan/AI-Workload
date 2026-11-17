import { request } from "./client";
import type { StructureSnapshot } from "../types";

export interface StructureOptions { kv_dtype: string; include_mtp: boolean; }
const base = (model: string) => `/models/${encodeURIComponent(model)}`;
export const structureApi = {
  create: (model: string, options: StructureOptions, signal?: AbortSignal) =>
    request<StructureSnapshot>(`${base(model)}/structure`, {
      method: "POST", body: JSON.stringify(options), signal,
    }),
  history: (model: string, signal?: AbortSignal) =>
    request<StructureSnapshot[]>(`${base(model)}/structures`, { signal }),
  snapshot: (model: string, id: string, signal?: AbortSignal) =>
    request<StructureSnapshot>(`${base(model)}/structures/${encodeURIComponent(id)}`, { signal }),
  resource: <T>(model: string, id: string, suffix: string, signal: AbortSignal, body?: unknown) =>
    request<T>(`${base(model)}/structures/${encodeURIComponent(id)}/${suffix}`, {
      signal, method: body === undefined ? "GET" : "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  glossary: (signal?: AbortSignal) =>
    request<Record<string, { title: string; summary: string; formula?: string }>>("/structure/glossary", { signal }),
};
