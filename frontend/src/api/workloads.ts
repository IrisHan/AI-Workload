import { request } from "./client";
import type { Capabilities, Comparison, Design, DesignInput, DetailPage, Graph, LayerDetail,
  Page, Plan, PlanRef, Plans, Preview, RankDetail, Revision, Run, Search, View, Inspection } from "../types/workload";

const root = "/workloads";
const id = encodeURIComponent;
const send = <T>(path: string, body: unknown, headers?: Record<string, string>, signal?: AbortSignal) =>
  request<T>(root + path, { method: "POST", body: JSON.stringify(body), headers, signal });
const planPath = (run: string, plan: string) => `${root}/runs/${id(run)}/plans/${id(plan)}`;
export const workloadApi = {
  capabilities: (model: string, hardware: string, signal?: AbortSignal) =>
    request<Capabilities>(`${root}/capabilities?${new URLSearchParams({
      ...(model ? { model_id: model } : {}), ...(hardware ? { hardware_id: hardware } : {}),
    })}`, { signal }),
  designs: (cursor?: string, q = "") => request<Page<Design>>(`${root}/designs?${new URLSearchParams({
    limit: "100", q, ...(cursor ? { cursor } : {}),
  })}`),
  design: (identifier: string) => request<Design>(`${root}/designs/${id(identifier)}`),
  create: (name: string, input: DesignInput, key: string) =>
    send<Design>("/designs", { name, input }, { "Idempotency-Key": key }),
  save: (identifier: string, input: DesignInput, etag: string) =>
    send<Design>(`/designs/${id(identifier)}/revisions`, { input }, { "If-Match": etag }),
  revision: (identifier: string, number: number) => request<Revision>(`${root}/designs/${id(identifier)}/revisions/${number}`),
  remove: (identifier: string, etag: string) => request<void>(`${root}/designs/${id(identifier)}`,
    { method: "DELETE", headers: { "If-Match": etag } }),
  preview: (input: DesignInput, client_revision: number, signal: AbortSignal) =>
    send<Preview>("/preview", { input, client_revision }, undefined, signal),
  submit: (revision: string, search: Search, key: string) =>
    send<Run>("/runs", { design_revision_id: revision, search, source_policy: "current_engine" },
      { "Idempotency-Key": key }),
  run: (identifier: string) => request<Run>(`${root}/runs/${id(identifier)}`),
  cancel: (identifier: string) => send<Run>(`/runs/${id(identifier)}/cancel`, {}),
  retry: (identifier: string, key: string) => send<Run>(`/runs/${id(identifier)}/retry`, {}, { "Idempotency-Key": key }),
  plans: (run: string) => request<Plans>(`${root}/runs/${id(run)}/plans?limit=100`),
  plan: (run: string, plan: string) => request<Plan>(planPath(run, plan)),
  layers: (run: string, plan: string, offset: number, rank?: number) =>
    request<DetailPage<LayerDetail>>(`${planPath(run, plan)}/layers?limit=100&offset=${offset}${rank == null ? "" : `&rank=${rank}`}`),
  ranks: (run: string, plan: string, offset: number) =>
    request<DetailPage<RankDetail>>(`${planPath(run, plan)}/ranks?limit=100&offset=${offset}`),
  communication: (run: string, plan: string, offset: number) =>
    request<DetailPage<Record<string, unknown>>>(`${planPath(run, plan)}/communication?limit=50&offset=${offset}`),
  graph: (run: string, plan: string, view: View, layer?: number) =>
    request<Graph>(`${planPath(run, plan)}/graph?view=${view}${layer == null ? "" : `&layer=${layer}`}`),
  inspection: (run: string, plan: string, layer: number, rank: number, phase: string) =>
    request<Inspection>(`${planPath(run, plan)}/inspection?${new URLSearchParams({
      layer: String(layer), rank: String(rank), phase,
    })}`),
  compare: (refs: PlanRef[]) => send<Comparison>("/compare", { plan_refs: refs }),
  exportUrl: (run: string) => `/api${root}/runs/${id(run)}/export`,
};
