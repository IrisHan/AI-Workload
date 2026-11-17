import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { workloadApi as api } from "../../api/workloads";
import type { PlanRef, Run, View } from "../../types/workload";

export const active = (status?: Run["status"]) => ["queued", "running", "cancel_requested"].includes(status ?? "");
export const useDesigns = (q = "") => useInfiniteQuery({
  queryKey: ["workload-designs", q], queryFn: ({ pageParam }) => api.designs(pageParam, q),
  initialPageParam: undefined as string | undefined, getNextPageParam: (page) => page.next_cursor ?? undefined,
});
export const useDesign = (id: string) => useQuery({
  queryKey: ["workload-design", id], queryFn: () => api.design(id), enabled: !!id,
});
export const useRevision = (id: string, revision: number | null) => useQuery({
  queryKey: ["workload-revision", id, revision], queryFn: () => api.revision(id, revision!),
  enabled: !!id && revision != null,
});
export const useCapabilities = (model: string, hardware: string) => useQuery({
  queryKey: ["workload-capabilities", model, hardware],
  queryFn: ({ signal }) => api.capabilities(model, hardware, signal), retry: false,
});
export const useRun = (id: string) => useQuery({
  queryKey: ["workload-run", id], queryFn: () => api.run(id), enabled: !!id,
  refetchInterval: (query) => active(query.state.data?.status) ? 1000 : false,
});
export const usePlans = (id: string, ready: boolean) => useQuery({
  queryKey: ["workload-plans", id], queryFn: () => api.plans(id), enabled: !!id && ready,
});
export const useGraph = (run: string, plan: string, view: View, layer?: number) => useQuery({
  queryKey: ["workload-graph", run, plan, view, layer], queryFn: () => api.graph(run, plan, view, layer),
  enabled: !!run && !!plan,
});
export const useRanks = (run: string, plan: string, offset: number) => useQuery({
  queryKey: ["workload-ranks", run, plan, offset], queryFn: () => api.ranks(run, plan, offset), enabled: !!run && !!plan,
});
export const usePlacementRanks = (run: string, plan: string) => useQuery({
  queryKey: ["workload-placement-ranks", run, plan],
  queryFn: async () => {
    const first = await api.ranks(run, plan, 0);
    const items = [...first.items];
    let next = first.next_offset;
    while (next != null) {
      const page = await api.ranks(run, plan, next);
      if (page.result_hash !== first.result_hash) throw new Error("Placement result changed");
      items.push(...page.items);
      next = page.next_offset;
    }
    return items;
  },
  enabled: !!run && !!plan,
});
export const useLayers = (run: string, plan: string, rank?: number) => useInfiniteQuery({
  queryKey: ["workload-layers", run, plan, rank],
  queryFn: ({ pageParam }) => api.layers(run, plan, pageParam, rank), initialPageParam: 0,
  getNextPageParam: (page) => page.next_offset ?? undefined, enabled: !!run && !!plan,
});
export const useCommunication = (run: string, plan: string) => useInfiniteQuery({
  queryKey: ["workload-communication", run, plan],
  queryFn: ({ pageParam }) => api.communication(run, plan, pageParam), initialPageParam: 0,
  getNextPageParam: (page) => page.next_offset ?? undefined, enabled: !!run && !!plan,
});
export const useComparison = (refs: PlanRef[]) => useQuery({
  queryKey: ["workload-compare", refs], queryFn: () => api.compare(refs), enabled: refs.length >= 2 && refs.length <= 4,
});
