import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/planner";
import type { ModelMemoryInput } from "../types";

export const useModelMemory = (id: string, input: ModelMemoryInput) =>
  useQuery({ queryKey: ["model-memory", id, input], queryFn: () => api.modelMemory(id, input), retry: false });

export const useModels = () =>
  useQuery({ queryKey: ["models"], queryFn: api.models });
export const useModel = (id: string) =>
  useQuery({ queryKey: ["models", id], queryFn: () => api.model(id) });
export const useHardware = () =>
  useQuery({ queryKey: ["hardware"], queryFn: api.hardware });
export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30000,
  });
export const useAnalysis = (id: string) =>
  useQuery({ queryKey: ["analysis", id], queryFn: () => api.analyze(id) });
export const useStructure = (id: string) =>
  useQuery({ queryKey: ["structure", id], queryFn: () => api.structure(id), staleTime: 60000 });
export const useStructureHistory = (id: string, enabled: boolean) =>
  useQuery({ queryKey: ["structure-history", id], queryFn: () => api.structures(id), enabled });
export function useImportModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.importModel,
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });
}
