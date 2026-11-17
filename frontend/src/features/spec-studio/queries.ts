import { useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { structureApi, type StructureOptions } from "../../api/structure";

export const useStudioSnapshot = (modelId: string, options: StructureOptions, snapshotId: string) =>
  useQuery({
    queryKey: ["studio", modelId, snapshotId || options],
    queryFn: ({ signal }) => snapshotId
      ? structureApi.snapshot(modelId, snapshotId, signal)
      : structureApi.create(modelId, options, signal),
    staleTime: Infinity,
  });

export const useStudioHistory = (modelId: string, currentId?: string) =>
  useQuery({
    queryKey: ["studio-history", modelId, currentId],
    queryFn: ({ signal }) => structureApi.history(modelId, signal),
    enabled: !!currentId,
  });

export const useStudioGlossary = () =>
  useQuery({ queryKey: ["studio-glossary"], queryFn: ({ signal }) => structureApi.glossary(signal), staleTime: Infinity });

export type AnalysisRequest = <T>(id: string, suffix: string, signal: AbortSignal, body?: unknown) => Promise<T>;
export function useAnalysisRequest(modelId: string): AnalysisRequest {
  return useCallback(<T,>(id: string, suffix: string, signal: AbortSignal, body?: unknown) =>
    structureApi.resource<T>(modelId, id, suffix, signal, body), [modelId]);
}
