import { useQuery } from "@tanstack/react-query";
import { hardwareApi } from "../../api/hardware";

export const useHardwareCatalog = () => useQuery({ queryKey: ["hardware-catalog"], queryFn: hardwareApi.list });
export const useHardwareDetail = (id: string) => useQuery({
  queryKey: ["hardware-detail", id], queryFn: () => hardwareApi.detail(id),
});
export const useArchitecture = (id: string) => useQuery({
  queryKey: ["hardware-architecture", id], queryFn: () => hardwareApi.architecture(id),
});
export const useHardwareRoofline = (id: string, precision: string, sparse: boolean, intensity: number, enabled: boolean) =>
  useQuery({
    queryKey: ["hardware-roofline", id, precision, sparse, intensity],
    queryFn: () => hardwareApi.roofline(id, precision, sparse, intensity), enabled, retry: false,
  });
