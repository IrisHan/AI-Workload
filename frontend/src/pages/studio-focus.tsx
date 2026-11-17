import { useParams } from "react-router-dom";
import { StructureView } from "./structure";

export function StudioFocusPage() {
  const { id = "" } = useParams();
  return <StructureView modelId={id} variant="focus" />;
}
