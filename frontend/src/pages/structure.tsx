import { lazy, Suspense, useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { useSearchParams } from "react-router-dom";
import { RefreshCw } from "lucide-react";
import { ErrorState, Loading } from "../components/common";
import { useStudioHistory, useStudioSnapshot } from "../features/spec-studio/queries";
import type { Analysis } from "../features/spec-studio/analysis";
import type { StructureOptions } from "../api/structure";
import "../features/spec-studio/studio.css";

const Studio = lazy(() => import("../features/spec-studio/Studio"));

export function StructureView({ modelId, variant = "embedded" }: { modelId: string; variant?: "embedded" | "focus" }) {
  const [params, setParams] = useSearchParams();
  const [snapshotId, setSnapshotId] = useState(params.get("snapshot") || "");
  const [options, setOptions] = useState<StructureOptions>({
    kv_dtype: ["bf16", "fp16", "fp32", "fp8"].includes(params.get("kv_dtype") || "")
      ? params.get("kv_dtype")! : "bf16",
    include_mtp: params.get("mtp") === "true",
  });
  const [draft, setDraft] = useState(options);
  const current = useStudioSnapshot(modelId, options, snapshotId);
  const history = useStudioHistory(modelId, current.data?.id);
  useEffect(() => {
    if (!current.data) return;
    const analysis = current.data.analysis as unknown as Analysis;
    setDraft({ kv_dtype: analysis.serving.kv_layout.dtype_assumed, include_mtp: analysis.serving.include_mtp });
  }, [current.data]);

  function apply() {
    const query = new URLSearchParams(params);
    query.delete("snapshot"); query.delete("group"); query.delete("module"); query.delete("branch");
    query.set("kv_dtype", draft.kv_dtype); query.set("mtp", String(draft.include_mtp));
    setParams(query, { replace: true });
    setSnapshotId(""); setOptions({ ...draft });
    if (!snapshotId && options.kv_dtype === draft.kv_dtype && options.include_mtp === draft.include_mtp)
      void current.refetch();
  }
  return <section className={`spec-studio${variant === "focus" ? " focus-page" : ""}`} aria-label="Spec Studio model workspace">
    <div className="studio-toolbar">
      <label>历史快照<select aria-label="结构历史快照" value={current.data?.id || ""}
        onChange={event => setSnapshotId(event.target.value)}>
        {!current.data && <option value="">加载中</option>}
        {(history.data?.length ? history.data : current.data ? [current.data] : []).map(snapshot =>
          <option key={snapshot.id} value={snapshot.id}>{snapshot.created_at} · {snapshot.id.slice(0, 8)}</option>)}
      </select></label>
      <label>KV dtype<select aria-label="KV dtype" value={draft.kv_dtype}
        onChange={event => setDraft({ ...draft, kv_dtype: event.target.value })}>
        {["bf16", "fp16", "fp32", "fp8"].map(dtype => <option key={dtype}>{dtype}</option>)}
      </select></label>
      <label className="check-field"><input aria-label="MTP" type="checkbox" checked={draft.include_mtp}
        onChange={event => setDraft({ ...draft, include_mtp: event.target.checked })} />MTP</label>
      <button disabled={current.isFetching} onClick={apply}><RefreshCw size={15} />应用配置</button>
      {current.data && <span className="shadow-status">
        {current.data.comparison.matched ? "Shadow comparison matched" : `Parameter delta: ${current.data.comparison.parameter_delta}`}
      </span>}
    </div>
    {current.isPending ? <Loading /> : current.isError ? <ErrorState error={current.error} /> :
      <Suspense fallback={<Loading />}><ReactFlowProvider key={current.data.id}>
        <Studio key={`${modelId}:${current.data.id}`} snapshot={current.data} />
      </ReactFlowProvider></Suspense>}
    {history.isError && <ErrorState error={history.error} />}
  </section>;
}
