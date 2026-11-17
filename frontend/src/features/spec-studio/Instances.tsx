import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import type { AnalysisRequest } from "./queries";

interface Page { items: { index: number; group_id: string; template_id: string }[];
  total: number; offset: number; next_offset: number | null; }

export function Instances({ analysisId, analysisRequest, onSelect }: { analysisId: string; analysisRequest: AnalysisRequest; onSelect: (group: string, index: number) => void }) {
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setPage(null); setError("");
    void analysisRequest<Page>(analysisId, `instances?offset=${offset}&limit=20`, controller.signal)
      .then(setPage).catch(cause => { if (!controller.signal.aborted) setError(String(cause.message)); });
    return () => controller.abort();
  }, [analysisId, offset, analysisRequest]);
  return <details className="execution"><summary>实例分页</summary>
    {error && <p role="alert">{error}</p>}
    {!page && !error && <p role="status">加载中…</p>}
    {page?.items.map(item => <button key={item.index} onClick={() => onSelect(item.group_id, item.index)}>
      <span>L{item.index}</span><span>{item.group_id}</span></button>)}
    {page && <div className="pagination">
      <button aria-label="上一页实例" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 20))}><ChevronLeft size={16} /></button>
      <span>{offset + 1}–{Math.min(offset + 20, page.total)} / {page.total}</span>
      <button aria-label="下一页实例" disabled={page.next_offset === null} onClick={() => setOffset(page.next_offset!)}><ChevronRight size={16} /></button>
    </div>}
  </details>;
}
