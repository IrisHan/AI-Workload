import { useState } from "react";
import type { RankDetail } from "../../types/workload";
import { MemoryRows } from "./Inspector";
import { bytes, number } from "./state";

export function MemoryTimeline({ rank, coverage }: { rank: RankDetail; coverage: string }) {
  const timeline = rank.event_memory.timeline;
  const [selected, setSelected] = useState<number | null>(null);
  if (!timeline.length) return <p>没有时间轴数据。</p>;
  const peakIndex = timeline.reduce((best, p, i) => p.total_bytes > timeline[best].total_bytes ? i : best, 0);
  const index = Math.min(selected ?? peakIndex, timeline.length - 1);
  const point = timeline[index], until = timeline.at(-1)!.at || 1;
  const max = Math.max(rank.capacity_bytes, ...timeline.map(p => p.total_bytes));
  const points = timeline.map(p => `${p.at / until * 780 + 10},${140 - p.total_bytes / max * 120}`).join(" ");
  return <section className="wc-timeline"><div className="wc-toolbar"><h3>Memory Timeline · GPU {rank.rank}</h3>
    <span>{coverage} · Modeled</span><button onClick={() => setSelected(peakIndex)}>Locate Peak</button></div>
    <svg viewBox="0 0 800 150" role="img" aria-label="Modeled GPU memory over time">
      <line x1="10" x2="790" y1={140 - rank.capacity_bytes / max * 120} y2={140 - rank.capacity_bytes / max * 120} stroke="#b76348" strokeDasharray="4 4" />
      <polyline points={points} fill="none" stroke="#176849" strokeWidth="2" />
      <circle cx={point.at / until * 780 + 10} cy={140 - point.total_bytes / max * 120} r="4" fill="#176849" />
    </svg>
    <label>Time {number(point.at * 1000)} ms · {bytes(point.total_bytes)}
      <input aria-label="Memory Time" type="range" min="0" max={timeline.length - 1} value={index} onChange={e => setSelected(Number(e.target.value))} /></label>
    <MemoryRows components={point.components} />
    <p className="muted">同一时刻的七项显存。{rank.event_memory.timeline_sampled ? "曲线已采样并保留峰值。" : ""}
      低负载运行峰值不能证明部署容量；容量判定使用准入上界。</p>
  </section>;
}
