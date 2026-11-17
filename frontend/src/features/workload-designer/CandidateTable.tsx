import type { Plan, PlanRef, Plans } from "../../types/workload";
import { bytes, number } from "./state";

export function CandidateTable({ data, selected, select, compare, toggle }: {
  data?: Plans; selected?: string; select: (p: Plan) => void; compare: PlanRef[]; toggle: (p: Plan) => void;
}) {
  return <div className="wc-candidates">
    <div className="wc-toolbar"><h3>Candidates</h3></div>
    {!data?.items.length ? <p className="muted">运行后展示候选；无可行方案时请检查约束与拒绝原因。</p> :
      <div className="table-wrap"><table><thead><tr><th>Compare</th><th>Plan</th><th>TP / PP / EP / DP</th><th>GPUs</th>
        <th>Peak / GPU</th><th>Capacity</th><th>TTFT</th><th>ITL</th><th>Execution</th></tr></thead>
        <tbody>{data.items.map(p => <tr key={p.id} className={selected === p.id ? "selected" : ""}>
          <td><input type="checkbox" aria-label={`Compare ${p.candidate_key}`}
            checked={compare.some(r => r.plan_id === p.id)} onChange={() => toggle(p)} /></td>
          <td><button onClick={() => select(p)}>{p.summary.recommended ? "★ " : ""}Plan {p.rank}</button></td>
          <td>{[p.summary.candidate.tp, p.summary.candidate.pp, p.summary.candidate.ep, p.summary.candidate.dp].join(" / ")}</td>
          <td>{p.summary.gpu_count}</td><td>{bytes(p.summary.memory.peak_bytes)}</td>
          <td className={p.summary.capacity_status === "fail" ? "bad" : ""}>{p.summary.capacity_status}</td>
          <td>{number(p.summary.metric_values.ttft_ms?.value)} ms</td><td>{number(p.summary.metric_values.itl_ms?.value)} ms</td>
          <td>{p.summary.execution_status}</td></tr>)}</tbody></table></div>}
  </div>;
}
