import type { Plan, RankDetail } from "../../types/workload";
import { bytes } from "./state";

export function PlacementCanvas({ plan, ranks, selectedRank, onSelect }: {
  plan: Plan; ranks: RankDetail[]; selectedRank: number; onSelect: (rank: number) => void;
}) {
  const placement = plan.summary.placement.filter(r => r.d === 0);
  const nodes = [...new Set(placement.map(r => r.node))];
  return <div className="wc-placement">
    <p className="wc-topology-scope">DP 0 / {plan.summary.candidate.dp} replicas · {placement.length} GPUs</p>
    {nodes.map(node => <section key={node}><h3>Node {node}</h3><div className="wc-gpus">
      {placement.filter(r => r.node === node).map(r => {
        const memory = ranks.find(m => m.rank === r.rank);
        return <button key={r.rank} aria-pressed={selectedRank === r.rank} className={selectedRank === r.rank ? "selected" : ""}
          onClick={() => onSelect(r.rank)}>
          <strong>GPU {r.gpu} <small>R{r.rank}</small></strong>
          <span>TP {r.t} · PP {r.p}</span><span>AG {r.e}</span>
          {memory ? <><span>{bytes(memory.peak_bytes)}</span><span>of {bytes(memory.capacity_bytes)}</span>
            <meter aria-label={`GPU ${r.rank} 准入显存`} value={memory.peak_bytes} max={memory.capacity_bytes} />
          </> : <span>点击加载显存</span>}
        </button>;
      })}
    </div></section>)}
  </div>;
}
