import { useState } from "react";
import { Calculator, Download } from "lucide-react";
import { useModelMemory } from "../features/queries";
import { ErrorState, Facts, Loading } from "../components/common";
import { downloadJson, gib } from "../utils/format";
import type { Model, ModelMemoryInput } from "../types";

export function ModelMemoryView({ model }: { model: Model }) {
  const [input, setInput] = useState<ModelMemoryInput>({
    batch_size: 1, context_length: Math.min(4096, model.max_position_embeddings),
    dtype: "bf16", include_training_state: false,
  });
  const [draft, setDraft] = useState(input);
  const result = useModelMemory(model.id, input);
  return <section className="section">
    <h2>Static memory</h2>
    <form className="form-section memory-form" onSubmit={event => { event.preventDefault(); setInput({ ...draft }); }}>
      <div className="memory-params">
        <label>Batch size<input type="number" min={1} max={4096} required value={draft.batch_size}
          onChange={event => setDraft({ ...draft, batch_size: Number(event.target.value) })} /></label>
        <label>Context tokens<input type="number" min={1} max={model.max_position_embeddings} required
          value={draft.context_length} onChange={event => setDraft({ ...draft, context_length: Number(event.target.value) })} /></label>
        <label>Resident dtype<select value={draft.dtype}
          onChange={event => setDraft({ ...draft, dtype: event.target.value as "bf16" | "fp32" })}>
          <option value="bf16">BF16</option><option value="fp32">FP32</option>
        </select></label>
      </div>
      <label className="checkbox"><input type="checkbox" checked={draft.include_training_state}
        onChange={event => setDraft({ ...draft, include_training_state: event.target.checked })} />Full-parameter Adam state</label>
      <div className="actions"><button className="primary" disabled={result.isFetching}><Calculator size={16} />Calculate</button>
        {result.data && <button type="button" className="icon-button" title="Export memory JSON"
          onClick={() => downloadJson(result.data, model.name + "-memory")}><Download size={17} /></button>}</div>
    </form>
    {result.isPending ? <Loading /> : result.isError ? <ErrorState error={result.error} /> :
      <>
        <Facts rows={[
          ["Weight residency", gib(result.data.weight_bytes)],
          ["KV / recurrent state", gib(result.data.kv_cache_bytes)],
          ["Gradients", gib(result.data.gradient_bytes)],
          ["Optimizer / master", gib(result.data.optimizer_bytes)],
          ["Integer buffers", gib(result.data.integer_buffer_bytes ?? 0)],
          ["Static subtotal", gib(result.data.static_total_bytes)],
          ["Activation peak", "Not modeled"], ["Runtime peak", "Not modeled"],
          ["Formula", result.data.formula_version],
        ]} />
        <ul className="assumptions">{result.data.assumptions.map(note => <li key={note}>{note}</li>)}</ul>
      </>}
  </section>;
}
