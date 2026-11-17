import { useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  ExternalLink,
  Plus,
  Search,
  Upload,
} from "lucide-react";
import {
  useAnalysis,
  useImportModel,
  useModel,
  useModels,
} from "../features/queries";
import {
  Bars,
  Empty,
  ErrorState,
  Facts,
  Loading,
  PageTitle,
} from "../components/common";
import { compact, downloadJson, number, time } from "../utils/format";
import type { Model } from "../types";
import { StructureView } from "./structure";
import { ModelMemoryView } from "./model-memory";

function workloadFormula(model: Model): string {
  if (model.workload_supported === false) return "Not supported";
  if (model.architecture === "qwen2" || model.architecture === "llama") return "dense-gqa-1.0.0";
  return "morphology-workload-1.1.0";
}

export function ModelsPage() {
  const models = useModels();
  const [query, setQuery] = useState("");
  const visible = models.data?.filter((m) =>
    `${m.name} ${m.architecture}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <>
      <PageTitle title="Model Registry">
        <Link className="button primary" to="/models/import">
          <Plus size={16} />
          Import model
        </Link>
      </PageTitle>
      <div className="tabs">
        <span className="selected">All models</span>
        <Link to="/models/import">HuggingFace Import</Link>
      </div>
      <div className="toolbar">
        <label className="search">
          <Search size={17} />
          <input
            aria-label="Search models"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search models..."
          />
        </label>
        <span>{visible?.length ?? 0} models</span>
      </div>
      {models.isPending ? (
        <Loading />
      ) : models.isError ? (
        <ErrorState error={models.error} />
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Architecture</th>
                <th>Parameters</th>
                <th>Layers</th>
                <th>Attention / KV</th>
                <th>Source</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {visible?.map((m) => (
                <tr key={m.id}>
                  <td>
                    <Link className="model-name" to={`/models/${m.id}`}>
                      {m.name}
                    </Link>
                    <div className="muted">{m.provider}</div>
                  </td>
                  <td>
                    {m.architecture}
                    <div className="muted">
                      {m.parameter_scope === "main-with-vision" ? "Multimodal" : m.is_moe ? "MoE" : "Dense transformer"}
                    </div>
                  </td>
                  <td className="numeric">{compact(m.parameters)}</td>
                  <td>{m.layers}</td>
                  <td>
                    {m.attention_heads} / {m.kv_heads}
                  </td>
                  <td>
                    <span className="source">HuggingFace</span>
                    <div className="muted mono">{m.revision.slice(0, 10)}</div>
                  </td>
                  <td>
                    <Link
                      className="icon-button"
                      title={`Open ${m.name}`}
                      to={`/models/${m.id}`}
                    >
                      <ArrowRight size={17} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!visible?.length && <Empty>No matching models.</Empty>}
        </div>
      )}
    </>
  );
}

export function ImportModelPage() {
  const mutation = useImportModel();
  const navigate = useNavigate();
  const [repo, setRepo] = useState("Qwen/Qwen2.5-0.5B");
  const [revision, setRevision] = useState("");
  return (
    <>
      <Link className="back-link" to="/models">
        <ArrowLeft size={16} />
        Model Registry
      </Link>
      <PageTitle title="Import from HuggingFace" />
      <form
        className="form-section narrow"
        onSubmit={async (e) => {
          e.preventDefault();
          try {
            const model = await mutation.mutateAsync({
              repo_id: repo.trim(),
              ...(revision.trim() ? { revision: revision.trim() } : {}),
            });
            navigate(`/models/${model.id}`);
          } catch {
            /* rendered below */
          }
        }}
      >
        <label>
          Repository ID or URL
          <input
            required
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="Qwen/Qwen2.5-0.5B"
          />
        </label>
        <label>
          Revision
          <input
            value={revision}
            onChange={(e) => setRevision(e.target.value)}
            placeholder="main or commit SHA"
          />
        </label>
        <Facts
          rows={[
            ["Import artifact", "config.json"],
            ["Architecture rules", "Dense / MoE / MLA / DSA / compressed / vision"],
            ["Model weights", "Not downloaded"],
            ["Version policy", "Immutable commit SHA"],
          ]}
        />
        {mutation.isError && <ErrorState error={mutation.error} />}
        <div className="actions">
          <button className="primary" disabled={mutation.isPending}>
            <Upload size={16} />
            {mutation.isPending ? "Importing..." : "Import model"}
          </button>
          <Link className="button" to="/models">
            Cancel
          </Link>
        </div>
      </form>
    </>
  );
}

export function ModelDetailPage() {
  const { id = "" } = useParams();
  const model = useModel(id);
  const analysis = useAnalysis(id);
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "overview";
  if (model.isPending) return <Loading />;
  if (model.isError) return <ErrorState error={model.error} />;
  const m = model.data;
  return (
    <>
      <Link className="back-link" to="/models">
        <ArrowLeft size={16} />
        Model Registry
      </Link>
      <PageTitle title={m.name}>
        <button
          className="icon-button"
          title="Export model JSON"
          onClick={() => downloadJson(m, m.name.replaceAll("/", "-"))}
        >
          <Download size={18} />
        </button>
        {m.workload_supported !== false && <Link className="button primary" to={`/workloads/new?model=${m.id}`}>
          <Plus size={16} />
          Create workload
        </Link>}
      </PageTitle>
      <div className="tabs" role="tablist">
        {["overview", "architecture", "structure", "memory", "source"].map((t) => (
          <button
            role="tab"
            aria-selected={t === tab}
            className={t === tab ? "selected" : ""}
            key={t}
            onClick={() => setParams({ tab: t })}
          >
            {t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      {tab === "structure" && <StructureView key={id} modelId={id} />}
      {tab === "memory" && <ModelMemoryView key={id} model={m} />}
      {tab === "overview" && (
        <>
          <div className="metric-strip">
            <div>
              <span>Parameters</span>
              <strong>{compact(m.parameters)}</strong>
            </div>
            <div>
              <span>Transformer layers</span>
              <strong>{m.layers}</strong>
            </div>
            <div>
              <span>Hidden dimension</span>
              <strong>{number(m.hidden_size)}</strong>
            </div>
            <div>
              <span>Context capacity</span>
              <strong>{number(m.max_position_embeddings)}</strong>
            </div>
          </div>
          <div className="split-section">
            <section>
              <h2>Architecture specification</h2>
              <Facts
                rows={[
                  ["Family", m.architecture],
                  ["Attention heads", m.attention_heads],
                  ["KV heads", m.kv_heads],
                  ["FFN intermediate size", number(m.intermediate_size)],
                  ["Vocabulary", number(m.vocab_size)],
                  ["Tied embeddings", m.tie_word_embeddings ? "Yes" : "No"],
                  ["Source dtype", m.dtype],
                  ["Parameter count", number(m.parameters)],
                  ["Parameter scope", m.parameter_scope ?? "main"],
                  ["Active parameters", m.active_parameters == null ? "Not recorded" : number(m.active_parameters)],
                  ["Workload formula", workloadFormula(m)],
                ]}
              />
            </section>
            <section>
              <h2>Parameter distribution</h2>
              {analysis.isPending ? (
                <Loading />
              ) : analysis.isError ? (
                <ErrorState error={analysis.error} />
              ) : (
                <Bars
                  values={analysis.data.parameter_breakdown}
                  formatter={compact}
                />
              )}
            </section>
          </div>
          {!!m.support_notes?.length && <ul className="assumptions">{m.support_notes.map(note => <li key={note}>{note}</li>)}</ul>}
        </>
      )}
      {tab === "architecture" && (
        <section className="section">
          <h2>Transformer structure</h2>
          <div className="architecture-flow">
            {[
              "Token embedding",
              `Transformer block x ${m.layers}`,
              "RMSNorm",
              "LM head",
            ].map((s, i) => (
              <div key={s}>
                <span>{s}</span>
                {i < 3 && <ArrowRight size={20} />}
              </div>
            ))}
          </div>
          {analysis.isPending ? (
            <Loading />
          ) : analysis.isError ? (
            <ErrorState error={analysis.error} />
          ) : (
            <>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Operation</th>
                      <th>Input shape</th>
                      <th>Output shape</th>
                      <th>Parameters</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.data.layers.map((layer, i) => (
                      <tr key={`${layer.name}-${i}`}>
                        <td>{layer.name}</td>
                        <td className="mono">{String(layer.input_shape)}</td>
                        <td className="mono">{String(layer.output_shape)}</td>
                        <td>{number(layer.parameters)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <h2>Accounting assumptions</h2>
              <ul className="assumptions">
                {analysis.data.assumptions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
      {tab === "source" && (
        <section className="section">
          <h2>Provenance</h2>
          <Facts
            rows={[
              [
                "Repository",
                <a
                  href={`https://huggingface.co/${m.repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {m.repo_id} <ExternalLink size={13} />
                </a>,
              ],
              ["Revision", <code>{m.revision}</code>],
              ["Config SHA-256", <code>{m.config_sha256}</code>],
              ["Retrieved", time(m.retrieved_at)],
              ["Model version", m.version],
              [
                "Source artifact",
                <a href={m.source_url} target="_blank" rel="noreferrer">
                  config.json <ExternalLink size={13} />
                </a>,
              ],
            ]}
          />
          <h2>Original configuration</h2>
          <pre>{JSON.stringify(m.raw_config, null, 2)}</pre>
          <h2>Checkpoint comparison</h2>
          <pre>{JSON.stringify(m.source_metadata ?? {}, null, 2)}</pre>
        </section>
      )}
    </>
  );
}
