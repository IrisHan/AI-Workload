import { Link, useLocation } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import {
  useHardware,
  useHealth,
  useModels,
} from "../features/queries";
import { useDesigns } from "../features/workload-designer/queries";
import {
  Empty,
  ErrorState,
  Facts,
  Loading,
  PageTitle,
} from "../components/common";
import { time } from "../utils/format";

export function DashboardPage() {
  const models = useModels();
  const hardware = useHardware();
  const workloads = useDesigns();
  const designs = workloads.data?.pages.flatMap(page => page.items) ?? [];
  return (
    <>
      <PageTitle title="Workspace Overview">
        <Link className="button primary" to="/workloads/inference/new">
          New Inference Design
          <ArrowRight size={16} />
        </Link>
      </PageTitle>
      {models.isError || hardware.isError || workloads.isError ? (
        <ErrorState error={models.error ?? hardware.error ?? workloads.error} />
      ) : (
        <div className="metric-strip">
          <div>
            <span>Registered models</span>
            <strong>{models.data?.length ?? "-"}</strong>
          </div>
          <div>
            <span>GPU specifications</span>
            <strong>{hardware.data?.length ?? "-"}</strong>
          </div>
          <div>
            <span>Loaded designs</span>
            <strong>{workloads.data ? `${designs.length}${workloads.hasNextPage ? "+" : ""}` : "-"}</strong>
          </div>
          <div>
            <span>Runtime connection</span>
            <strong className="muted small-value">Not connected</strong>
          </div>
        </div>
      )}
      <section className="section">
        <h2>Planning workflow</h2>
        <div className="workflow-list">
          {[
            ["Model", "Registry and architecture", "/models", "Available"],
            [
              "Workload",
              "Inference design, parallelism and GPU placement",
              "/workloads",
              "Available",
            ],
            [
              "Resource",
              "Parallelism candidate search",
              "/planning/resources",
              "Planned",
            ],
            [
              "Topology",
              "Physical GPU placement",
              "/planning/topology",
              "Planned",
            ],
            [
              "Deployment",
              "Scheduler and runtime artifacts",
              "/planning/deployments",
              "Planned",
            ],
          ].map(([name, detail, to, status], i) => (
            <Link to={to} key={name}>
              <span className="step-number">
                {String(i + 1).padStart(2, "0")}
              </span>
              <strong>{name}</strong>
              <span className="workflow-detail">{detail}</span>
              <span className={status === "Available" ? "ok" : "muted"}>
                {status}
              </span>
              <ArrowRight size={16} />
            </Link>
          ))}
        </div>
      </section>
      <section className="section">
        <h2>Recent designs</h2>
        {workloads.isPending ? (
          <Loading />
        ) : designs.length ? (
          <div className="workflow-list">
            {designs
              .slice(0, 5)
              .map((w) => (
                <Link to={`/workloads/inference/${w.id}`} key={w.id}>
                  <strong>{w.name}</strong>
                  <span>{w.revision.input.mode}</span>
                  <span>{time(w.updated_at)}</span>
                  <ArrowRight size={16} />
                </Link>
              ))}
          </div>
        ) : (
          <Empty>No inference designs.</Empty>
        )}
      </section>
    </>
  );
}
export function AdministrationPage() {
  const health = useHealth();
  const models = useModels();
  return (
    <>
      <PageTitle title="Data & Versions" />
      <section className="section">
        <h2>System</h2>
        {health.isPending ? (
          <Loading />
        ) : health.isError ? (
          <ErrorState error={health.error} />
        ) : (
          <Facts
            rows={[
              ["API version", health.data.version],
              ["Database", String(health.data.database)],
              ["Status", health.data.status],
              [
                "Result semantics",
                "Analytical prediction; not runtime measurement",
              ],
            ]}
          />
        )}
      </section>
      <section className="section">
        <h2>Model source revisions</h2>
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
                  <th>Revision</th>
                  <th>Retrieved</th>
                </tr>
              </thead>
              <tbody>
                {models.data.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <Link to={`/models/${m.id}?tab=source`}>{m.name}</Link>
                    </td>
                    <td className="mono">{m.revision}</td>
                    <td>{time(m.retrieved_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
const future: Record<string, [string, string[]]> = {
  "/planning/roofline": [
    "Roofline Analyzer",
    [
      "Standalone operation comparison",
      "Layer-level HBM traffic",
      "Measured efficiency calibration",
    ],
  ],
  "/planning/resources": [
    "Resource Planner",
    [
      "TP / PP / DP candidate search",
      "Top-k memory-feasible plans",
      "Versioned scoring and constraints",
    ],
  ],
  "/planning/topology": [
    "Topology Planner",
    [
      "Cluster / rack / node / GPU catalog",
      "Network-aware placement",
      "2D and 3D topology",
    ],
  ],
  "/planning/compare": [
    "Scenario Compare",
    [
      "Performance and memory comparison",
      "Cost and communication comparison",
      "Immutable scenario inputs",
    ],
  ],
  "/planning/deployments": [
    "Deployment Planner",
    [
      "Kubernetes and Volcano artifacts",
      "vLLM and Megatron configuration",
      "Deployment validation",
    ],
  ],
  "/runtime": [
    "Runtime & Benchmarks",
    [
      "Jobs and benchmark ingestion",
      "Prediction versus observed results",
      "Calibration history",
    ],
  ],
};
export function FuturePage() {
  const { pathname } = useLocation();
  const [title, milestones] = future[pathname] ?? ["Page not found", []];
  return (
    <>
      <PageTitle title={title} />
      <div className="availability">Not implemented in v0.1</div>
      <section className="section">
        <h2>Planned scope</h2>
        <ul className="roadmap">
          {milestones.map((m) => (
            <li key={m}>{m}</li>
          ))}
        </ul>
      </section>
      <Link className="button" to="/workloads">
        {pathname === "/planning/roofline"
          ? "Open workload analysis"
          : "Back to Workload Center"}
        <ArrowRight size={16} />
      </Link>
    </>
  );
}
