import type { ReactNode } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";

export function Loading() {
  return (
    <div className="state" role="status">
      <LoaderCircle className="spin" size={18} />
      Loading...
    </div>
  );
}
export function ErrorState({ error }: { error: unknown }) {
  return (
    <div role="alert" className="error">
      <AlertCircle size={18} />
      <span>
        {error instanceof Error ? error.message : "Unable to load data."}
      </span>
    </div>
  );
}
export function PageTitle({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <header className="page-title">
      <h1>{title}</h1>
      <div className="actions">{children}</div>
    </header>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Facts({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="facts">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}
export function Bars({
  values,
  unit = "",
  formatter = (v: number) => v.toLocaleString(),
}: {
  values: Record<string, number>;
  unit?: string;
  formatter?: (v: number) => string;
}) {
  const max = Math.max(1, ...Object.values(values));
  return (
    <div className="bars">
      {Object.entries(values).map(([label, value], index) => (
        <div className="bar-row" key={label}>
          <span>{label.replaceAll("_", " ")}</span>
          <div className="bar-track">
            <div
              className={`bar-fill color-${index % 4}`}
              style={{ width: `${(value / max) * 100}%` }}
            />
          </div>
          <strong>
            {formatter(value)} {unit}
          </strong>
        </div>
      ))}
    </div>
  );
}
