import { createContext, useContext } from "react";
import type { Fact, Trace } from "../../types/workload";
import { CircleHelp } from "lucide-react";
import { bytes, number } from "./state";
import { useEvidence } from "./Evidence";
import { metricNames } from "./calculationEvidence";

export const DialogFacts = createContext(false);

export function factValue(value: unknown, unit = ""): string {
  if (value == null) return "Unavailable";
  if (Array.isArray(value)) return value.join(" × ");
  if (typeof value === "number") {
    if (unit === "bytes") return bytes(value);
    if (unit === "bytes/s") return `${number(value / 1e9)} GB/s`;
    if (unit === "s") return `${number(value * 1000)} ms`;
    if (unit === "fraction") return `${number(value * 100)}%`;
    return `${number(value)}${unit ? ` ${unit}` : ""}`;
  }
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

export function Facts({ items }: { items: Fact[] }) {
  const inDialog = useContext(DialogFacts);
  return <dl className="wc-values wc-facts">{items.map((f, i) => <div key={`${f.key}-${i}`}>
    <dt title={f.source_ref ?? undefined}>{f.label}{!(inDialog && f.source === "Derived") &&
      <> <span className="wc-source-tag">{f.source}</span></>}</dt>
    <dd>{factValue(f.value, f.unit)}{f.formula && <small>{f.formula}</small>}</dd>
  </div>)}</dl>;
}

export function CalculationTrace({ trace, label, inline = false }: { trace?: Trace; label?: string; inline?: boolean }) {
  const { open } = useEvidence();
  if (!trace) return null;
  const name = label ?? metricNames[trace.id] ?? trace.title;
  const icon = <button type="button" className="wc-help" aria-label={`Why ${name}`} title={`Show calculation · ${name}`}
    onClick={() => open(trace)}><CircleHelp size={12} aria-hidden="true" /></button>;
  return inline ? icon : <span className="wc-calculation"><span>{name}</span>
    <span className="wc-inline-value">{factValue(trace.result.value, trace.result.unit)}{icon}</span></span>;
}
