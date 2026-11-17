import { useEffect, useRef, useState } from "react";
import type { HardwareRecord, RooflineResult } from "../../api/hardware";
import { ErrorState, Facts, Loading } from "../../components/common";
import { number } from "../../utils/format";
import { useHardwareRoofline } from "./queries";

function Plot({ data }: { data: RooflineResult }) {
  const container = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(760);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const height = 300, left = 55, right = 24, top = 22, bottom = 48;
  const minX = Math.log10(data.curve[0].intensity);
  const maxX = Math.log10(data.curve[data.curve.length - 1].intensity);
  const minY = Math.log10(Math.min(...data.curve.map(p => p.performance_tops))) - 0.15;
  const maxY = Math.log10(data.peak_tops) + 0.3;
  const x = (v: number) => left + (Math.log10(v) - minX) / (maxX - minX) * (width - left - right);
  const y = (v: number) => height - bottom - (Math.log10(v) - minY) / (maxY - minY) * (height - top - bottom);
  const ticks = Array.from({ length: Math.ceil(maxX) - Math.ceil(minX) + 1 }, (_, i) => 10 ** (Math.ceil(minX) + i));
  return <div className="hw-plot" ref={container}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${data.precision.toUpperCase()} theoretical roofline`}>
      <title>HBM / GDDR theoretical roofline</title>
      {ticks.filter((t, i) => Math.log10(t) <= maxX && (width >= 500 || i % 2 === 0)).map(t => <g key={t}>
        <line x1={x(t)} x2={x(t)} y1={top} y2={height - bottom} stroke="#e3e6e8" />
        <text x={x(t)} y={height - bottom + 20} textAnchor="middle">{number(t, 3)}</text>
      </g>)}
      {[data.peak_tops / 100, data.peak_tops / 10, data.peak_tops].filter(v => Math.log10(v) >= minY).map(v => <g key={v}>
        <line x1={left} x2={width - right} y1={y(v)} y2={y(v)} stroke="#e3e6e8" />
        <text x={left - 8} y={y(v) + 4} textAnchor="end">{number(v)}</text>
      </g>)}
      <line x1={x(data.ridge_point)} x2={x(data.ridge_point)} y1={top} y2={height - bottom} stroke="#9d7791" strokeDasharray="4 4" />
      <polyline fill="none" stroke="#087f68" strokeWidth="3" points={data.curve.map(p => `${x(p.intensity)},${y(p.performance_tops)}`).join(" ")} />
      <circle cx={x(data.operating_point.intensity)} cy={y(data.operating_point.performance_tops)} r="5" fill="#ba6a26" />
      <text x={left} y="14">{data.unit}</text>
      <text x={width / 2} y={height - 5} textAnchor="middle">{width < 500 ? "AI" : "Arithmetic intensity"} ({data.precision === "int8" ? "op" : "FLOP"}/byte), log scale</text>
    </svg>
  </div>;
}

export function Roofline({ hardware: h }: { hardware: HardwareRecord }) {
  const [precision, setPrecision] = useState("bf16");
  const [sparse, setSparse] = useState(false);
  const [input, setInput] = useState("64");
  const intensity = Number(input);
  const valid = input.trim() !== "" && Number.isFinite(intensity) && intensity > 0 && intensity <= 1e9;
  const roofline = useHardwareRoofline(h.id, precision, sparse, intensity, valid);
  return <section className="section">
    <div className="hw-controls">
      <label>Precision<select aria-label="Precision" value={precision} onChange={e => setPrecision(e.target.value)}>
        {["fp32", "tf32", "fp16", "bf16", "fp8", "fp4", "int8"].map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
      </select></label>
      <label>Arithmetic intensity<input type="number" min="0.000001" max="1000000000" step="any" value={input} onChange={e => setInput(e.target.value)} /></label>
      <label className="checkbox"><input type="checkbox" checked={sparse} onChange={e => setSparse(e.target.checked)} />Structured sparsity</label>
    </div>
    {!valid ? <div role="alert" className="error">Arithmetic intensity must be finite, greater than 0 and at most 1e9.</div> :
      roofline.isPending ? <Loading /> : roofline.isError ? <ErrorState error={roofline.error} /> :
        <>
          <Plot data={roofline.data} />
          <div className="hw-legend"><span className="hw-line-key">Theoretical ceiling</span><span>Ridge point</span><span>Selected intensity</span></div>
          <Facts rows={[
            ["Peak compute", `${number(roofline.data.peak_tops)} ${roofline.data.unit}`],
            ["Ridge point", `${number(roofline.data.ridge_point)} ${precision === "int8" ? "op" : "FLOP"}/byte`],
            ["Performance upper bound", `${number(roofline.data.operating_point.performance_tops)} ${roofline.data.unit}`],
            ["Bottleneck", roofline.data.bottleneck],
            ["Formula", roofline.data.formula_version],
          ]} />
          <ul className="assumptions">{roofline.data.assumptions.map(a => <li key={a}>{a}</li>)}</ul>
        </>}
  </section>;
}
