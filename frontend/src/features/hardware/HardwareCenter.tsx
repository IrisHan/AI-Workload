import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { ArrowDown, ArrowLeft, ArrowRight, ArrowUpDown, Cpu, Download, ExternalLink, Layers3, RefreshCw, Search } from "lucide-react";
import { hardwareApi, type Evidence, type HardwareRecord } from "../../api/hardware";
import { Empty, ErrorState, Facts, Loading, PageTitle } from "../../components/common";
import { downloadJson, number, time } from "../../utils/format";
import { useHardwareCatalog, useHardwareDetail } from "./queries";
import { Roofline } from "./Roofline";
import { Architecture } from "./Architecture";
import "./hardware.css";

const value = (v: number | null | undefined) => v == null ? "-" : number(v);
const confidence: Record<string, string> = {
  official: "官方", vendor_system: "厂商资料存档", user_reference: "用户参考", assumed: "假设", unknown: "未核验",
};
function EvidenceNote({ evidence }: { evidence: Evidence }) {
  return <span className={`hw-evidence ${evidence.confidence}`} title={evidence.note}>{confidence[evidence.confidence] ?? evidence.confidence}</span>;
}
function Retry({ retry }: { retry: () => void }) {
  return <button onClick={retry}><RefreshCw size={15} />Retry</button>;
}
function Metric({ v, evidence }: { v: number | null | undefined; evidence?: Evidence }) {
  return <><span className="numeric">{value(v)}</span>{evidence && !["official", "vendor_system"].includes(evidence.confidence) && v != null &&
    <div><EvidenceNote evidence={evidence} /></div>}</>;
}

export function HardwareCatalog() {
  const catalog = useHardwareCatalog();
  const [params, setParams] = useSearchParams();
  const [descending, setDescending] = useState(false);
  const search = params.get("q") ?? "", vendor = params.get("vendor") ?? "", form = params.get("form") ?? "";
  const sort = params.get("sort") ?? "name";
  const update = (key: string, v: string) => {
    const next = new URLSearchParams(params); if (v) next.set(key, v); else next.delete(key); setParams(next, { replace: true });
  };
  const rows = (catalog.data ?? []).filter(h =>
    `${h.name} ${h.architecture} ${h.vendor}`.toLowerCase().includes(search.toLowerCase()) &&
    (!vendor || h.vendor === vendor) && (!form || h.form_factor === form),
  ).sort((a, b) => {
    const compared = sort === "name" ? a.name.localeCompare(b.name) :
      sort === "memory" ? a.memory_bytes - b.memory_bytes : (a.bf16_tflops ?? -1) - (b.bf16_tflops ?? -1);
    return descending ? -compared : compared;
  });
  return <div className="hardware-center">
    <PageTitle title="Hardware Catalog">
      <button className="icon-button" title="Export GPU catalog JSON" disabled={!catalog.data} onClick={() => downloadJson(rows, "hardware-catalog")}><Download size={18} /></button>
    </PageTitle>
    <div className="tabs"><span className="selected">GPU Catalog</span></div>
    <div className="hw-controls">
      <label className="search"><Search size={16} /><input aria-label="Search GPUs" value={search} placeholder="Search GPU or architecture" onChange={e => update("q", e.target.value)} /></label>
      <label>Vendor<select aria-label="Vendor" value={vendor} onChange={e => update("vendor", e.target.value)}><option value="">All vendors</option><option>NVIDIA</option><option>Huawei</option></select></label>
      <label>Form factor<select aria-label="Form factor" value={form} onChange={e => update("form", e.target.value)}><option value="">All forms</option><option>SXM</option><option>PCIe</option><option>Module</option></select></label>
      <label>Sort by<select aria-label="Sort by" value={sort} onChange={e => update("sort", e.target.value)}><option value="name">GPU name</option><option value="memory">Memory</option><option value="compute">BF16 compute</option></select></label>
      <button className="icon-button" title={descending ? "Sort ascending" : "Sort descending"} onClick={() => setDescending(!descending)}><ArrowUpDown size={16} /></button>
      <span className="muted">{rows.length} GPU SKUs</span>
    </div>
    {catalog.isPending ? <Loading /> : catalog.isError ? <><ErrorState error={catalog.error} /><Retry retry={() => { void catalog.refetch(); }} /></> :
      <div className="table-scroll"><table className="hw-table">
        <thead><tr><th>GPU SKU</th><th>BF16 dense<br />TFLOPS</th><th>FP8 dense<br />TFLOPS</th><th>Memory<br />GB</th><th>Memory BW<br />GB/s</th><th>GPU link<br />GB/s bidir.</th><th>NIC / GPU<br />GB/s one-way</th><th>PCIe<br />GB/s bidir.</th><th>CNY/day/chip<br />Reference</th><th /></tr></thead>
        <tbody>{rows.map(h => <tr key={h.id}>
          <td><Link className="model-name" to={`/hardware/${h.id}`}>{h.name}</Link><div className="muted">{h.architecture} / {h.form_factor}</div></td>
          <td><Metric v={h.bf16_tflops} evidence={h.compute.bf16?.evidence} /></td>
          <td><Metric v={h.compute.fp8?.dense} evidence={h.compute.fp8?.evidence} /></td>
          <td><Metric v={h.memory_bytes / 1e9} evidence={h.field_evidence.memory_bytes} /></td>
          <td><Metric v={h.memory_bandwidth_bytes_s / 1e9} evidence={h.field_evidence.memory_bandwidth_bytes_s} /></td>
          <td><Metric v={h.interconnect[0]?.bandwidth_gb_s} evidence={h.interconnect[0]?.evidence} /><div className="muted">{h.interconnect[0]?.name}</div></td>
          <td title={h.node.evidence.note}>{value(h.node.nic_gb_s_per_gpu)}<div className="muted">{h.node.name}</div></td>
          <td><Metric v={h.interconnect.find(l => l.name === "PCIe")?.bandwidth_gb_s} evidence={h.interconnect.find(l => l.name === "PCIe")?.evidence} /></td>
          <td>{value(h.cost.cny_per_day)}{h.cost.cny_per_day != null && <div><EvidenceNote evidence={h.cost.evidence} /></div>}</td>
          <td><Link className="icon-button" title={`Open ${h.name}`} to={`/hardware/${h.id}`}><ArrowRight size={16} /></Link></td>
        </tr>)}</tbody>
      </table>{!rows.length && <Empty>No matching GPUs.<button onClick={() => setParams({})}>Clear filters</button></Empty>}</div>}
    <p className="hw-accounting">GB = 10⁹ bytes。算力默认稠密；GPU link / PCIe 为双向，NIC 为指定节点单向分摊。价格为用户参考，日期与采购口径未确认；“-”表示未核验或不支持，0 表示已记录无此链路。</p>
  </div>;
}

function FormFactor({ hardware: h }: { hardware: HardwareRecord }) {
  return <figure className="hw-form-factor" aria-label="Accelerator packaging diagram">
    <div className="hw-package"><Cpu size={44} strokeWidth={1.2} /><strong>{h.form_factor} {h.form_factor === "PCIe" ? "板卡" : "模块"}</strong><span>{h.memory_type}</span></div>
    <div className="hw-package-link"><ArrowDown size={20} /><span>{h.form_factor === "PCIe" ? "PCIe 插槽" : "专用基板连接"}</span></div>
    <div className="hw-baseboard"><Layers3 size={25} strokeWidth={1.3} /><span>{h.form_factor === "SXM" ? "HGX / OEM 基板" : "服务器主板 / OEM 系统"}</span></div>
    <figcaption>封装与连接关系示意，非实物照片或芯片 floorplan；具体外形、接口和散热以对应厂商资料为准。</figcaption>
  </figure>;
}

function Specifications({ h }: { h: HardwareRecord }) {
  return <>
    <section className="section"><h2>Compute capabilities</h2>
      <div className="table-scroll"><table><thead><tr><th>Precision</th><th>Dense</th><th>Sparse</th><th>Unit</th><th>Evidence / status</th></tr></thead>
        <tbody>{Object.entries(h.compute).map(([p, c]) => <tr key={p}><td>{p.toUpperCase()}</td><td>{value(c.dense)}</td><td>{value(c.sparse)}</td><td>{c.unit}</td><td><EvidenceNote evidence={c.evidence} /> / {c.status}</td></tr>)}</tbody></table></div>
    </section>
    <section className="section"><h2>Memory & power</h2><Facts rows={[
      ["Memory", `${value(h.memory_bytes / 1e9)} GB / ${h.memory_type}`],
      ["Memory bandwidth", <>{value(h.memory_bandwidth_bytes_s / 1e9)} GB/s <EvidenceNote evidence={h.field_evidence.memory_bandwidth_bytes_s} /></>],
      ["Maximum GPU power", <>{value(h.power_watts)} W <EvidenceNote evidence={h.field_evidence.power_watts} /></>],
      ["Planning precisions", h.planner_precisions.join(", ").toUpperCase() || "待核验，暂不开放正式 Workload 计算"],
    ]} /></section>
    <section className="section"><h2>Interconnect & node context</h2><Facts rows={[
      ...h.interconnect.map(l => [l.name, <>{value(l.bandwidth_gb_s)} GB/s ({l.direction})<p className="muted">{l.scope} / {l.evidence.note}</p></>] as [string, React.ReactNode]),
      ["Reference system", h.node.name], ["Accelerators / node", value(h.node.accelerators)],
      ["NIC per GPU (one-way)", `${value(h.node.nic_gb_s_per_gpu)} GB/s`],
      ["Full-mesh", h.node.full_mesh == null ? "未核验" : h.node.full_mesh ? "Yes (reference system)" : "No"],
      ["Node scope", h.node.evidence.note],
    ]} /></section>
    <section className="section"><h2>Cost reference</h2><Facts rows={[
      ["CNY/day/chip", value(h.cost.cny_per_day)], ["Basis", h.cost.basis], ["Quote date", h.cost.as_of ?? "未提供"],
      ["Assumptions", h.cost.evidence.note],
    ]} /></section>
  </>;
}

export function HardwareDetail() {
  const { id = "" } = useParams();
  const query = useHardwareDetail(id);
  const [params, setParams] = useSearchParams();
  const [exportError, setExportError] = useState<Error | null>(null);
  const [exporting, setExporting] = useState(false);
  const tab = ["specs", "roofline", "architecture", "sources"].includes(params.get("tab") ?? "") ? params.get("tab")! : "specs";
  if (query.isPending) return <Loading />;
  if (query.isError) return <><Link className="back-link" to="/hardware"><ArrowLeft size={16} />GPU Catalog</Link><ErrorState error={query.error} /><Retry retry={() => { void query.refetch(); }} /></>;
  const h = query.data;
  return <div className="hardware-center">
    <Link className="back-link" to="/hardware"><ArrowLeft size={16} />GPU Catalog</Link>
    <PageTitle title={h.name}>
      <button className="icon-button" title="Export hardware JSON" disabled={exporting} onClick={async () => {
        setExportError(null); setExporting(true);
        try { downloadJson(await hardwareApi.export(id), h.id); } catch (e) { setExportError(e as Error); } finally { setExporting(false); }
      }}><Download size={18} /></button>
      {!h.legacy && h.planner_precisions.includes("bf16") && <Link className="button" to={`/workloads/new?hardware=${h.id}`}>Create workload<ArrowRight size={16} /></Link>}
    </PageTitle>
    <div className="hw-meta"><span>{h.vendor} / {h.architecture} / {h.form_factor}</span><span>{h.version}</span><span>{h.legacy ? "Historical version" : "Catalog version"}</span></div>
    {exportError && <ErrorState error={exportError} />}
    <div className="tabs" role="tablist" aria-label="Hardware details">{Object.entries({ specs: "Specifications", roofline: "Roofline", architecture: "Architecture", sources: "Sources & caveats" }).map(([key, label]) =>
      <button role="tab" aria-selected={tab === key} className={tab === key ? "selected" : ""} key={key} onClick={() => setParams({ tab: key })}>{label}</button>)}</div>
    {tab === "specs" && <Specifications h={h} />}
    {tab === "roofline" && <Roofline hardware={h} />}
    {tab === "architecture" && <Architecture key={h.id} hardware={h} packaging={<FormFactor hardware={h} />} />}
    {tab === "sources" && <>
      <section className="section"><h2>Accounting & differences</h2><p>{h.notes}</p><ul className="hw-caveats">{h.caveats.map(c => <li key={c}>{c}</li>)}</ul>
        <Facts rows={[["Content SHA256", <code>{h.content_sha256}</code>], ["Retrieved", time(h.retrieved_at)]]} /></section>
      {h.sources.map(s => <section className="section hw-source" key={s.id}><h2>{s.id} / {s.title}</h2>
        {s.url && <a href={s.url} target="_blank" rel="noreferrer">{s.url} <ExternalLink size={13} /></a>}
        <Facts rows={[["Archived file", s.artifact], ["Retrieved", time(s.retrieved_at)], ["Source SHA256", <code>{s.sha256 ?? "Legacy: unavailable"}</code>]]} />
      </section>)}
      <section className="section"><h2>Field-level evidence</h2><Facts rows={[
        ...Object.entries(h.field_evidence).map(([key, e]) => [key, `${confidence[e.confidence]} / ${e.source_ids.join(", ")}: ${e.note}`] as [string, string]),
        ...Object.entries(h.compute).map(([key, c]) => [`Compute ${key.toUpperCase()}`, `${confidence[c.evidence.confidence]} / ${c.evidence.source_ids.join(", ")}: ${c.evidence.note}`] as [string, string]),
      ]} /></section>
    </>}
  </div>;
}
