import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowRight, Cpu, Download, Layers3, MemoryStick, Network, RefreshCw, ShieldCheck, Waypoints, Zap } from "lucide-react";
import type { ArchitectureBlock, ArchitectureDocument, ArchitectureFact, HardwareRecord } from "../../api/hardware";
import { ErrorState, Loading } from "../../components/common";
import { downloadJson } from "../../utils/format";
import { useArchitecture } from "./queries";
import "./architecture.css";

const icons = [Cpu, Waypoints, Layers3, Zap, MemoryStick, Network, Layers3, ShieldCheck];
const rowOrder = [[1, 2], [4, 3], [6, 5], [7, 8]];
type Section = ArchitectureDocument["sections"][number];
const tables = (blocks: ArchitectureBlock[]) => blocks.filter(b => b.kind === "table");
const textBlocks = (blocks: ArchitectureBlock[]) => blocks.filter(b => b.kind === "text");

function Blocks({ blocks }: { blocks: ArchitectureBlock[] }) {
  return <div className="arch-content">{blocks.map((block, i) => {
    if (block.kind === "text") return <p key={i}>{block.text}</p>;
    const columns = block.headers.map((header, index) => ({ header, index }))
      .filter(c => c.header !== "当前目录状态");
    return <div className="table-scroll" key={i}><table><thead><tr>{columns.map(c => <th key={c.index}>{c.header}</th>)}</tr></thead>
      <tbody>{block.rows.map((row, j) => <tr key={j}>{columns.map(c => <td key={c.index}>{row[c.index]}</td>)}</tr>)}</tbody></table></div>;
  })}</div>;
}

function TechnicalFooter({ blocks }: { blocks: ArchitectureBlock[] }) {
  const paragraphs = [...new Set(textBlocks(blocks).map(b => b.text.trim()).filter(Boolean))];
  if (!paragraphs.length) return null;
  return <div className="arch-technical-footer"><p>{paragraphs.join(" ")}</p></div>;
}

function Fact({ fact: f }: { fact: ArchitectureFact }) {
  return <div className="arch-fact" data-fact={f.key} title={`${f.note}\n来源：${f.source_ids.join(", ")}`}>
    <span className="arch-fact-label">{f.label}</span>
    <strong>{f.value}</strong>
  </div>;
}

function MemoryHierarchy({ facts }: { facts: ArchitectureFact[] }) {
  const local = facts.filter(f => ["register", "tmem", "l1_pool"].includes(f.key));
  const shared = facts.find(f => f.key === "shared_max");
  return <figure className="arch-memory" aria-label="Register TMEM L1 Shared L2 device memory hierarchy">
    <figcaption>{local.length ? "SM 片上存储" : "设备内存（缺少片上微架构资料）"}</figcaption>
    <div className="arch-memory-local">{local.map(f => <Fact key={f.key} fact={f} />)}</div>
    {shared && <div className="arch-carveout"><MemoryStick size={15} />Shared 上限 <strong>{shared.value}</strong> / SM，包含在 L1/Shared 总池内</div>}
    <div className="arch-memory-arrow"><ArrowDown size={18} /></div>
    <div className="arch-memory-global">{facts.filter(f => ["l2", "device_memory"].includes(f.key)).map((f, i) =>
      <div key={f.key}>{i > 0 && <ArrowRight size={18} />}<Fact fact={f} /></div>)}</div>
  </figure>;
}

export function Architecture({ hardware: h, packaging }: { hardware: HardwareRecord; packaging: ReactNode }) {
  const query = useArchitecture(h.id);
  if (query.isPending) return <Loading />;
  if (query.isError) return <><ErrorState error={query.error} /><button onClick={() => { void query.refetch(); }}><RefreshCw size={16} />Retry architecture</button></>;
  const d = query.data;
  const renderSection = (number: number, displayNumber: number) => {
    const s = d.sections.find(section => section.number === number);
    if (!s) return null;
    const Icon = icons[number - 1];
    const title = number === 1 ? "基础概述" : s.title;
    return <section className={`arch-section arch-section-${number}`} id={`architecture-${number}`} key={number} aria-label={title}>
      <h3><span className="arch-section-number">{displayNumber}</span><Icon size={17} />{title}</h3>
      <div className="arch-section-body">
        {number === 2 ? <ChipOrganization document={d} section={s} /> : <>
          {number === 3 && <div className="arch-facts arch-sm-facts">{d.architecture.specs.filter(f => f.key !== "l2").map(f => <Fact key={f.key} fact={f} />)}</div>}
          {number === 5 && <MemoryHierarchy facts={d.memory_hierarchy} />}
          <Blocks blocks={[3, 5].includes(number) ? tables(s.blocks) : s.blocks} />
          {[3, 5].includes(number) && <TechnicalFooter blocks={s.blocks} />}
        </>}
        {!s.blocks.length && !d.architecture.specs.length && <p className="muted">此架构暂无对应微架构参数。</p>}
      </div>
    </section>;
  };
  return <div className="architecture-atlas">
    <div className="arch-title">
      <h2>{d.architecture.name} 架构深度解析</h2>
      <button className="icon-button" title="Export architecture JSON" onClick={() => downloadJson(d, `${h.id}-architecture`)}><Download size={18} /></button>
    </div>
    <div className="arch-grid">
      {rowOrder.map((row, index) => <div className="arch-row" key={index} data-row={index + 1}>
        {row.map((number, column) => renderSection(number, index * 2 + column + 1))}
      </div>)}
    </div>
    <section className="arch-related"><h2>{h.name}</h2><div>{d.related_skus.map(s => <Link key={s.id} to={`/hardware/${s.id}?tab=architecture`} aria-current={s.id === h.id ? "page" : undefined}>{s.name}<ArrowRight size={14} /></Link>)}</div>
    </section>
    <div className="hw-education arch-legacy"><div>{h.education.map(e => <section className="section" key={e.title}><h2>{e.title}</h2><p>{e.text}</p></section>)}</div>{packaging}</div>
    <section className="arch-comparison"><h2>型号总览</h2><Blocks blocks={d.comparison} /></section>
  </div>;
}

function ChipOrganization({ document: d, section: s }: { document: ArchitectureDocument; section: Section }) {
  const referenceTables = tables(s.reference_blocks);
  return <>
    <div className="arch-chip-layout">
      <div className="arch-chip-resources">
        <div className="arch-facts">{d.headline_metrics.map(f => <Fact key={f.key} fact={f} />)}</div>
        <figure className="arch-organization" aria-label="GPU GPC TPC SM hierarchy">
          {["GPU", "GPC", "TPC", "SM"].map((name, i) => <div key={name}>{i > 0 && <ArrowRight size={14} />}<span>{name}</span></div>)}
        </figure>
      </div>
      <div className="arch-chip-table"><Blocks blocks={referenceTables.length ? referenceTables : tables(s.blocks)} /></div>
    </div>
    <TechnicalFooter blocks={[...s.blocks, ...textBlocks(s.reference_blocks)]} />
  </>;
}
