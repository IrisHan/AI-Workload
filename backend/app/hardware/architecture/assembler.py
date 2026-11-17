"""Join reusable educational architecture data to a frozen SKU without changing it."""
from app.hardware.architecture.repository import content_hash, load_bundle
from app.hardware.architecture.schemas import (
    ArchitectureDocument,
    ArchitectureFact,
    ArchitectureSection,
    ContentBlock,
)
from app.hardware.schemas import HardwareRecord

TITLES = [
    "整体架构概述", "芯片顶层组织", "SM 微架构", "Tensor Core 与精度",
    "内存子系统", "互联技术", "型号与系统对比", "关键特性与架构演进",
]


def text(value):
    return ContentBlock(kind="text", text=value)


def table(headers, rows):
    return ContentBlock(kind="table", headers=headers, rows=[[str(c) for c in row] for row in rows])


def assemble(hardware: HardwareRecord, related: list[HardwareRecord]) -> ArchitectureDocument:
    bundle = load_bundle()
    variant = next((v for v in bundle["variants"] if v["id"] == f"{hardware.family}-{hardware.form_factor}"), None)
    if variant is None:
        raise ValueError("Architecture reference is unavailable for this SKU")
    template = next(t for t in bundle["templates"] if t["id"] == variant["architecture_id"])
    reference = bundle["references"].get(variant["source_family"], {})
    shared = [ArchitectureFact.model_validate(f) for f in template["specs"]]
    memory_keys = {"register", "tmem", "l1_pool", "shared_max", "l2"}
    memory = [f for f in shared if f.key in memory_keys]
    memory.append(ArchitectureFact(
        key="device_memory", label=hardware.memory_type,
        value=f"{hardware.memory_bytes / 1e9:g} GB", scope="当前 SKU / 全 GPU",
        status="catalog", source_ids=[f"catalog:{s}" for s in hardware.field_evidence["memory_bytes"].source_ids],
        note=hardware.field_evidence["memory_bytes"].note,
    ))
    headlines = [ArchitectureFact.model_validate(f) for f in variant["facts"]]
    sections = []
    for index, title in enumerate(TITLES, 1):
        # A100/H100/L20 supply seven sections: their features belong in block eight.
        ref_index = str(index)
        if reference and "8" not in reference and index >= 7:
            ref_index = "7" if index == 8 else "0"
        ref = reference.get(ref_index, {})
        notes = list(variant["notes"].get(str(index), []))
        content = []
        if index == 1:
            content = [ContentBlock.model_validate(b) for b in ref.get("blocks", [])]
            if content:
                notes += ["本块为附件整理参考，完整Die、出货启用数量和其他封装不可混用；启用数量见第2块。"]
        if index == 2:
            content = [table(["资源", "数量", "范围"], [[f.label, f.value, f.scope] for f in headlines])]
            content += [text("GPU → GPC → TPC → SM → CUDA / Tensor 执行单元。GPC/TPC的启用布局依SKU裁剪而异，不能由CUDA总数倒推固定物理拓扑。")]
        if index == 3:
            content = [text(t) for t in template["explanations"]]
        if index == 4:
            content = [table(["精度", "Dense", "Sparse", "单位", "当前目录状态"], [
                [p.upper(), c.dense if c.dense is not None else "未提供",
                 c.sparse if c.sparse is not None else "未提供", c.unit, c.status]
                for p, c in hardware.compute.items()
            ])]
            if variant["architecture_id"].startswith("blackwell"):
                content += [text("NVFP4 E2M1：每16值配FP8 E4M3微缩放，并使用FP32张量级缩放。"
                                 "2SM MMA（CTA pair）共享操作数；tcgen05与Hopper WGMMA路径需匹配编译目标。"
                                 "附件约1.8倍存储节省与精度近似FP8是场景性描述，不保证所有模型精度。")]
            elif variant["architecture_id"] == "hopper":
                content += [text("FP8 E4M3/E5M2与Transformer Engine；WGMMA以4个warp协作，"
                                 "TMA负责张量搬运。TMA不是TMEM，精度选择仍依赖训练配方和内核。")]
        if index == 5:
            content = [text("同一SM内，Register / TMEM / L1-Shared承担不同任务，并非每次访问都逐级经过。"
                            "共享内存carveout占用统一池；L2是全GPU共享缓存，HBM/GDDR是当前SKU的片外显存。"),
                       text(f"当前SKU：{hardware.memory_bytes / 1e9:g} GB，"
                            f"{hardware.memory_bandwidth_bytes_s / 1e9:g} GB/s；"
                            "缓存MB沿用厂商表示，寄存器/片上KB按KiB解释，禁止把容量与传输速率混合。")]
        if index == 6:
            content = [table(["链路", "GB/s", "方向", "作用域"], [
                [link.name, link.bandwidth_gb_s if link.bandwidth_gb_s is not None else "未提供", link.direction, link.scope]
                for link in hardware.interconnect
            ]), text(f"节点参考：{hardware.node.name}；NIC/GPU单向"
                     f"{hardware.node.nic_gb_s_per_gpu if hardware.node.nic_gb_s_per_gpu is not None else '未指定'} GB/s。"
                     "C2C是Grace系统CPU-GPU互联，不能当成普通HGX主机PCIe；NVSwitch机架总数不等于板载数量。")]
        if index == 7:
            content = [table(["目录 SKU", "显存 GB", "带宽 GB/s", "BF16 TFLOPS", "封装"], [
                [h.name, h.memory_bytes / 1e9, h.memory_bandwidth_bytes_s / 1e9,
                 h.bf16_tflops or "未提供", h.form_factor]
                for h in related if h.architecture == hardware.architecture
            ])]
            content += [ContentBlock.model_validate(b) for b in ref.get("blocks", [])]
            notes += ["下方附件系统/对比内容保留原始口径，未核验条目不替代当前SKU冻结值。"]
        if index == 8:
            content = [ContentBlock.model_validate(b) for b in ref.get("blocks", [])]
            content += [text(s) for s in bundle["image_notes"].get(hardware.family, [])]
        if not reference and index in (1, 8):
            content += [text(f"{e.title}：{e.text}") for e in hardware.education]
        sections.append(ArchitectureSection(
            number=index, title=title, notes=notes, blocks=content,
            reference_blocks=[ContentBlock.model_validate(b) for b in ref.get("blocks", [])],
            reference_text=ref.get("text", ""),
        ))
    sources = list(bundle["sources"])
    sources += [
        {**s.model_dump(), "id": f"catalog:{s.id}"}
        for s in hardware.sources
    ]
    document = ArchitectureDocument(
        hardware_id=hardware.id, hardware_version=hardware.version, hardware_sha256=hardware.content_sha256,
        architecture=template, variant=variant, document_version=bundle["version"],
        content_sha256="", headline_metrics=headlines, memory_hierarchy=memory, sections=sections,
        sources=sources,
        related_skus=[dict(id=h.id, name=h.name) for h in related if h.architecture == hardware.architecture],
        comparison=bundle["comparison"],
        caveats=[
            "架构科普与可计算规格分离。用户附件中的推算、物理容量和跨型号对照按参考保留，未写入Workload输入。",
            "用户原文及五张附图包含冲突，校核说明优先。官方依据是固定CUDA版本与归档哈希，不是GPU实测。",
            "同架构模板复用不代表同SM启用数、算力、显存或互联；消费级SM12.x不可混入数据中心SM10.x。",
        ],
    )
    payload = document.model_dump()
    payload["content_sha256"] = content_hash(payload)
    return ArchitectureDocument.model_validate(payload)
