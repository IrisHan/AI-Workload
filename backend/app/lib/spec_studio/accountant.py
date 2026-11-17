from __future__ import annotations

from typing import Any

from .inventory import layer_inventory
from .ir import (
    ArchIR,
    AttentionKind,
    CalcRecord,
    FfnKind,
    KvLayout,
    LayerSpec,
    MLAConfig,
    MoEConfig,
    ServingMetrics,
    TensorModule,
    Totals,
)
from .ops import (
    bytes_for,
    dense_ffn_params,
    embedding_params,
    gib,
    gptq_linear_bytes,
    linear_params,
    rmsnorm_params,
)


def account(match: dict[str, Any], *, include_mtp: bool = False, kv_dtype: str = "bf16") -> ArchIR:
    bb = match["backbone"]
    mla: MLAConfig | None = match["mla"]
    moe: MoEConfig | None = match["moe"]
    cfg: dict[str, Any] = match["cfg"]
    calcs: list[CalcRecord] = []
    modules: list[TensorModule] = []

    dtype_weight, storage_dtype, dtype_expert, qcfg = _resolve_dtype(cfg)
    warnings: list[str] = list(match["warnings"])
    infos: list[str] = []

    d, v, layer_count = bb.hidden_size, bb.vocab_size, bb.num_layers
    calcs.extend(_cfg_calcs(bb))

    p_embed = embedding_params(v, d)
    calcs.append(
        CalcRecord(
            calc_id="param.embed",
            formula="P_embed = V × d",
            substitutions={"V": v, "d": d},
            steps=[f"{v} × {d} = {p_embed}"],
            result=p_embed,
            unit="params",
            source="config vocab_size × hidden_size",
            confidence="calc-from-official",
        )
    )
    p_head = 0 if bb.tied_embeddings else embedding_params(v, d)
    calcs.append(
        CalcRecord(
            calc_id="param.lm_head",
            formula="P_head = 0 if tied else V × d",
            substitutions={"tied": bb.tied_embeddings, "V": v, "d": d},
            steps=[f"{p_head}"],
            result=p_head,
            unit="params",
            source="tie_word_embeddings",
            confidence="calc-from-official",
        )
    )
    modules.append(TensorModule(name="embed", shape=[v, d], params=p_embed, dtype=dtype_weight))
    if p_head:
        modules.append(TensorModule(name="lm_head", shape=[v, d], params=p_head, dtype=dtype_weight))

    inventory = {}
    for spec in match["layers"]:
        key = (spec.ffn_kind, spec.extras)
        if key not in inventory:
            inventory[key] = layer_inventory(match, spec.ffn_kind, dtype_weight, extras=spec.extras)
    match["module_inventory"] = inventory
    first_rows = next(iter(inventory.values()))
    p_attn = sum(row["params"] for row in first_rows if row["category"] == "attention")
    calcs.append(
        CalcRecord(
            calc_id="param.attn.layer",
            formula=" + ".join(row["formula"] for row in first_rows if row["category"] == "attention"),
            substitutions=_attn_subs(bb, mla),
            steps=[f"P_attn/layer = {p_attn}"],
            result=p_attn,
            unit="params",
            source="Linear shapes from matched attention kind",
            confidence="calc-from-official",
        )
    )

    p_expert = 0
    if moe:
        p_expert = dense_ffn_params(d, moe.expert_dim, activation=bb.activation, bias=bb.mlp_bias)
        calcs.append(
            CalcRecord(
                calc_id="param.moe.one_expert",
                formula="P_exp = 3 × d × d_moe  (SwiGLU) or 2× for ungated",
                substitutions={"d": d, "d_moe": moe.expert_dim, "act": bb.activation},
                steps=[f"{p_expert}"],
                result=p_expert,
                unit="params",
                source="MoE expert SwiGLU",
                confidence="calc-from-official",
            )
        )

    p_final = d

    layers: list[LayerSpec] = []
    attn_all = ffn_total = ffn_active = other = 0
    matrix_active = 0
    for spec in match["layers"]:
        layer = spec.model_copy()
        rows = inventory[(spec.ffn_kind, spec.extras)]
        layer.attn_params = sum(row["params"] for row in rows if row["category"] == "attention")
        layer.norm_params = sum(row["params"] for row in rows if row["category"] == "norm")
        layer.ffn_params_total = sum(row["params"] for row in rows if row["category"] == "ffn")
        layer.ffn_params_active = sum(row["active_params"] for row in rows if row["category"] == "ffn")
        layer.params_total = layer.attn_params + layer.ffn_params_total + layer.norm_params
        layer.params_active = layer.attn_params + layer.ffn_params_active + layer.norm_params
        layers.append(layer)
        attn_all += layer.attn_params * layer.count
        ffn_total += layer.ffn_params_total * layer.count
        ffn_active += layer.ffn_params_active * layer.count
        other += layer.norm_params * layer.count
        matrix_active += sum(row["matrix_params"] for row in rows) * layer.count

    other += p_final
    p_mtp = 0
    active_mtp = 0
    if include_mtp and bb.mtp_layers:
        # MTP block matches the last main layer (learned MoE if present) plus 2×d×d fuse.
        last = max(layers, key=lambda layer: max(stop for _, stop in layer.ranges)) if layers else None
        fuse = 2 * d * d + 2 * d
        if last:
            p_mtp = bb.mtp_layers * (last.params_total + fuse)
            active_mtp = bb.mtp_layers * (last.params_active + fuse)
            matrix_active += bb.mtp_layers * (
                sum(row["matrix_params"] for row in inventory[(last.ffn_kind, last.extras)]) + 2 * d * d
            )
        else:
            p_mtp = bb.mtp_layers * fuse
        calcs.append(
            CalcRecord(
                calc_id="param.mtp",
                formula="P_mtp = n_mtp × (P_last_block + 2·d·d + 2·d)",
                substitutions={"n_mtp": bb.mtp_layers, "d": d},
                steps=[f"{p_mtp}"],
                result=p_mtp,
                unit="params",
                source="DeepSeek-style MTPBlock + eh_proj",
                confidence="calc-from-official",
                notes="excluded from card total unless include_mtp=true",
            )
        )
    elif bb.mtp_layers:
        infos.append(
            "MTP present; excluded from card totals (toggle include_mtp). "
            "Checkpoint on-disk bytes still include MTP modules."
        )

    total = p_embed + p_head + attn_all + ffn_total + other + (p_mtp if include_mtp else 0)
    active_body = attn_all + ffn_active + other + active_mtp
    # Card-style active includes embed + lm_head params (tied: embed once).
    active = active_body + p_embed + p_head

    calcs.append(
        CalcRecord(
            calc_id="param.total",
            formula="N_tot = P_embed + P_head + Σ_layer (P_attn + P_ffn + P_norm) + P_final [+ P_mtp]",
            substitutions={
                "P_embed": p_embed,
                "P_head": p_head,
                "P_attn": attn_all,
                "P_ffn": ffn_total,
                "P_other": other,
                "P_mtp": p_mtp if include_mtp else 0,
            },
            steps=[
                f"embed+head = {p_embed + p_head}",
                f"attn = {attn_all}",
                f"ffn = {ffn_total}",
                f"norm+final = {other}",
                f"total = {total}",
            ],
            result=total,
            unit="params",
            source="operator inventory from config",
            confidence="calc-from-official",
        )
    )
    calcs.append(
        CalcRecord(
            calc_id="param.active",
            formula="N_act = P_embed + P_head + Σ (P_attn + P_ffn_active + P_norm) + P_final",
            substitutions={"P_ffn_active": ffn_active, "body": active_body},
            steps=[f"active_body = {active_body}", f"active = {active}"],
            result=active,
            unit="params",
            source="routed experts: only top-k + shared counted",
            confidence="calc-from-official",
        )
    )

    p_fp4 = 0
    p_fp8 = 0
    p_rest = total
    if dtype_expert == "fp4" and moe:
        n_moe_layers = sum(x.count for x in layers if x.ffn_kind == FfnKind.MOE)
        p_fp4 = n_moe_layers * moe.n_routed * p_expert
        p_rest -= p_fp4
    if storage_dtype == "fp8":
        p_fp8 = p_rest
        p_rest = 0
    estimated_bytes = bytes_for(p_fp4, "fp4") + bytes_for(p_fp8, "fp8") + bytes_for(p_rest, dtype_weight)
    gptq_bytes = (
        _gptq_model_bytes(match["attention_kind"], bb, qcfg)
        if str(qcfg.get("quant_method", "")).lower() == "gptq"
        else None
    )
    if gptq_bytes is not None:
        estimated_bytes = gptq_bytes
    logical_bpe = _logical_bytes_per_param(storage_dtype, dtype_weight)
    logical_bytes = int(total * logical_bpe)
    # decode GEMM: body + lm_head matmul (tied still does the GEMM)
    p_head_flop = embedding_params(v, d)
    flop = 2 * (matrix_active + p_head_flop)

    if kv_dtype not in {"bf16", "fp16", "fp32", "fp8"}:
        raise ValueError("unsupported kv_dtype")
    kv = _kv_layout(match["attention_kind"], bb, mla, layer_count, kv_dtype)
    if bb.sliding_window:
        kv.notes += f"; per retained token, sliding layers retain at most {bb.sliding_window} tokens"
    serving = ServingMetrics(
        nparams_total=total,
        nparams_active=active,
        nparams_active_body=active_body,
        weight_bytes=int(estimated_bytes),
        weight_bytes_gib=round(gib(estimated_bytes), 4),
        flop_gemm_decode=float(flop),
        kv_layout=kv,
        include_mtp=include_mtp,
        weight_bytes_logical=logical_bytes,
        weight_bytes_source="gptq-closed-form" if gptq_bytes is not None else "estimated",
    )
    calcs.append(
        CalcRecord(
            calc_id="flop.decode_gemm",
            formula="F_gemm = 2 * (active matrix weights + logits matrix); excludes bias/norm",
            substitutions={"matrix_weights": matrix_active, "P_head": p_head_flop},
            steps=[f"{flop}"],
            result=flop,
            unit="FLOP/token",
            source="matrix-multiply convention 2KN",
            confidence="calc-from-official",
        )
    )
    calcs.append(
        CalcRecord(
            calc_id="mem.weight_bytes",
            formula="bytes = Σ params_d × sizeof(d)",
            substitutions={"dtype_weight": dtype_weight, "storage": storage_dtype, "bytes": estimated_bytes},
            steps=[f"{estimated_bytes} B = {serving.weight_bytes_gib} GiB"],
            result=serving.weight_bytes_gib,
            unit="GiB",
            source="quantization_config / torch_dtype / packed layout",
            confidence="assumed" if not qcfg else "calc-from-official",
        )
    )

    for kind, rows in inventory.items():
        for row in rows:
            modules.append(
                TensorModule(
                    name=row["id"] if len(inventory) == 1 else f"{kind[0].value}.{row['id']}",
                    shape=row["shape"],
                    params=row["params"],
                    dtype=row["dtype"],
                    per_layer=True,
                    notes=row["notes"],
                )
            )
    modules.append(TensorModule(name="final_norm", shape=[d], params=d, dtype=dtype_weight))

    totals = Totals(
        total=total,
        active=active,
        active_body=active_body,
        embed=p_embed,
        lm_head=p_head,
        attn=attn_all,
        ffn_total=ffn_total,
        ffn_active=ffn_active,
        other=other,
        mtp=p_mtp if include_mtp else 0,
        bytes_by_dtype={
            key: value
            for key, value in {
                "fp4": bytes_for(p_fp4, "fp4"),
                storage_dtype: bytes_for(p_fp8, "fp8") + bytes_for(p_rest, dtype_weight),
            }.items()
            if value
        },
    )

    return ArchIR(
        identity=match["identity"],
        backbone=bb,
        capabilities=match["capabilities"],
        attention_kind=match["attention_kind"],
        ffn_kind=match["ffn_kind"],
        norm_kind=match["norm_kind"],
        norm_mode=match["norm_mode"],
        positional=match["positional"],
        residual=match["residual"],
        mla=mla,
        moe=moe,
        layers=layers,
        modules=modules,
        calcs=calcs,
        totals=totals,
        serving=serving,
        warnings=warnings,
        infos=infos,
        unread_fields=match["unread_fields"],
        dtype_weight=dtype_weight,
        dtype_expert=dtype_expert,
    )


def _cfg_calcs(bb: Any) -> list[CalcRecord]:
    rows = [
        ("cfg.vocab", "V = vocab_size", bb.vocab_size, "token"),
        ("cfg.layers", "L = num_hidden_layers", bb.num_layers, "layer"),
        ("cfg.hidden", "d = hidden_size", bb.hidden_size, ""),
        ("cfg.n_head", "n_h = num_attention_heads", bb.num_heads, ""),
        ("cfg.head_dim", "c = head_dim", bb.head_dim, ""),
    ]
    out: list[CalcRecord] = []
    for calc_id, formula, result, unit in rows:
        out.append(
            CalcRecord(
                calc_id=calc_id,
                formula=formula,
                substitutions={},
                steps=["direct from config"],
                result=result,
                unit=unit,
                source="config.json",
                confidence="official",
            )
        )
    return out


def _resolve_dtype(cfg: dict[str, Any]) -> tuple[str, str, str | None, dict[str, Any]]:
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        qcfg = {}
    expert = cfg.get("expert_dtype")
    method = str(qcfg.get("quant_method") or "").lower()
    fmt = str(qcfg.get("fmt") or "").lower()

    if cfg.get("fp16"):
        compute = "fp16"
    elif cfg.get("bf16"):
        compute = "bf16"
    else:
        td = str(cfg.get("torch_dtype") or "bfloat16").lower()
        compute = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(td, td)

    if method == "fp8" or fmt in {"e4m3", "fp8"} or method == "mxfp8":
        storage = "fp8"
    elif method == "mxfp4" or "mxfp4" in fmt:
        storage = "mxfp4"
    elif method in {"gptq", "awq"}:
        bits = int(qcfg.get("bits") or 4)
        storage = f"{method}-{bits}"
    elif method in {"bitsandbytes", "bnb", "bnb-4bit"}:
        storage = "bnb"
    else:
        storage = compute
    dtype_out = "fp8" if storage == "fp8" else compute
    return dtype_out, storage, str(expert).lower() if expert else None, qcfg


def _logical_bytes_per_param(storage: str, compute: str) -> float:
    if storage.startswith("gptq-") or storage.startswith("awq-") or storage == "bnb":
        bits = 4
        if "-" in storage:
            try:
                bits = int(storage.split("-")[-1])
            except ValueError:
                bits = 4
        return bits / 8.0
    if storage == "fp8":
        return 1.0
    if storage == "mxfp4":
        return 0.5
    return {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp4": 0.5}.get(compute, 2.0)


def _gptq_model_bytes(attn_kind: AttentionKind, bb: Any, qcfg: dict[str, Any]) -> int | None:
    bits = int(qcfg.get("bits") or 4)
    gs = int(qcfg.get("group_size") or 128)
    d, d_ff, layers = bb.hidden_size, bb.intermediate_size, bb.num_layers
    q_dim = bb.num_heads * bb.head_dim
    kv_dim = bb.num_kv_heads * bb.head_dim
    linears: list[tuple[int, int]] = []
    if attn_kind == AttentionKind.MHA:
        linears.append((d, 3 * q_dim))
        linears.append((q_dim, d))
    elif attn_kind in {AttentionKind.GQA, AttentionKind.MQA}:
        linears.extend([(d, q_dim), (d, kv_dim), (d, kv_dim), (q_dim, d)])
    else:
        return None
    linears.extend([(d, d_ff), (d, d_ff), (d_ff, d)])
    packed = 0
    for inf, outf in linears:
        packed += layers * gptq_linear_bytes(inf, outf, bits=bits, group_size=gs)
    unquant = bytes_for(embedding_params(bb.vocab_size, d) * (1 if bb.tied_embeddings else 2), "fp16")
    unquant += bytes_for(2 * d * layers + d, "fp16")
    return int(packed + unquant)


def _attn_params(bb: Any, mla: MLAConfig | None, qk_norm: bool) -> int:
    h, heads, kv, c = bb.hidden_size, bb.num_heads, bb.num_kv_heads, bb.head_dim
    bias = bb.attention_bias
    if mla is not None:
        qk_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
        if mla.q_lora_rank:
            q = (
                linear_params(h, mla.q_lora_rank)
                + rmsnorm_params(mla.q_lora_rank)
                + linear_params(mla.q_lora_rank, heads * qk_dim)
            )
        else:
            q = linear_params(h, heads * qk_dim)
        kv_p = (
            linear_params(h, mla.kv_lora_rank + mla.qk_rope_head_dim)
            + rmsnorm_params(mla.kv_lora_rank)
            + linear_params(mla.kv_lora_rank, heads * (mla.qk_nope_head_dim + mla.v_head_dim))
        )
        o = linear_params(heads * mla.v_head_dim, h)
        return q + kv_p + o
    q_dim = heads * c
    kv_dim = kv * c
    p = (
        linear_params(h, q_dim, bias=bias)
        + 2 * linear_params(h, kv_dim, bias=bias)
        + linear_params(q_dim, h, bias=bias)
    )
    if qk_norm:
        p += 2 * c
    return p


def _attn_formula(mla: MLAConfig | None) -> str:
    if mla is None:
        return "P_attn = d·(n_h c) + 2·d·(n_kv c) + (n_h c)·d  [+ QK-Norm]"
    return (
        "P_attn = d·d_c + d_c + d_c·n_h·(c_nope+c_rope) "
        "+ d·(r_kv+c_rope) + r_kv + r_kv·n_h·(c_nope+c_v) "
        "+ n_h·c_v·d"
    )


def _attn_subs(bb: Any, mla: MLAConfig | None) -> dict[str, Any]:
    if mla is None:
        return {
            "d": bb.hidden_size,
            "n_h": bb.num_heads,
            "n_kv": bb.num_kv_heads,
            "c": bb.head_dim,
        }
    return {
        "d": bb.hidden_size,
        "n_h": bb.num_heads,
        "d_c": mla.q_lora_rank,
        "r_kv": mla.kv_lora_rank,
        "c_nope": mla.qk_nope_head_dim,
        "c_rope": mla.qk_rope_head_dim,
        "c_v": mla.v_head_dim,
    }


def _attn_modules(bb: Any, mla: MLAConfig | None, dtype: str) -> list[TensorModule]:
    h, heads, kv, c = bb.hidden_size, bb.num_heads, bb.num_kv_heads, bb.head_dim
    if mla is None:
        return [
            TensorModule(
                name="q_proj", shape=[h, heads * c], params=h * heads * c, dtype=dtype, per_layer=True
            ),
            TensorModule(name="k_proj", shape=[h, kv * c], params=h * kv * c, dtype=dtype, per_layer=True),
            TensorModule(name="v_proj", shape=[h, kv * c], params=h * kv * c, dtype=dtype, per_layer=True),
            TensorModule(
                name="o_proj", shape=[heads * c, h], params=heads * c * h, dtype=dtype, per_layer=True
            ),
        ]
    qk = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    mods = []
    if mla.q_lora_rank:
        mods.append(
            TensorModule(
                name="wq_a",
                shape=[h, mla.q_lora_rank],
                params=h * mla.q_lora_rank,
                dtype=dtype,
                per_layer=True,
            )
        )
        mods.append(
            TensorModule(
                name="wq_b",
                shape=[mla.q_lora_rank, heads * qk],
                params=mla.q_lora_rank * heads * qk,
                dtype=dtype,
                per_layer=True,
            )
        )
    mods.append(
        TensorModule(
            name="wkv_a",
            shape=[h, mla.kv_lora_rank + mla.qk_rope_head_dim],
            params=h * (mla.kv_lora_rank + mla.qk_rope_head_dim),
            dtype=dtype,
            per_layer=True,
        )
    )
    mods.append(
        TensorModule(
            name="wkv_b",
            shape=[mla.kv_lora_rank, heads * (mla.qk_nope_head_dim + mla.v_head_dim)],
            params=mla.kv_lora_rank * heads * (mla.qk_nope_head_dim + mla.v_head_dim),
            dtype=dtype,
            per_layer=True,
        )
    )
    mods.append(
        TensorModule(
            name="wo",
            shape=[heads * mla.v_head_dim, h],
            params=heads * mla.v_head_dim * h,
            dtype=dtype,
            per_layer=True,
        )
    )
    return mods


def _kv_layout(kind: AttentionKind, bb: Any, mla: MLAConfig | None, layers: int, dtype: str) -> KvLayout:
    bpe = {"fp8": 1, "fp4": 1, "bf16": 2, "fp16": 2, "fp32": 4}.get(dtype, 2)
    if mla is not None:
        per_layer = mla.kv_lora_rank + mla.qk_rope_head_dim
        formula = "L × (kv_lora_rank + qk_rope_head_dim) × sizeof"
        notes = "MLA latent KV; not full K/V heads"
    else:
        per_layer = 2 * bb.num_kv_heads * bb.head_dim
        formula = "L × 2 × n_kv × head_dim × sizeof"
        notes = f"{kind.value} full K and V"
    return KvLayout(
        kind=kind.value,
        formula=formula,
        bytes_per_token=int(layers * per_layer * bpe),
        dtype_assumed=dtype,
        notes=notes,
    )


def cross_check_index(
    arch: ArchIR,
    index: dict[str, Any] | None,
    *,
    hub_used_storage: int | None = None,
    hub_safetensors: dict[str, int] | None = None,
) -> None:
    """Pick on-disk weight_bytes. Never rewrite nparams."""
    index_size = None
    if index:
        meta = index.get("metadata") or {}
        raw = meta.get("total_size")
        if raw:
            index_size = int(raw)
    arch.serving.index_total_size = index_size
    arch.serving.hub_used_storage = int(hub_used_storage) if hub_used_storage else None

    hist_bytes, packed_hist = _histogram_bytes(hub_safetensors)
    estimated = arch.serving.weight_bytes
    chosen = estimated
    source = arch.serving.weight_bytes_source or "estimated"
    is_fp8 = arch.dtype_weight == "fp8" or any(
        c.id == "quant" and "fp8" in (c.label or "") for c in arch.capabilities
    )
    is_gptq = any(c.id == "quant" and "gptq" in (c.label or "") for c in arch.capabilities)

    if hist_bytes is not None and not packed_hist:
        chosen = hist_bytes
        source = "hub-histogram"
        if hub_used_storage and abs(hist_bytes - hub_used_storage) / max(hub_used_storage, 1) > 0.05:
            arch.infos.append(
                f"Hub safetensors histogram bytes={hist_bytes} vs usedStorage={hub_used_storage}; using histogram"
            )
    elif is_gptq and index_size and abs(estimated - index_size) / max(index_size, 1) < 0.005:
        chosen = estimated
        source = "gptq-closed-form"
    elif packed_hist and hub_used_storage:
        chosen = int(hub_used_storage)
        source = "usedStorage"
    elif packed_hist and index_size:
        chosen = index_size
        source = "index"
    elif (
        is_fp8
        and index_size
        and _looks_bf16_inflated(index_size, hub_safetensors, arch.totals.total, estimated)
    ):
        chosen = hist_bytes or hub_used_storage or estimated
        source = "hub-histogram" if hist_bytes else ("usedStorage" if hub_used_storage else "estimated")
        arch.infos.append(
            f"safetensors index total_size={index_size} looks BF16-inflated "
            f"(~2× param count); ignored. Using {source}={chosen}. nparams unchanged."
        )
    elif hub_used_storage and index_size:
        ratio = index_size / max(hub_used_storage, 1)
        if 1.8 < ratio < 2.2 and is_fp8:
            chosen = int(hub_used_storage)
            source = "usedStorage"
            arch.infos.append(
                f"index total_size={index_size} ≈ 2 × usedStorage={hub_used_storage}; "
                "treating index as BF16-inflated. nparams unchanged."
            )
        elif abs(index_size - hub_used_storage) / max(hub_used_storage, 1) < 0.05:
            chosen = index_size
            source = "index"
        else:
            chosen = int(hub_used_storage)
            source = "usedStorage"
    elif hub_used_storage:
        chosen = int(hub_used_storage)
        source = "usedStorage"
    elif index_size:
        rel = abs(estimated - index_size) / max(index_size, 1)
        if rel > 0.03:
            if is_fp8 and _looks_bf16_inflated(index_size, hub_safetensors, arch.totals.total, estimated):
                arch.infos.append(
                    f"safetensors index total_size={index_size} looks BF16-inflated; "
                    f"keeping estimated weight_bytes={estimated}. nparams unchanged."
                )
            else:
                arch.warnings.append(
                    f"safetensors index total_size={index_size} vs computed weight_bytes={estimated} "
                    f"({rel:.1%} off); not rewriting nparams"
                )
                for c in arch.calcs:
                    if c.calc_id == "mem.weight_bytes":
                        c.confidence = "assumed"
                        c.notes = (c.notes or "") + " index mismatch >3%"
        else:
            chosen = index_size
            source = "index"

    chosen = int(chosen)
    if chosen < 0:
        raise ValueError("negative weight metadata")
    if source == "usedStorage":
        source = "repository-storage-estimate"
        arch.warnings.append("usedStorage includes repository files, not verified tensor bytes.")
    if source == "estimated":
        arch.warnings.append(
            "Physical weight bytes estimated; quantization buffers/scales may be unavailable."
        )
    arch.serving.weight_bytes = chosen
    arch.serving.weight_bytes_gib = round(gib(chosen), 4)
    arch.serving.weight_bytes_source = source
    if (
        index_size
        and is_fp8
        and _looks_bf16_inflated(index_size, hub_safetensors, arch.totals.total, estimated)
        and not any("BF16-inflated" in msg for msg in arch.infos)
    ):
        arch.infos.append(
            f"safetensors index total_size={index_size} looks BF16-inflated; "
            f"ignored. Using {source}={chosen}. nparams unchanged."
        )

    trusted = hist_bytes
    if trusted:
        rel_est = abs(estimated - trusted) / max(trusted, 1)
        if rel_est > 0.20 and source in {"hub-histogram", "usedStorage", "index"}:
            arch.errors.append(
                f"closed-form disk estimate {estimated} is {rel_est:.1%} off trusted {trusted}; "
                f"serving.weight_bytes uses {source}={chosen}. nparams unchanged."
            )
        elif 0.03 < rel_est <= 0.20 and source in {"hub-histogram", "usedStorage"}:
            arch.infos.append(
                f"closed-form disk estimate {estimated} vs trusted {trusted} ({rel_est:.1%}); using {source}."
            )

    for c in arch.calcs:
        if c.calc_id == "mem.weight_bytes":
            c.result = arch.serving.weight_bytes_gib
            c.steps = [f"{chosen} B = {arch.serving.weight_bytes_gib} GiB ({source})"]
            c.substitutions = {
                **c.substitutions,
                "bytes": chosen,
                "source": source,
                "index_total_size": index_size,
                "hub_used_storage": hub_used_storage,
            }
            c.confidence = (
                "official"
                if source == "hub-histogram"
                else ("calc-from-official" if source == "gptq-closed-form" else "assumed")
            )


def _histogram_bytes(hist: dict[str, int] | None) -> tuple[int | None, bool]:
    if not hist:
        return None, False
    native = {
        "BF16": 2,
        "F16": 2,
        "FP16": 2,
        "F32": 4,
        "FP32": 4,
        "F64": 8,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "F8_E4M3FN": 1,
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "I64": 8,
    }
    packed = False
    total = 0
    for key, n in hist.items():
        if type(n) is not int or n < 0:
            raise ValueError("invalid safetensors histogram count")
        k = str(key).upper()
        if k in {"I32", "INT32", "U32"}:
            packed = True
            continue
        bpe = native.get(k)
        if bpe is None:
            packed = True
            continue
        total += int(n) * bpe
    if packed and total == 0:
        return None, True
    return int(total) if total else None, packed


def _looks_bf16_inflated(
    index_size: int,
    hist: dict[str, int] | None,
    nparams: int,
    estimated: int,
) -> bool:
    if hist:
        hist_total = sum(int(v) for v in hist.values())
        if hist_total and abs(index_size / hist_total - 2.0) < 0.05:
            return True
    if nparams and abs(index_size / nparams - 2.0) < 0.08:
        return True
    if estimated and estimated > 0 and abs(index_size / estimated - 2.0) < 0.12:
        return True
    return False
