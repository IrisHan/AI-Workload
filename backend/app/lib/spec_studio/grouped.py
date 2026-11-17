"""Compact semantic templates; no expanded graph or layer array in v2."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .accountant import account, cross_check_index
from .inventory import inventory_metrics, module_row, moe_layout, operator_graph
from .ir import FfnKind
from .matcher import match_config
from .mtp import build_mtp
from .validation import ensure_safe_numbers

VERSION = "2.0.0"


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def inspect_grouped(
    cfg,
    *,
    name="model",
    include_mtp=False,
    kv_dtype="bf16",
    safetensors_index=None,
    repo_id=None,
    revision=None,
    sha=None,
    hub_used_storage=None,
    hub_safetensors=None,
    config_sha256=None,
    fetched_at=None,
):
    if "layer_configs" in cfg:
        from .heterogeneous import inspect_heterogeneous

        return inspect_heterogeneous(
            cfg,
            name=name,
            include_mtp=include_mtp,
            kv_dtype=kv_dtype,
            repo_id=repo_id,
            revision=revision,
            sha=sha,
            config_sha256=config_sha256,
            fetched_at=fetched_at,
            safetensors_index=safetensors_index,
            hub_used_storage=hub_used_storage,
            hub_safetensors=hub_safetensors,
        )
    match = match_config(cfg, name=name, compact=True)
    match["identity"].repo_id = repo_id
    match["identity"].revision = revision
    match["identity"].sha = sha
    arch = account(match, include_mtp=include_mtp, kv_dtype=kv_dtype)
    cross_check_index(
        arch, safetensors_index, hub_used_storage=hub_used_storage, hub_safetensors=hub_safetensors
    )
    templates, groups, execution = [], [], []
    semantic_backbone = arch.backbone.model_dump(exclude={"num_layers", "mtp_layers", "vocab_size"})
    positional_config = {
        key: value
        for key, value in cfg.items()
        if key.startswith(("rope_", "rotary_", "use_dynamic_ntk", "use_logn_attn"))
    }
    for index, layer in enumerate(arch.layers):
        rows = match["module_inventory"][(layer.ffn_kind, layer.extras)]
        metrics = inventory_metrics(rows)
        signature = config_hash(
            dict(
                attn_kind=layer.attn_kind.value,
                ffn_kind=layer.ffn_kind.value,
                backbone=semantic_backbone,
                role=layer.role,
                formula_version=VERSION,
                modules=rows,
                position=positional_config,
                quantization=cfg.get("quantization_config"),
                norm=arch.norm_kind.value,
                residual=arch.residual.value,
                **({"extras": list(layer.extras)} if layer.extras else {}),
            )
        )
        template_id = f"t-{signature[:12]}"
        group_id = f"g{index + 1}"
        nodes, edges = operator_graph(match, rows, layer.attn_kind, layer.ffn_kind)
        item = dict(
                id=template_id,
                label=f"{layer.attn_kind.value} + {layer.ffn_kind.value}",
                signature=signature,
                role=layer.role,
                attn_kind=layer.attn_kind.value,
                ffn_kind=layer.ffn_kind.value,
                modules=rows,
                nodes=nodes,
                edges=edges,
                metrics=metrics,
                backbone=arch.backbone.model_dump(),
                mla=arch.mla.model_dump() if arch.mla else None,
        )
        if layer.ffn_kind == FfnKind.MOE:
            item["moe"] = moe_layout(rows)
        templates.append(item)
        groups.append(
            dict(
                id=group_id,
                template_id=template_id,
                count=layer.count,
                ranges=layer.ranges,
                metrics={key: value * layer.count for key, value in metrics.items()},
            )
        )
        execution.extend(
            dict(group_id=group_id, start=start, stop_exclusive=stop) for start, stop in layer.ranges
        )
    execution.sort(key=lambda row: row["start"])
    bb = arch.backbone
    branches = []
    shared = [
        module_row(
            "embed",
            "embedding",
            [bb.vocab_size, bb.hidden_size],
            arch.totals.embed,
            dtype=arch.dtype_weight,
            formula=f"{bb.vocab_size} * {bb.hidden_size}",
            input_shape="[B, S]",
            output_shape=f"[B, S, {bb.hidden_size}]",
        ),
        module_row(
            "final_norm",
            "norm",
            [bb.hidden_size],
            bb.hidden_size,
            dtype={"float32": "fp32", "float16": "fp16"}.get(cfg.get("torch_dtype"), "bf16"),
            formula=f"RMSNorm scale: {bb.hidden_size}",
            input_shape=f"[B, S, {bb.hidden_size}]",
            output_shape=f"[B, S, {bb.hidden_size}]",
        ),
        module_row(
            "lm_head",
            "head",
            [bb.vocab_size, bb.hidden_size],
            arch.totals.lm_head,
            matrix=bb.vocab_size * bb.hidden_size,
            dtype=arch.dtype_weight,
            formula="0 unique params; shared embedding"
            if bb.tied_embeddings
            else f"{bb.vocab_size} * {bb.hidden_size}",
            shared_with="embed" if bb.tied_embeddings else None,
            input_shape=f"[B, S, {bb.hidden_size}]",
            output_shape=f"[B, S, {bb.vocab_size}]",
        ),
    ]
    if bb.mtp_layers:
        last = max(arch.layers, key=lambda layer: max(stop for _, stop in layer.ranges))
        branch = build_mtp(match, arch, match["module_inventory"][(last.ffn_kind, last.extras)], include_mtp)
        branches.append(branch)
        active = (
            arch.totals.active
            - sum(g["metrics"]["params_active"] for g in groups)
            - sum(row["active_params"] for row in shared)
        )
        shared.append(
            module_row(
                "mtp",
                "mtp",
                [],
                arch.totals.mtp,
                active=active,
                matrix=(
                    arch.serving.flop_gemm_decode // 2
                    - sum(g["metrics"]["matrix_params"] for g in groups)
                    - sum(row["matrix_params"] for row in shared)
                ),
                formula=f"{bb.mtp_layers} * (last block + 2*d*d + 2*d)" if include_mtp else "excluded",
                notes="Optional prediction branch, not a sequential decoder layer. Includes shared embedding/head references.",
                count=bb.mtp_layers,
            )
        )
        if include_mtp and branch["metrics"]["params_total"] * bb.mtp_layers != arch.totals.mtp:
            raise ValueError("MTP branch inventory mismatch")
    if (
        sum(row["params"] for row in shared) + sum(g["metrics"]["params_total"] for g in groups)
        != arch.totals.total
    ):
        raise ValueError("internal parameter inventory mismatch")
    if (
        sum(row["active_params"] for row in shared) + sum(g["metrics"]["params_active"] for g in groups)
        != arch.totals.active
    ):
        raise ValueError("internal active inventory mismatch")
    diagnostics = []
    for severity, messages in (("warning", arch.warnings), ("info", arch.infos), ("error", arch.errors)):
        diagnostics.extend(
            dict(severity=severity, code=f"accounting.{severity}", message=text) for text in messages
        )
    diagnostics.extend(
        dict(severity="info", code="config.unread", message=f"{field.key}: {field.reason}")
        for field in arch.unread_fields
    )
    if arch.moe and arch.moe.topk_method in {"noaux_tc", "noaux-tc"}:
        diagnostics.append(
            dict(
                severity="info",
                code="router.checkpoint_buffer",
                message="Structural count includes routing correction vectors; not all checkpoint elements are trainable weights.",
            )
        )
    diagnostics.append(
        dict(
            severity="info",
            code="cost.static",
            message="GEMM is matrix-only single-token decode work, not full FLOPs or latency. KV dtype is an explicit assumption.",
        )
    )
    result = dict(
        schema_version="2",
        engine_version="2.1.0",
        formula_version=VERSION,
        status="supported",
        source=dict(
            repo_id=repo_id,
            requested_revision=revision,
            resolved_sha=sha,
            config_sha256=config_sha256 or config_hash(cfg),
            fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(),
            kind="huggingface-config" if sha else "local-config",
        ),
        identity=arch.identity.model_dump(),
        backbone=bb.model_dump(),
        capabilities=[cap.model_dump() for cap in arch.capabilities],
        totals=arch.totals.model_dump(),
        serving=arch.serving.model_dump(),
        templates=templates,
        groups=groups,
        execution=execution,
        shared_modules=shared,
        diagnostics=diagnostics,
        branches=branches,
    )
    ensure_safe_numbers(result)
    from app.engines.morphology import attach_morphology
    return attach_morphology(result, cfg)
