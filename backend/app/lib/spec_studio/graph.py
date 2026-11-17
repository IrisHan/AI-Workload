from __future__ import annotations

from .ir import ArchIR, FfnKind, GraphEdge, GraphIR, GraphNode


def build_graph(arch: ArchIR) -> GraphIR:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    nodes.append(
        GraphNode(
            id="embed",
            kind="embed",
            label="Token embedding",
            params=arch.totals.embed,
            capability_ids=[
                arch.positional.value.lower() if hasattr(arch.positional, "value") else "position"
            ],
            detail=f"V={arch.backbone.vocab_size} × d={arch.backbone.hidden_size}",
        )
    )
    prev = "embed"
    for layer in arch.layers:
        lid = f"layer-{layer.index}"
        nodes.append(
            GraphNode(
                id=lid,
                kind="layer",
                label=f"L{layer.index}  {layer.attn_kind.value} + {layer.ffn_kind.value}",
                params=layer.params_total,
                capability_ids=[layer.attn_kind.value.lower(), layer.ffn_kind.value.lower()],
                layer_index=layer.index,
                extras={
                    "attn_params": layer.attn_params,
                    "ffn_params_total": layer.ffn_params_total,
                    "ffn_params_active": layer.ffn_params_active,
                    "active": layer.params_active,
                },
            )
        )
        edges.append(GraphEdge(source=prev, target=lid))
        prev = lid

        nodes.extend(
            [
                GraphNode(
                    id=f"{lid}-n1",
                    kind="norm",
                    label=arch.norm_kind.value,
                    params=arch.backbone.hidden_size,
                    capability_ids=[arch.norm_kind.value.lower()],
                    parent=lid,
                    layer_index=layer.index,
                ),
                GraphNode(
                    id=f"{lid}-attn",
                    kind="attn",
                    label=layer.attn_kind.value,
                    params=layer.attn_params,
                    capability_ids=[layer.attn_kind.value.lower()],
                    parent=lid,
                    layer_index=layer.index,
                    detail=_attn_detail(arch),
                ),
                GraphNode(
                    id=f"{lid}-n2",
                    kind="norm",
                    label=arch.norm_kind.value,
                    params=arch.backbone.hidden_size,
                    capability_ids=[arch.norm_kind.value.lower()],
                    parent=lid,
                    layer_index=layer.index,
                ),
                GraphNode(
                    id=f"{lid}-ffn",
                    kind="ffn",
                    label="MoE" if layer.ffn_kind == FfnKind.MOE else "Dense FFN",
                    params=layer.ffn_params_total,
                    capability_ids=["moe" if layer.ffn_kind == FfnKind.MOE else "dense-ffn"],
                    parent=lid,
                    layer_index=layer.index,
                    extras={"active": layer.ffn_params_active},
                    detail=_ffn_detail(arch, layer.ffn_kind),
                ),
            ]
        )

    if arch.backbone.mtp_layers:
        nodes.append(
            GraphNode(
                id="mtp",
                kind="mtp",
                label=f"MTP ×{arch.backbone.mtp_layers}",
                params=arch.totals.mtp,
                capability_ids=["mtp"],
                detail="optional; excluded from card total by default",
            )
        )
        edges.append(GraphEdge(source=prev, target="mtp"))

    nodes.append(
        GraphNode(
            id="final-norm",
            kind="norm",
            label=arch.norm_kind.value,
            params=arch.backbone.hidden_size,
            capability_ids=["rmsnorm"],
        )
    )
    edges.append(GraphEdge(source=prev, target="final-norm"))
    prev = "final-norm"
    nodes.append(
        GraphNode(
            id="head",
            kind="head",
            label="LM head" + (" (tied)" if arch.backbone.tied_embeddings else ""),
            params=arch.totals.lm_head,
            capability_ids=[],
            detail=f"V={arch.backbone.vocab_size}; unique weights only; logits GEMM still executes",
        )
    )
    edges.append(GraphEdge(source=prev, target="head"))
    return GraphIR(nodes=nodes, edges=edges)


def _attn_detail(arch: ArchIR) -> str:
    if arch.mla:
        m = arch.mla
        return (
            f"MLA  q_lora={m.q_lora_rank} kv_lora={m.kv_lora_rank} "
            f"nope={m.qk_nope_head_dim} rope={m.qk_rope_head_dim}"
        )
    b = arch.backbone
    return f"{arch.attention_kind.value}  heads={b.num_heads} kv={b.num_kv_heads} dim={b.head_dim}"


def _ffn_detail(arch: ArchIR, kind: FfnKind) -> str:
    if kind == FfnKind.MOE and arch.moe:
        m = arch.moe
        return f"MoE routed={m.n_routed} topk={m.top_k} shared={m.n_shared} d_moe={m.expert_dim}"
    return f"Dense {arch.backbone.activation} d_ff={arch.backbone.intermediate_size}"
