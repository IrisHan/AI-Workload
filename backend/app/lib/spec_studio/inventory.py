"""One operator inventory drives parameter accounting and the grouped UI."""

from __future__ import annotations

from types import SimpleNamespace

from app.engines.operator_methods import (
    dsa_projections,
    feed_forward,
    mla_projections,
    msa_projections,
    qk_norm,
    qkv_projections,
)

from .ir import FfnKind


def module_row(
    name,
    category,
    shape,
    params,
    *,
    active=None,
    matrix=0,
    dtype="bf16",
    formula="",
    source="config.json",
    count=1,
    notes=None,
    shared_with=None,
    input_shape="",
    output_shape="",
):
    return dict(
        id=name,
        label=name,
        category=category,
        shape=shape,
        params=params,
        active_params=params if active is None else active,
        matrix_params=matrix,
        dtype=dtype,
        formula=formula,
        source=source,
        count=count,
        notes=notes,
        shared_with=shared_with,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def layer_inventory(match, ffn_kind, dtype, extras=()):
    bb, mla, moe = match["backbone"], match["mla"], match["moe"]
    d, heads, kv, c = bb.hidden_size, bb.num_heads, bb.num_kv_heads, bb.head_dim
    extras = tuple(extras or ())
    rows = []
    cfg = match["cfg"]
    norm_dtype = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
        cfg.get("torch_dtype") or cfg.get("dtype"), "bf16"
    )

    def linear(name, inf, outf, category="attention", bias=False, count=1, active_count=None):
        active_count = count if active_count is None else active_count
        matrix = inf * outf
        rows.append(
            module_row(
                name,
                category,
                [outf, inf],
                (matrix + (outf if bias else 0)) * count,
                active=(matrix + (outf if bias else 0)) * active_count,
                matrix=matrix * active_count,
                dtype=dtype,
                count=count,
                formula=f"({outf} * {inf} + {outf if bias else 0} bias) * {count}",
                source=f"config dimensions; {match['identity'].model_type} operator rules",
                input_shape=f"[B, S, {inf}]",
                output_shape=f"[B, S, {outf}]",
                notes=f"active multiplicity={active_count}; bias={bias}",
            )
        )

    def norm(name, dim, category="norm", *, layernorm=False):
        if layernorm:
            rows.append(module_row(
                name, category, [dim], 2 * dim,
                dtype=norm_dtype, formula=f"LayerNorm scale+bias: 2 * {dim}",
            ))
            return
        rows.append(
            module_row(
                name,
                category,
                [dim],
                dim,
                dtype=norm_dtype,
                formula=f"RMSNorm scale: {dim}",
                source="RMSNorm affine scale",
                input_shape=f"[..., {dim}]",
                output_shape=f"[..., {dim}]",
            )
        )

    writer = SimpleNamespace(linear=linear, norm=norm)
    norm("attn_norm", d)
    if mla:
        mla_projections(writer, hidden=d, heads=heads, q_rank=mla.q_lora_rank,
                        kv_rank=mla.kv_lora_rank, nope_dim=mla.qk_nope_head_dim,
                        rope_dim=mla.qk_rope_head_dim, value_dim=mla.v_head_dim)
        linear("o_proj", heads * mla.v_head_dim, d)
        if match["cfg"].get("mla_use_output_gate"):
            linear("g_proj", d, heads * mla.v_head_dim)
        if "DSA_INDEXER" in extras:
            ic, ih = cfg["index_head_dim"], cfg["index_n_heads"]
            dsa_projections(writer, hidden=d, q_rank=mla.q_lora_rank,
                            index_heads=ih, index_dim=ic)
        if "INDEX_KPOOL" in extras:
            ic = cfg["index_head_dim"]
            kpool = int(cfg.get("index_kpool") or 1)
            rows.append(module_row(
                "indexer.index_kpool_compress_ape", "attention", [kpool, ic], kpool * ic,
                dtype=dtype, formula=f"{kpool} * {ic}",
                source="index_kpool compressor ape; llama.cpp tensor shape",
            ))
            linear("indexer.index_kpool_compress_gate", d, ic)
    else:
        qkv_projections(writer, hidden=d, heads=heads, kv_heads=kv, head_dim=c,
                        qkv_bias=bb.qkv_bias, output_bias=bb.o_bias)
        if "QK_NORM_FUSED" in extras:
            qk_norm(writer, query_dim=heads * c, key_dim=kv * c)
        elif match["qk_norm"] or "QK_NORM" in extras:
            qk_norm(writer, query_dim=c, key_dim=c)
        if "MSA" in extras:
            from .validation import integer

            sparse = cfg.get("sparse_attention_config") if isinstance(cfg.get("sparse_attention_config"), dict) else {}
            ih = cfg.get("index_n_heads") or sparse.get("sparse_num_index_heads")
            ic = cfg.get("index_head_dim") or sparse.get("sparse_index_dim")
            integer(ih, "sparse_num_index_heads", high=4096)
            integer(ic, "sparse_index_dim", high=65536)
            msa_projections(writer, hidden=d, index_heads=ih, index_dim=ic)
        if "ATTN_SINKS" in extras:
            rows.append(module_row(
                "attn_sink", "attention", [heads], heads, dtype="fp32",
                formula=str(heads), notes="Per-head attention sink; not a GEMM.",
                source="GQA attention sinks",
            ))
    norm("ffn_norm", d)

    def ffn(prefix, width, count=1, active_count=None):
        feed_forward(writer, d, width, prefix=prefix, bias=bb.mlp_bias,
                     count=count, active_count=active_count)

    if ffn_kind == FfnKind.MOE and moe:
        router_bias = "ROUTER_BIAS" in extras
        linear("router", d, moe.n_routed, "ffn", bias=router_bias)
        if moe.topk_method in {"noaux_tc", "noaux-tc"} or "ROUTER_CORRECTION" in extras:
            rows.append(
                module_row(
                    "router_correction",
                    "ffn",
                    [moe.n_routed],
                    moe.n_routed,
                    dtype="fp32",
                    formula=str(moe.n_routed),
                    notes="Checkpoint routing correction vector; included in structural count, not GEMM.",
                    source="router correction buffer",
                )
            )
        ffn("routed_expert.", moe.expert_dim, moe.n_routed, moe.top_k)
        if "EXPERT_BIAS" in extras:
            rows.append(module_row(
                "routed_expert.gate_up_bias", "ffn", [moe.n_routed, 2 * moe.expert_dim],
                moe.n_routed * 2 * moe.expert_dim,
                active=moe.top_k * 2 * moe.expert_dim,
                dtype=dtype, count=moe.n_routed,
                formula=f"{moe.n_routed} * 2 * {moe.expert_dim}",
                notes="Fused gate/up expert bias",
            ))
            rows.append(module_row(
                "routed_expert.down_bias", "ffn", [moe.n_routed, d],
                moe.n_routed * d, active=moe.top_k * d,
                dtype=dtype, count=moe.n_routed,
                formula=f"{moe.n_routed} * {d}",
                notes="Expert down-projection bias",
            ))
        if moe.n_shared:
            ffn("shared_expert.", moe.expert_dim * moe.n_shared)
    else:
        ffn("", bb.intermediate_size)
    return rows


def inventory_metrics(rows):
    return dict(
        params_total=sum(row["params"] for row in rows),
        params_active=sum(row["active_params"] for row in rows),
        matrix_params=sum(row["matrix_params"] for row in rows),
    )


def moe_layout(rows):
    """Derive MoE display metadata from ledger rows. Does not change accounting."""
    by_id = {row["id"]: row for row in rows}
    routed = by_id.get("routed_expert.gate_proj")
    if not routed:
        return None
    n_routed = routed.get("count") or 1
    expert_dim = routed["shape"][0]
    hidden = routed["shape"][1] if len(routed["shape"]) > 1 else None
    per = routed["params"] / n_routed if n_routed else 0
    top_k = int(round(routed["active_params"] / per)) if per else None
    shared = by_id.get("shared_expert.gate_proj")
    n_shared = 0
    shared_width = 0
    if shared:
        shared_width = shared["shape"][0]
        counted = shared.get("count") or 1
        if counted > 1:
            n_shared = counted
            shared_width = expert_dim * n_shared
        elif expert_dim:
            n_shared = shared_width // expert_dim
        else:
            n_shared = 1
    latent = by_id.get("routed_expert_down_proj")
    return dict(
        n_routed=n_routed,
        top_k=top_k,
        n_shared=n_shared,
        expert_dim=expert_dim,
        hidden_size=hidden,
        shared_width=shared_width,
        has_shared_expert_gate="shared_expert_gate" in by_id,
        has_router="router" in by_id,
        latent_dim=latent["shape"][0] if latent and latent.get("shape") else None,
    )


def graph_cluster(name):
    if name in {"routed_expert_down_proj", "routed_expert_up_proj", "routed_expert_norm.weight"}:
        return "latent"
    if name.startswith("routed_expert"):
        return "routed"
    if name.startswith("shared_expert") or name == "shared_expert_gate":
        return "shared"
    if name in {"router", "router_correction", "router_scoring", "top_k", "expert_selection", "hash_lookup"}:
        return "router"
    if name in {"expert_sum", "weighted_expert_sum"}:
        return "merge"
    if name.startswith(("sam.patch", "siglip.patch", "vision.patch")) or name in {
        "sam.position", "siglip.position", "vision.position",
    }:
        return "patch"
    if name.startswith(("sam.block", "siglip.block", "vision.block.")):
        return "encoder"
    if name.startswith(("sam.", "siglip.", "vision.merger", "vision.final", "connector.", "hybrid.")):
        return "projector"
    if name.endswith("_conv1d") or name in {"depthwise_conv"}:
        return "score"
    if name.startswith(("q_", "in_proj", "rope_q")) or name in {"q_heads", "q_norm", "q_norm.weight", "q_a_norm", "q_a_norm.weight"}:
        return "q_path"
    if name.startswith(("k_", "v_", "kv_", "compressor", "rope_k")) or name in {
        "k_heads", "v_heads", "k_split", "v_split", "kv_split", "k_norm", "k_norm.weight",
        "kv_norm.weight", "kv_a_norm", "kv_a_norm.weight", "compressed_rope",
    }:
        return "kv_path"
    if name.startswith("indexer") or name in {
        "qk_matmul", "scale_mask", "softmax", "av_matmul", "causal_window", "mask_softmax_sink",
        "kda_recurrence", "gated_delta_recurrence", "decay_gate", "beta_gate", "l2_normalize",
        "depthwise_conv", "A_log", "dt_bias", "attn_sink",
    } or name.endswith("_conv1d"):
        return "score"
    if name.startswith(("o_proj", "o_a", "o_b", "g_proj", "g_a_proj", "g_b_proj", "out_proj")) or name in {
        "concat", "output_gate", "inverse_rope",
    }:
        return "out"
    return None


def graph_label(node_id, row=None, layout=None):
    count = (row or {}).get("count") or 1
    label = node_id
    if count > 1:
        label = f"{node_id} ×{count}"
    elif layout and node_id.startswith("shared_expert.") and layout.get("n_shared"):
        n_shared = layout["n_shared"]
        label = f"{node_id} · {n_shared}×W" if n_shared > 1 else f"{node_id} ×1"
    if node_id.endswith("gate_proj") and "expert" in node_id:
        label += " · gate"
    elif node_id.endswith("up_proj") and "expert" in node_id:
        label += " · up"
    elif node_id == "shared_expert_gate":
        label = "shared_expert_gate d→1"
    elif node_id == "router" and layout:
        label = f"router d→{layout['n_routed']}"
    elif node_id == "top_k" and layout and layout.get("top_k") is not None:
        label = f"top_k {layout['top_k']}/{layout['n_routed']}"
    return label


def annotate_graph(nodes, rows):
    by_id = {row["id"]: row for row in rows}
    layout = moe_layout(rows)
    for node in nodes:
        name = node["id"]
        cluster = graph_cluster(name)
        if cluster:
            node["cluster"] = cluster
        row = by_id.get(node.get("module_id") or name)
        label = graph_label(name, row, layout)
        if label != node.get("label", name):
            node["label"] = label
    return nodes


def operator_graph(match, rows, attn_kind, ffn_kind):
    modules = {row["id"]: row for row in rows}
    nodes, edges = [], []

    def node(name, category="attention", detail=None, label=None):
        nodes.append(
            dict(
                id=name,
                label=label or name,
                category=category,
                module_id=name if name in modules else None,
                detail=detail,
            )
        )
        return name

    def edge(source, target, label=None):
        edges.append(dict(source=source, target=target, label=label))

    def chain(*ids):
        for source, target in zip(ids, ids[1:]):
            edge(source, target)

    node("input", "other", "[B, S, hidden_size]")
    node("attn_norm", "norm")
    chain("input", "attn_norm")
    mla = match["mla"]
    if mla:
        if mla.q_lora_rank:
            for name in ("q_a_proj", "q_a_norm", "q_b_proj"):
                node(name)
            chain("attn_norm", "q_a_proj", "q_a_norm", "q_b_proj")
            q = "q_b_proj"
        else:
            q = node("q_proj")
            edge("attn_norm", q)
        for name in ("kv_a_proj", "kv_split", "kv_a_norm", "kv_b_proj", "k_split", "v_split"):
            node(name)
        chain("attn_norm", "kv_a_proj", "kv_split", "kv_a_norm", "kv_b_proj")
        edge("kv_b_proj", "k_split")
        edge("kv_b_proj", "v_split")
        k, v = "k_split", "v_split"
    else:
        q, k, v = "q_proj", "k_proj", "v_proj"
        for name in (q, k, v):
            node(name)
            edge("attn_norm", name)
        if "q_norm" in modules:
            node("q_norm")
            node("k_norm")
            edge(q, "q_norm")
            edge(k, "k_norm")
            q, k = "q_norm", "k_norm"
    for name in ("q_heads", "k_heads", "v_heads"):
        node(name, detail="reshape into attention heads")
    edge(q, "q_heads")
    edge(k, "k_heads")
    edge(v, "v_heads")
    node("rope_q", detail=str(match["positional"].value))
    node("rope_k", detail=str(match["positional"].value))
    edge("q_heads", "rope_q")
    edge("k_heads", "rope_k")
    if mla:
        edge("kv_split", "rope_k", "shared RoPE component")
    node("qk_matmul", detail="Q @ K^T")
    edge("rope_q", "qk_matmul")
    edge("rope_k", "qk_matmul")
    node("scale_mask", detail=f"{attn_kind.value}; causal mask; window={match['backbone'].sliding_window}")
    if "indexer.q_proj" in modules:
        if mla:
            for name in ("indexer.q_proj", "indexer.k_proj", "indexer.k_norm", "indexer.weights_proj",
                         "indexer.score", "indexer.top_k"):
                node(name, detail="Sparse token selection; full index-key cache retained")
            chain("q_a_norm", "indexer.q_proj", "indexer.score")
            chain("attn_norm", "indexer.k_proj", "indexer.k_norm", "indexer.score")
            chain("attn_norm", "indexer.weights_proj", "indexer.score")
            chain("indexer.score", "indexer.top_k", "scale_mask")
        else:
            for name in ("indexer.q_proj", "indexer.k_proj", "indexer.q_norm", "indexer.k_norm",
                         "indexer.score", "indexer.top_k"):
                if name in modules or name in {"indexer.score", "indexer.top_k"}:
                    node(name, detail="Lightning indexer; main GQA QKV unchanged")
            chain("attn_norm", "indexer.q_proj", "indexer.q_norm", "indexer.score")
            chain("attn_norm", "indexer.k_proj", "indexer.k_norm", "indexer.score")
            chain("indexer.score", "indexer.top_k", "scale_mask")
    if "attn_sink" in modules:
        node("attn_sink", detail="per-head attention sink")
        edge("attn_sink", "scale_mask")
    node("softmax", detail="softmax(scores)")
    node("av_matmul", detail="softmax(scores) @ V")
    node("concat", detail="merge heads")
    node("o_proj")
    if "g_proj" in modules:
        node("g_proj")
        node("output_gate", detail="MLA output gate; not a second residual")
    node("attn_add", "residual", "residual addition")
    chain("qk_matmul", "scale_mask", "softmax", "av_matmul", "concat", "o_proj")
    if "g_proj" in modules:
        edge("g_proj", "output_gate")
        edge("o_proj", "output_gate")
        edge("output_gate", "attn_add")
        edge("attn_norm", "g_proj")
    else:
        edge("o_proj", "attn_add")
    edge("v_heads", "av_matmul")
    edge("input", "attn_add", "residual")
    node("ffn_norm", "norm")
    edge("attn_add", "ffn_norm")

    def ffn(prefix, start):
        for suffix in ("gate_proj", "up_proj", "silu", "multiply", "down_proj"):
            node(prefix + suffix, "ffn")
        edge(start, prefix + "gate_proj")
        edge(start, prefix + "up_proj")
        chain(prefix + "gate_proj", prefix + "silu", prefix + "multiply", prefix + "down_proj")
        edge(prefix + "up_proj", prefix + "multiply")
        return prefix + "down_proj"

    moe = match["moe"]
    if ffn_kind == FfnKind.MOE and moe:
        node("router", "ffn")
        node("top_k", "ffn", f"{moe.top_k} of {moe.n_routed} experts; not observed routing")
        chain("ffn_norm", "router")
        routing = "router"
        if "router_correction" in modules:
            node("router_correction", "ffn")
            edge("router", "router_correction")
            routing = "router_correction"
        edge(routing, "top_k")
        end = ffn("routed_expert.", "ffn_norm")
        edge("top_k", "routed_expert.gate_proj", f"select {moe.top_k}/{moe.n_routed}")
        node("expert_sum", "ffn", "weighted sum of selected experts")
        edge(end, "expert_sum")
        edge("top_k", "expert_sum", "routing weights")
        if moe.n_shared:
            shared = ffn("shared_expert.", "ffn_norm")
            edge(shared, "expert_sum")
        end = "expert_sum"
    else:
        end = ffn("", "ffn_norm")
    node("ffn_add", "residual", "residual addition")
    node("output", "other", "[B, S, hidden_size]")
    edge(end, "ffn_add")
    edge("attn_add", "ffn_add", "residual")
    edge("ffn_add", "output")
    annotate_graph(nodes, rows)
    return nodes, edges
