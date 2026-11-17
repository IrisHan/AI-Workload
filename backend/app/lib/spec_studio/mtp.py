"""DeepSeek V3 MTP branch, using the documented shared weight identities."""

from copy import deepcopy

from .inventory import inventory_metrics, module_row, operator_graph

SOURCE = (
    "https://github.com/deepseek-ai/DeepSeek-V3/blob/"
    "4c2fdb8f55e049553b9f4f1a3241f86d739c8cf8/README_WEIGHTS.md"
)


def build_mtp(match, arch, rows, include_mtp):
    bb = arch.backbone
    d, v = bb.hidden_size, bb.vocab_size
    last = max(arch.layers, key=lambda layer: max(stop for _, stop in layer.ranges))
    modules = deepcopy(rows)
    for row in modules:
        row["id"] = "block." + row["id"]
        row["label"] = row["id"]
    extra = [
        module_row(
            "shared_embed",
            "embedding",
            [v, d],
            0,
            shared_with="embed",
            formula="0 unique; same tensor as main embedding",
            source=SOURCE,
        ),
        module_row("enorm", "norm", [d], d, formula=f"{d} scale", source=SOURCE),
        module_row("hnorm", "norm", [d], d, formula=f"{d} scale", source=SOURCE),
        module_row(
            "eh_proj",
            "fusion",
            [d, 2 * d],
            2 * d * d,
            matrix=2 * d * d,
            formula=f"{d} * {2 * d}",
            source=SOURCE,
            input_shape=f"[B,S,{2 * d}]",
            output_shape=f"[B,S,{d}]",
        ),
        module_row(
            "shared_head",
            "head",
            [v, d],
            0,
            shared_with="lm_head",
            formula="0 unique; main output head reference",
            source=SOURCE,
            notes="Logits compute still executes; not part of the unique branch body matrix sum.",
        ),
        module_row(
            "shared_norm",
            "norm",
            [d],
            0,
            shared_with="final_norm",
            formula="0 unique; shared output normalization reference",
            source=SOURCE,
        ),
    ]
    modules.extend(extra)
    nodes, edges = operator_graph(match, rows, last.attn_kind, last.ffn_kind)
    for node in nodes:
        node["id"] = "block." + node["id"]
        if node["module_id"]:
            node["module_id"] = "block." + node["module_id"]
    for edge in edges:
        edge["source"] = "block." + edge["source"]
        edge["target"] = "block." + edge["target"]
    for name, category in (
        ("hidden_input", "other"),
        ("token_input", "other"),
        ("shared_embed", "embedding"),
        ("enorm", "norm"),
        ("hnorm", "norm"),
        ("concat_inputs", "fusion"),
        ("eh_proj", "fusion"),
        ("shared_norm", "norm"),
        ("shared_head", "head"),
    ):
        nodes.append(
            dict(
                id=name,
                label=name,
                category=category,
                module_id=name if name in {row["id"] for row in extra} else None,
                detail=SOURCE,
            )
        )
    for source, target in (
        ("hidden_input", "hnorm"),
        ("token_input", "shared_embed"),
        ("shared_embed", "enorm"),
        ("hnorm", "concat_inputs"),
        ("enorm", "concat_inputs"),
        ("concat_inputs", "eh_proj"),
        ("eh_proj", "block.input"),
        ("block.output", "shared_norm"),
        ("shared_norm", "shared_head"),
    ):
        edges.append(dict(source=source, target=target))
    metrics = inventory_metrics(modules)
    return dict(
        id="mtp",
        label="MTP",
        signature="deepseek-v3-mtp-1",
        role="mtp",
        attn_kind=last.attn_kind.value,
        ffn_kind=last.ffn_kind.value,
        modules=modules,
        nodes=nodes,
        edges=edges,
        metrics=metrics,
        count=bb.mtp_layers,
        included=include_mtp,
        source=SOURCE,
        backbone=bb.model_dump(),
        mla=arch.mla.model_dump() if arch.mla else None,
    )
