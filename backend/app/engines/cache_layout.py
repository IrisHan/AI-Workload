"""Scenario cache residency shared by Structure Cost and static Memory."""
from math import prod


def retained_length(length, window=None):
    return min(length, window) if window and window > 0 else length


def attach_cache_layout(analysis, config):
    text = config.get("text_config") or config
    for template in analysis["templates"]:
        kind = template["attn_kind"]
        bb, mla = template["backbone"], template.get("mla")
        if kind in {"KDA", "GATED_DELTA"}:
            if kind == "KDA":
                spec = text["linear_attn_config"]
                heads, key, value = spec["num_heads"], spec["head_dim"], spec["head_dim"]
                conv = [3 * heads * key, spec["short_conv_kernel_size"] - 1]
            else:
                heads = text["linear_num_value_heads"]
                key, value = text["linear_key_head_dim"], text["linear_value_head_dim"]
                conv = [2 * text["linear_num_key_heads"] * key + heads * value,
                        text["linear_conv_kernel_dim"]]
            template["recurrent"] = dict(
                heads=heads, key_dim=key, value_dim=value, state_dtype="fp32",
                convolution_shape=conv,
            )
            template["cache"] = dict(kind="recurrent", **template["recurrent"])
        elif config.get("model_type") == "deepseek_v4":
            template["cache"] = dict(
                kind="compressed", dim=config["head_dim"], window=config["sliding_window"],
                ratio=template.get("compress_ratio") or 0, index_dim=config["index_head_dim"],
            )
        else:
            dim = (mla["kv_lora_rank"] + mla["qk_rope_head_dim"]) if mla else 2 * bb["num_kv_heads"] * bb["head_dim"]
            # Keep the established grouped DSA index-key scope. KDA kpool/MSA
            # retention policies require a separate source-backed extension.
            index_dim = 0
            if not text.get("linear_attn_config") and any(
                row["id"].startswith("indexer.") for row in template["modules"]
            ):
                index_dim = int(config.get("index_head_dim") or 0)
            template["cache"] = dict(
                kind="kv", dim=dim + index_dim,
                window=bb.get("sliding_window") if kind == "SWA" else None,
            )
    return analysis


def template_cache(template, batch, context, elem):
    spec = template.get("cache")
    if spec is None:
        if template["attn_kind"] in {"KDA", "GATED_DELTA", "CSA", "HCA"}:
            raise ValueError("Snapshot lacks cache/recurrent layout; create a new structure snapshot")
        bb, mla = template["backbone"], template.get("mla")
        spec = dict(
            kind="kv",
            dim=mla["kv_lora_rank"] + mla["qk_rope_head_dim"] if mla else 2 * bb["num_kv_heads"] * bb["head_dim"],
            window=bb.get("sliding_window") if template["attn_kind"] == "SWA" else None,
        )
    if spec["kind"] == "recurrent":
        state = batch * spec["heads"] * spec["key_dim"] * spec["value_dim"] * 4
        conv = batch * prod(spec["convolution_shape"]) * elem
        return dict(recurrent_state_bytes=state, convolution_bytes=conv, total_bytes=state + conv)
    if spec["kind"] == "compressed":
        ratio, dim = spec["ratio"], spec["dim"]
        window = batch * retained_length(context, spec["window"]) * dim * elem
        compressed = batch * (context // ratio) * dim * elem if ratio else 0
        index = batch * (context // ratio) * spec["index_dim"] * elem if ratio == 4 else 0
        coff = 2 if ratio == 4 else 1
        state = batch * 2 * coff * ratio * coff * dim * 4 if ratio else 0
        if ratio == 4:
            state += batch * 2 * coff * ratio * coff * spec["index_dim"] * 4
        return dict(window_bytes=window, compressed_bytes=compressed, index_bytes=index,
                    compression_state_bytes=state, total_bytes=window + compressed + index + state)
    size = batch * retained_length(context, spec.get("window")) * spec["dim"] * elem
    return dict(kv_bytes=size, total_bytes=size)


def cache_components(analysis, batch, context, elem):
    if analysis["serving"]["include_mtp"]:
        raise ValueError("MTP cache execution policy is not specified for this architecture")
    rows = []
    templates = {t["id"]: t for t in analysis["templates"]}
    per_layer = any((t.get("cache") or {}).get("kind") in {"recurrent", "compressed"}
                    for t in templates.values())
    for group in analysis["groups"]:
        template = templates[group["template_id"]]
        item = template_cache(template, batch, context, elem)
        if per_layer:
            for start, stop in group["ranges"]:
                rows.extend(dict(layer=i, **item) for i in range(start, stop))
        else:
            rows.append(dict(group_id=group["id"], count=group["count"],
                             **{k: v * group["count"] for k, v in item.items()}))
    return sorted(rows, key=lambda r: r.get("layer", -1))
