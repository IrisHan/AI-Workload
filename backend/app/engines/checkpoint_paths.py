"""Adapter-owned mapping from ledger names to checkpoint module paths."""


def module_paths(row, config, *, role=None):
    name = row["id"]
    text = config.get("text_config") or config
    family = str(text.get("model_type") or config.get("model_type") or "")
    if role == "vision":
        path = name.replace("vision.patch", "visual.patch_embed.proj", 1)
        path = path.replace("vision.block.", "visual.blocks.{block}.", 1)
        path = path.replace("vision.", "visual.", 1)
        for component in ("qkv", "proj", "q_norm", "k_norm"):
            path = path.replace("visual.blocks.{block}." + component,
                                "visual.blocks.{block}.attn." + component, 1)
        path = path.removesuffix(".weight")
        return [path, "model." + path, name]
    if name in {"embed", "embed_tokens"}:
        return ["model.embed_tokens"]
    if name == "lm_head":
        return ["lm_head"]
    if name == "final_norm":
        return ["model.norm"]
    root = "model.layers.{layer}."
    norms = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
    stem = name.removesuffix(".weight")
    if stem in norms:
        return [root + norms[stem]]
    if name.startswith("hc_"):
        return [root + name.replace(".", "_")]
    if name.startswith(("self_attention_res", "mlp_res")):
        return [root + name]
    if row["category"] == "attention":
        aliases = {"q_a_norm.weight": "q_a_layernorm", "kv_a_norm.weight": "kv_a_layernorm",
                   "q_a_norm": "q_a_layernorm", "kv_a_norm": "kv_a_layernorm",
                   "kv_a_proj": "kv_a_proj_with_mqa", "indexer.q_proj": "indexer.wq_b",
                   "indexer.k_proj": "indexer.wk", "depthwise_conv": "conv1d"}
        paths = [root + "self_attn." + name]
        if name in aliases:
            paths.append(root + "self_attn." + aliases[name])
        if family.startswith("qwen3") and text.get("linear_num_key_heads"):
            paths.append(root + "linear_attn." + aliases.get(name, name))
        return paths
    if row["category"] == "ffn":
        module = name.replace("routed_expert.", "experts.{expert}.")
        module = module.replace("shared_expert.", "shared_experts.")
        if name == "router":
            module = "router" if family == "gpt_oss" else "gate"
        elif name == "router_correction":
            module = "gate.e_score_correction_bias"
        paths = [root + "mlp." + module]
        if family.startswith("kimi") or family == "mixtral":
            paths.append(root + "block_sparse_moe." + module)
        return paths
    return [name]


def attach_checkpoint_paths(analysis, config):
    for template in analysis["templates"]:
        for row in template["modules"]:
            row["checkpoint_paths"] = module_paths(row, config)
    for row in analysis["shared_modules"]:
        row["checkpoint_paths"] = module_paths(row, config)
    for branch in analysis.get("branches") or []:
        for row in branch["modules"]:
            row["checkpoint_paths"] = module_paths(row, config, role=branch.get("role"))
    return analysis
