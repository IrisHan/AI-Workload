"""Compose language, recurrent attention, vision encoders and connectors."""
from copy import deepcopy

from app.engines.components import Ledger, assemble, template
from app.engines.layer_methods import (
    ATTN_GQA_GATE,
    ATTN_MLA,
    FFN_DENSE,
    FFN_LATENT_MOE,
    FFN_MOE,
    KIMI_SOURCE,
    declared_layer_types,
    fill_hybrid_layer,
    fill_linear_layer,
    hybrid_cache,
    kda_cache,
    layer_links,
    schedule_layers,
    top_k_experts,
)
from app.engines.vision_components import (
    QWEN_SOURCE,
    clip_rope_vit,
    deepseek_vision,
    glm_next_vision,
    intern_vit,
    is_moonvit,
    moonvit,
    qwen_vision,
)
from app.lib.spec_studio.grouped import inspect_grouped
from app.lib.spec_studio.inventory import module_row
from app.lib.spec_studio.validation import integer, validate_config


def unwrap_language(cfg):
    inner = cfg.get("text_config")
    vision = cfg.get("vision_config") if isinstance(cfg.get("vision_config"), dict) else None
    if not isinstance(inner, dict):
        return deepcopy(cfg), vision
    lang = deepcopy(inner)
    if not lang.get("model_type"):
        lang["model_type"] = cfg.get("model_type")
    if not lang.get("architectures"):
        lang["architectures"] = cfg.get("architectures", [])
    return lang, vision


def attach_optional_vision(result, cfg, lang, vision):
    if not vision:
        result["identity"]["model_type"] = cfg.get("model_type") or lang.get("model_type")
        result["identity"]["architectures"] = cfg.get("architectures", lang.get("architectures", []))
        return result
    text_hidden = lang["hidden_size"]
    if is_moonvit(vision):
        rows = moonvit(vision, text_hidden)
    elif vision.get("model_type") == "clip_vision_model" or vision.get("position_embedding_type") == "rope":
        rows = clip_rope_vit(vision, text_hidden, parent=cfg)
    elif (
        vision.get("model_type") in ("qianfan_ocr_vision", "intern_vit")
        or cfg.get("template") == "internvl2_5"
    ):
        rows = intern_vit(vision, text_hidden, parent=cfg)
    elif vision.get("model_type") == "glm5_next_vision" or (
        vision.get("projection_intermediate_size") and vision.get("out_hidden_size")
    ):
        rows = glm_next_vision(vision, text_hidden)
    elif isinstance(vision, dict) and vision.get("out_hidden_size") == text_hidden:
        rows = qwen_vision(vision)
    else:
        raise ValueError(
            "unsupported vision encoder; MoonViT, CLIP RoPE ViT, InternViT or Qwen-style merger required"
        )
    return attach_vision(result, rows, cfg)


def language_config(cfg):
    if cfg["model_type"] == "multi_modality":
        inner = cfg.get("language_config", {})
        if inner.get("model_type") != "llama":
            raise ValueError("unsupported vision language decoder")
        # LlamaConfig defaults used by the pinned MultiModalityConfig constructor.
        return {
            "hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32,
            "num_key_value_heads": inner.get("num_attention_heads", 32),
            "tie_word_embeddings": False, "hidden_act": "silu",
            "max_position_embeddings": 2048, "rope_theta": 10000,
            **inner, "architectures": ["LlamaForCausalLM"],
        }
    return deepcopy(cfg.get("text_config", cfg))


def attach_vision(result, rows, cfg):
    bb = result["backbone"]
    branch = template("Vision encoders + GELU connector", rows, bb,
                      attention="VISION", ffn="GELU", role="vision",
                      stages=[["image"], ["vision_encoders"], ["gelu_connector"], ["language_embeddings"]])
    for row in rows:
        stage = "gelu_connector" if row["category"] == "connector" else "vision_encoders"
        branch["edges"].append(dict(source=row["id"], target=stage, label="component"))
    branch.update(id="vision", count=1, included=True)
    result["branches"].append(branch)
    metric = branch["metrics"]
    result["shared_modules"].append(module_row(
        "vision", "vision", [], metric["params_total"], active=metric["params_active"],
        formula="sum(vision encoders + connector inventories)", source=rows[0]["source"],
        notes="Resident vision weights. Executed on image input, not on every decoded text token.",
    ))
    totals, serving = result["totals"], result["serving"]
    totals["vision"] = metric["params_total"]
    totals["total"] += metric["params_total"]
    # Active inventory includes an image pass, while decode GEMM remains text-only.
    totals["active"] += metric["params_active"]
    totals["active_body"] += metric["params_active"]
    size = sum(row["params"]*{"fp16": 2, "bf16": 2, "fp32": 4}.get(row["dtype"], 2) for row in rows)
    totals["bytes_by_dtype"]["vision_payload"] = size
    serving.update(nparams_total=totals["total"], nparams_active=totals["active"],
                   nparams_active_body=totals["active_body"])
    serving["weight_bytes"] += size
    serving["weight_bytes_logical"] += size
    serving["weight_bytes_gib"] = serving["weight_bytes"]/1024**3
    result["identity"]["model_type"] = cfg["model_type"]
    result["identity"]["architectures"] = cfg.get("architectures", [])
    result["formula_version"] = "components-1.0.0"
    result["capabilities"].append(dict(id="vision", label="Vision + connector", supported=True, evidence=[]))
    result["diagnostics"].append(dict(severity="info", code="modality.scope",
        message="Total includes resident vision and connector weights. Active count includes image path; decode GEMM and KV describe the language decoder only. Image FLOPs require image-grid inputs."))
    return result


def inspect_composite(cfg, **options):
    inner = language_config(cfg)
    if cfg["model_type"] == "multi_modality":
        if cfg.get("architectures") != ["MultiModalityCausalLM"]:
            raise ValueError("multimodal architecture mismatch")
        rows = deepseek_vision(cfg)
        if cfg["aligner_config"]["params"]["n_embed"] != inner["hidden_size"]:
            raise ValueError("vision connector output does not match language hidden size")
        result = inspect_grouped(inner, **options)
        result["diagnostics"].append(dict(severity="info", code="config.defaults",
            message="Missing language dimensions use pinned LlamaConfig defaults: hidden=4096, FFN=11008, heads=32, KV=heads. Original configuration remains unchanged."))
    else:
        result = inspect_hybrid_text(inner, **options)
        vision = cfg.get("vision_config")
        if not vision:
            result["identity"]["model_type"] = cfg["model_type"]
            result["identity"]["architectures"] = cfg.get("architectures", [])
            return result
        if not isinstance(vision, dict) or vision.get("out_hidden_size") != inner["hidden_size"]:
            raise ValueError("vision connector output does not match text hidden size")
        rows = qwen_vision(vision)
    return attach_vision(result, rows, cfg)


def inspect_hybrid_text(cfg, **options):
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "head_dim", "vocab_size", "max_position_embeddings",
                "linear_key_head_dim", "linear_value_head_dim", "linear_num_key_heads",
                "linear_num_value_heads", "linear_conv_kernel_dim"):
        integer(cfg.get(key), key, high=1048576)
    integer(cfg["num_hidden_layers"], "layers", high=4096)
    if cfg.get("hidden_act") != "silu" or cfg.get("attention_bias") or cfg.get("mlp_bias"):
        raise ValueError("unsupported hybrid projection/activation variant")
    types = declared_layer_types(cfg)
    schedule = schedule_layers(cfg)
    if any(ffn_id == FFN_DENSE for _, ffn_id, _ in schedule):
        integer(cfg.get("intermediate_size"), "intermediate_size", high=1048576)
    if any(ffn_id == FFN_MOE for _, ffn_id, _ in schedule):
        integer(cfg.get("moe_intermediate_size"), "moe_intermediate_size", high=1048576)
        integer(cfg.get("num_experts") or cfg.get("n_routed_experts"), "num_experts")
        integer(cfg.get("num_experts_per_tok"), "num_experts_per_tok")
        if cfg["num_experts_per_tok"] > (cfg.get("num_experts") or cfg.get("n_routed_experts")):
            raise ValueError("invalid top-k")
    d, c, h, kv = (cfg[k] for k in ("hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads"))
    if h % kv or cfg["linear_num_value_heads"] % cfg["linear_num_key_heads"]:
        raise ValueError("invalid attention head grouping")
    dense = any(ffn_id == FFN_DENSE for _, ffn_id, _ in schedule)
    bb = dict(hidden_size=d, num_layers=len(types), num_heads=h, num_kv_heads=kv, head_dim=c,
              intermediate_size=cfg["intermediate_size"] if dense else cfg.get("moe_intermediate_size"),
              vocab_size=cfg["vocab_size"],
              context_length=cfg["max_position_embeddings"], activation="silu")
    templates = []
    for attn_id, ffn_id, kind in schedule:
        ledger = fill_hybrid_layer(cfg, attn_id, ffn_id)
        suffix = "MoE" if ffn_id == FFN_MOE else "SwiGLU"
        templates.append(template(
            kind + " + " + suffix, ledger.rows, bb,
            attention=attn_id, ffn=ffn_id, links=layer_links(attn_id, ffn_id, cfg),
        ))
    shared = Ledger(QWEN_SOURCE)
    shared.tensor("embed", [cfg["vocab_size"], d], "embedding")
    shared.norm("final_norm", d)
    shared.linear("lm_head", d, cfg["vocab_size"], "head")
    if cfg.get("tie_word_embeddings"):
        shared.rows[-1].update(params=0, active_params=0, shared_with="embed")
    branches = []
    mtp_count = integer(cfg.get("mtp_num_hidden_layers", 0), "mtp_num_hidden_layers", low=0, high=16)
    if mtp_count:
        if cfg.get("mtp_use_dedicated_embeddings"):
            raise ValueError("dedicated MTP embeddings require separate accounting")
        full = next((t for t in templates if t["attn_kind"] == ATTN_GQA_GATE), None)
        if full is None:
            raise ValueError("MTP requires a full-attention block template")
        extra = Ledger(QWEN_SOURCE)
        extra.linear("fc", 2*d, d, "fusion")
        for name in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
            extra.norm(name, d)
        links = [(e["source"], e["target"]) for e in full["edges"]] + [
            ("token_embeddings", "pre_fc_norm_embedding.weight"),
            ("hidden_input", "pre_fc_norm_hidden.weight"),
            ("pre_fc_norm_embedding.weight", "fc"), ("pre_fc_norm_hidden.weight", "fc"),
            ("fc", "input"), ("output_add", "norm.weight"), ("norm.weight", "shared_lm_head"),
        ]
        extra.rows.append(module_row("shared_lm_head", "head", [cfg["vocab_size"], d], 0,
                                     shared_with="lm_head", source=QWEN_SOURCE, formula="shared; 0 unique"))
        branch = template("MTP", deepcopy(full["modules"]) + extra.rows, bb,
                          attention=ATTN_GQA_GATE, ffn=full["ffn_kind"], role="mtp", links=links)
        branch.update(id="mtp", count=mtp_count, included=options.get("include_mtp", False))
        branches.append(branch)
        metric = branch["metrics"]
        enabled = options.get("include_mtp", False)
        shared.rows.append(module_row(
            "mtp", "mtp", [], metric["params_total"]*mtp_count if enabled else 0,
            active=metric["params_active"]*mtp_count if enabled else 0,
            matrix=metric["matrix_params"]*mtp_count if enabled else 0,
            formula="full attention block + 2*d*d fusion + 3*d normalization; embed/head shared",
            source="Pinned checkpoint mtp.* index keys and shared full-attention operator shapes",
        ))
    cache = hybrid_cache(cfg, types, options)
    return assemble(cfg, bb, templates, shared.rows, options=options, cache=cache, branches=branches,
                    diagnostics=[dict(severity="info", code="mtp.scope",
                    message="Main inference decoder by default. MTP branch follows checkpoint keys; no speculative execution or speedup estimate.")])


def inspect_wrapped(cfg, **options):
    lang, vision = unwrap_language(cfg)
    result = inspect_grouped(lang, **options)
    return attach_optional_vision(result, cfg, lang, vision)


def inspect_linear(cfg, **options):
    lang, vision = unwrap_language(cfg)
    result = inspect_linear_text(lang, **options)
    return attach_optional_vision(result, cfg, lang, vision)


def inspect_linear_text(cfg, **options):
    validate_config(cfg)
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "vocab_size", "max_position_embeddings", "kv_lora_rank",
                "qk_nope_head_dim", "v_head_dim"):
        integer(cfg.get(key), key, high=1048576)
    if cfg.get("qk_rope_head_dim") is not None:
        integer(cfg["qk_rope_head_dim"], "qk_rope_head_dim", low=0, high=1048576)
    integer(cfg["num_hidden_layers"], "layers", high=4096)
    spec = cfg.get("linear_attn_config")
    if not isinstance(spec, dict):
        raise ValueError("KDA schedule requires linear_attn_config")
    for key in ("num_heads", "head_dim", "short_conv_kernel_size"):
        integer(spec.get(key), "linear_attn." + key, high=65536)
    if cfg.get("attention_bias") or cfg.get("mlp_bias"):
        raise ValueError("unsupported linear projection/activation variant")
    types = declared_layer_types(cfg)
    schedule = schedule_layers(cfg)
    if any(ffn_id == FFN_DENSE for _, ffn_id, _ in schedule):
        integer(cfg.get("intermediate_size"), "intermediate_size", high=1048576)
    if any(ffn_id in (FFN_MOE, FFN_LATENT_MOE) for _, ffn_id, _ in schedule):
        integer(cfg.get("moe_intermediate_size"), "moe_intermediate_size", high=1048576)
        integer(cfg.get("num_experts") or cfg.get("n_routed_experts"), "num_experts")
        integer(top_k_experts(cfg), "num_experts_per_tok")
        if top_k_experts(cfg) > (cfg.get("num_experts") or cfg.get("n_routed_experts")):
            raise ValueError("invalid top-k")
    if any(ffn_id == FFN_LATENT_MOE for _, ffn_id, _ in schedule):
        integer(cfg.get("routed_expert_hidden_size"), "routed_expert_hidden_size", high=1048576)
    d, h, kv = (cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads"))
    c = cfg.get("head_dim") or cfg["v_head_dim"]
    if h % kv:
        raise ValueError("invalid attention head grouping")
    dense = any(ffn_id == FFN_DENSE for _, ffn_id, _ in schedule)
    bb = dict(hidden_size=d, num_layers=len(types), num_heads=h, num_kv_heads=kv, head_dim=c,
              intermediate_size=cfg["intermediate_size"] if dense else cfg.get("moe_intermediate_size"),
              vocab_size=cfg["vocab_size"],
              context_length=cfg["max_position_embeddings"], activation=cfg.get("hidden_act", "silu"))
    mla = dict(
        q_lora_rank=cfg.get("q_lora_rank"), kv_lora_rank=cfg["kv_lora_rank"],
        qk_nope_head_dim=cfg["qk_nope_head_dim"], qk_rope_head_dim=cfg["qk_rope_head_dim"],
        v_head_dim=cfg["v_head_dim"],
    )
    templates = []
    for index, (attn_id, ffn_id, kind) in enumerate(schedule):
        ledger = fill_linear_layer(cfg, attn_id, ffn_id, index=index)
        display_ffn = "MOE" if ffn_id in (FFN_MOE, FFN_LATENT_MOE) else ffn_id
        suffix = "MoE" if display_ffn == "MOE" else "SwiGLU"
        display_attn = "DSA" if (
            attn_id == ATTN_MLA and any(row["id"] == "indexer.weights_proj" for row in ledger.rows)
        ) else attn_id
        item = template(
            ("DSA" if display_attn == "DSA" else kind) + " + " + suffix, ledger.rows, bb,
            attention=display_attn, ffn=display_ffn, links=layer_links(attn_id, ffn_id, cfg, index=index),
        )
        if attn_id == ATTN_MLA:
            item["mla"] = mla
        templates.append(item)
    shared = Ledger(KIMI_SOURCE)
    shared.tensor("embed", [cfg["vocab_size"], d], "embedding")
    shared.norm("final_norm", d)
    shared.linear("lm_head", d, cfg["vocab_size"], "head")
    if cfg.get("tie_word_embeddings"):
        shared.rows[-1].update(params=0, active_params=0, shared_with="embed")
    if cfg.get("attn_res_block_size"):
        integer(cfg["attn_res_block_size"], "attn_res_block_size", high=4096)
        shared.norm("output_attn_res_norm", d)
        shared.linear("output_attn_res_proj", d, 1, "residual")
    if cfg.get("hc_mult"):
        from app.engines.compressed_model import hyper_connection

        hyper_connection(shared, "hc_head", d, cfg["hc_mult"], head=True)
    branches = []
    mtp_count = integer(cfg.get("num_nextn_predict_layers", 0) or 0, "num_nextn_predict_layers", low=0, high=16)
    if mtp_count:
        full = next((t for t in templates if t["attn_kind"] in (ATTN_MLA, "DSA")), None)
        if full is None:
            raise ValueError("MTP requires an MLA/DSA block template")
        extra = Ledger(KIMI_SOURCE)
        extra.linear("eh_proj", 2 * d, d, "fusion")
        extra.norm("enorm", d)
        extra.norm("hnorm", d)
        extra.rows.append(module_row("shared_lm_head", "head", [cfg["vocab_size"], d], 0,
                                     shared_with="lm_head", source=KIMI_SOURCE, formula="shared; 0 unique"))
        links = [(e["source"], e["target"]) for e in full["edges"]] + [
            ("token_embeddings", "enorm.weight"), ("hidden_input", "hnorm.weight"),
            ("enorm.weight", "eh_proj"), ("hnorm.weight", "eh_proj"),
            ("eh_proj", "input"), ("output_add", "shared_lm_head"),
        ]
        branch = template("MTP", deepcopy(full["modules"]) + extra.rows, bb,
                          attention=full["attn_kind"], ffn=full["ffn_kind"], role="mtp", links=links)
        branch.update(id="mtp", count=mtp_count, included=options.get("include_mtp", False))
        branches.append(branch)
        metric = branch["metrics"]
        enabled = options.get("include_mtp", False)
        shared.rows.append(module_row(
            "mtp", "mtp", [], metric["params_total"] * mtp_count if enabled else 0,
            active=metric["params_active"] * mtp_count if enabled else 0,
            matrix=metric["matrix_params"] * mtp_count if enabled else 0,
            formula="MLA/DSA block + 2*d*d fusion + norms; embed/head shared",
            source="num_nextn_predict_layers with MLA methods",
        ))
    cache = kda_cache(cfg, types, options)
    return assemble(cfg, bb, templates, shared.rows, options=options, cache=cache, branches=branches,
                    diagnostics=[dict(severity="info", code="kda.scope",
                    message="KDA/MLA schedule from layer_types or kda_layers. DSA indexer, mHC and MTP extras included when present.")])
