"""Architecture adapters select reusable operator rules, never repository names."""
from copy import deepcopy

from app.engines.layer_methods import (
    is_hybrid_language,
    is_kda_language,
    unimplemented_hybrid_modules,
    unimplemented_linear_modules,
)
from app.lib.spec_studio.grouped import inspect_grouped
from app.lib.spec_studio.validation import ensure_safe_numbers

FORMULA_VERSION = "model-components-1.1.0"
ENGINE_VERSION = "planner-components-1.2.0"
QUANT_METHODS = ("fp8", "gptq", "awq", "bitsandbytes", "compressed-tensors", "mxfp4", "mxfp8")


def overlay_sparse_attention(result, cfg, options):
    inner = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    has_indexer = any(
        row["id"].startswith("indexer.")
        for item in result.get("templates") or []
        for row in item.get("modules") or []
    )
    if not has_indexer:
        return result
    sparse = inner.get("sparse_attention_config") if isinstance(inner.get("sparse_attention_config"), dict) else {}
    index_dim = inner.get("index_head_dim") or cfg.get("index_head_dim") or sparse.get("sparse_index_dim")
    for item in result["templates"]:
        if any(row["id"] == "indexer.weights_proj" for row in item["modules"]):
            item["attn_kind"] = "DSA"
            item["label"] = item["label"].replace("MLA", "DSA")
        elif any(row["id"] == "indexer.q_proj" for row in item["modules"]) and not item.get("mla"):
            item["attn_kind"] = "MSA"
            item["label"] = item["label"].replace("GQA", "MSA")
    if index_dim and any(t["attn_kind"] == "DSA" for t in result["templates"]):
        elem = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[options.get("kv_dtype", "bf16")]
        layout = result["serving"]["kv_layout"]
        dsa_layers = sum(
            group["count"]
            for group in result["groups"]
            for item in result["templates"]
            if item["id"] == group["template_id"]
            and any(row["id"].startswith("indexer.") for row in item["modules"])
        )
        layout["bytes_per_token"] = (layout.get("bytes_per_token") or 0) + dsa_layers * int(index_dim) * elem
        layout["formula"] = "L * (kv_lora_rank + rope_dim + index_head_dim) * selected_dtype_bytes"
        layout["notes"] = "MLA latent plus index keys on DSA layers; top-k only limits attended tokens, not cache residency."
        result["capabilities"].append(dict(id="dsa", label="DSA indexer", supported=True, evidence=[]))
    if any(t["attn_kind"] == "MSA" for t in result["templates"]):
        result["capabilities"].append(dict(id="msa", label="MSA indexer", supported=True, evidence=[]))
    return result


def inspect_model(config, **options):
    cfg = deepcopy(config)
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    if options.get("kv_dtype", "bf16") not in ("bf16", "fp16", "fp32", "fp8"):
        raise ValueError("unsupported kv_dtype")
    for key in ("tie_word_embeddings", "attention_bias", "mlp_bias"):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    quant = cfg.get("quantization_config")
    if quant is not None and (
        not isinstance(quant, dict) or quant.get("quant_method") not in QUANT_METHODS
    ):
        raise ValueError("quantization_config requires a supported quant_method")
    family = cfg.get("model_type")
    wrapped = isinstance(cfg.get("text_config"), dict) or isinstance(cfg.get("vision_config"), dict)
    if family == "deepseek_v4":
        from app.engines.compressed_model import inspect_compressed

        result = inspect_compressed(cfg, **options)
    elif family == "multi_modality":
        from app.engines.composite_model import inspect_composite

        result = inspect_composite(cfg, **options)
    elif is_hybrid_language(cfg):
        unimplemented_hybrid_modules(cfg)
        from app.engines.composite_model import inspect_composite

        result = inspect_composite(cfg, **options)
    elif is_kda_language(cfg):
        from app.engines.composite_model import inspect_linear

        result = inspect_linear(cfg, **options)
    elif wrapped:
        unimplemented_linear_modules(cfg)
        from app.engines.composite_model import inspect_wrapped

        result = inspect_wrapped(cfg, **options)
        overlay_sparse_attention(result, cfg, options)
    else:
        unimplemented_linear_modules(cfg)
        result = inspect_grouped(cfg, **options)
        overlay_sparse_attention(result, cfg, options)
    from app.engines.model_analyzer import validate_legacy_config

    try:
        validate_legacy_config(cfg)
    except ValueError:
        result["formula_version"] = FORMULA_VERSION
        result["engine_version"] = ENGINE_VERSION
    if (family == "multi_modality" or is_hybrid_language(cfg) or is_kda_language(cfg) or wrapped) and options.get("config_sha256") is None:
        from app.lib.spec_studio.grouped import config_hash

        result["source"]["config_sha256"] = config_hash(cfg)
    from app.engines.morphology import attach_morphology
    attach_morphology(result, cfg)
    from app.engines.cache_layout import attach_cache_layout
    attach_cache_layout(result, cfg)
    from app.engines.checkpoint_paths import attach_checkpoint_paths
    attach_checkpoint_paths(result, cfg)
    ensure_safe_numbers(result)
    return result


def breakdown(result):
    total = result["totals"]
    parts = dict(embedding=total["embed"], attention=total["attn"], mlp=total["ffn_total"],
                 normalization=total["other"], output_head=total["lm_head"])
    if total.get("vision"):
        parts["vision"] = total["vision"]
    if total.get("mtp"):
        parts["mtp"] = total["mtp"]
    if sum(parts.values()) != total["total"]:
        raise ValueError("component category accounting does not conserve total")
    return parts
