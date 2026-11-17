"""Architecture morphology tags. Methods authorize compute; family strings do not."""
import re
from functools import lru_cache

COST_ATTN = {
    "MHA", "GQA", "MQA", "MLA", "DSA", "MSA", "KDA",
    "GQA_GATE", "GATED_DELTA", "SWA", "CSA", "HCA",
}
COST_FFN = {"DENSE", "MOE", "LATENT_MOE"}


def collect_morphology(analysis):
    templates = list(analysis.get("templates") or [])
    branches = list(analysis.get("branches") or [])
    attn = []
    ffn = []
    for item in templates + branches:
        kind = item.get("attn_kind")
        if kind and kind not in attn:
            attn.append(kind)
        ffn_kind = item.get("ffn_kind")
        modules = item.get("modules") or []
        if any(row.get("id") == "routed_expert_down_proj" for row in modules):
            ffn_kind = "LATENT_MOE"
        if ffn_kind and ffn_kind not in ffn:
            ffn.append(ffn_kind)
    vision = "none"
    if any(item.get("role") == "vision" or item.get("attn_kind") == "VISION" for item in branches):
        ids = {row["id"] for item in branches for row in item.get("modules") or []}
        if any(name.startswith("sam.") or name.startswith("siglip.") for name in ids):
            vision = "SAM_SIGLIP"
        elif any(name == "vision.class_embedding" or name.startswith("vision.mlp1") for name in ids):
            vision = "InternViT"
        elif any("projector" in name for name in ids):
            vision = "CLIP"
        elif any(name.startswith("vt_") for name in ids):
            vision = "MoonViT"
        elif any("merger.gate_proj" in name or "merger.up_proj" in name for name in ids):
            vision = "QwenViT"
        elif any("patch" in name or "merger" in name for name in ids):
            vision = "MoonViT"
        else:
            vision = "VISION"
    quant = "none"
    for item in templates + branches:
        q = item.get("quantization") if isinstance(item.get("quantization"), dict) else None
        if q:
            quant = _quant_tag(q)
            break
    if quant == "none":
        # grouped templates do not copy quantization_config; callers may pass it.
        pass
    return {
        "attn": attn,
        "ffn": ffn,
        "vision": vision,
        "quant": quant,
        "cost_supported": cost_supported(attn, ffn),
    }


def _quant_tag(quant):
    method = quant.get("quant_method")
    fmt = str(quant.get("format") or "").lower()
    if method == "fp8" or "fp8" in fmt or method == "mxfp8":
        return "fp8"
    if method == "mxfp4" or "mxfp4" in fmt:
        return "mxfp4-group"
    if method == "compressed-tensors":
        if "mxfp4" in fmt:
            return "mxfp4-group"
        return "int4-group"
    if method in ("awq", "gptq"):
        return "int4-group"
    if method:
        return str(method)
    return "none"


def attach_morphology(analysis, config=None):
    morph = collect_morphology(analysis)
    if config and morph["quant"] == "none":
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            inner = config.get("text_config")
            quant = inner.get("quantization_config") if isinstance(inner, dict) else None
        if isinstance(quant, dict):
            morph["quant"] = _quant_tag(quant)
            morph["cost_supported"] = cost_supported(morph["attn"], morph["ffn"])
    analysis["morphology"] = morph
    return analysis


def cost_supported(attn, ffn):
    return bool(attn) and all(kind in COST_ATTN or kind == "VISION" for kind in attn) and all(
        kind in COST_FFN or kind == "GELU" for kind in ffn
    )


@lru_cache(maxsize=128)
def _ignore_patterns(patterns):
    result = []
    for pat in patterns:
        expr = pat[3:] if pat.startswith("re:") else r"(?:^|\.)" + re.escape(pat).replace(r"\*", ".*") + r"(?:\.|$)"
        try:
            result.append(re.compile(expr))
        except re.error:
            continue
    return result


def module_ignored(module_id, category, ignore, *, role=None, paths=None):
    aliases = [module_id, *(paths or [])]
    if category == "attention" or module_id.startswith(
        ("q_", "k_", "v_", "kv_", "o_", "g_proj", "indexer", "compressor", "attn_sink")
    ):
        aliases.append(f"self_attn.{module_id}")
        aliases.append("self_attn")
    if "shared_expert" in module_id:
        aliases.extend(["shared_experts", "mlp.shared_experts", f"mlp.{module_id}"])
    if module_id in {"gate_proj", "up_proj", "down_proj"}:
        aliases.append(f"mlp.{module_id}")
    if category == "head" or module_id in {"lm_head", "shared_lm_head"}:
        aliases.append("lm_head")
    if category == "vision" or role == "vision":
        aliases.append("vision_tower")
        if category == "connector":
            aliases.append("mm_projector")
    patterns = ignore if isinstance(ignore, list) else []
    for rx in _ignore_patterns(tuple(pat for pat in patterns if isinstance(pat, str))):
        if any(rx.search(name) for name in aliases):
            return True
    return False
