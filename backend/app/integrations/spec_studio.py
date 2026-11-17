"""Planner-owned adapter to the transplanted pure Spec Studio library."""
import copy

from app.domain.specs import ModelSpec
from app.engines.model_analyzer import analyze
from app.engines.model_structure import breakdown, inspect_model


def structure(model: ModelSpec, *, kv_dtype: str = "bf16", include_mtp: bool = False) -> dict:
    legacy = analyze(model)
    result = inspect_model(
        copy.deepcopy(model.raw_config), name=model.name, repo_id=model.repo_id,
        revision=model.revision, sha=model.revision, config_sha256=model.config_sha256,
        fetched_at=model.retrieved_at,
        kv_dtype=kv_dtype, include_mtp=include_mtp,
    )
    if result["status"] != "supported" or result["source"]["config_sha256"] != model.config_sha256:
        raise ValueError("Untrusted or unsupported structure analysis")
    parts = result["totals"]
    candidate = breakdown(result)
    deltas = {key: candidate.get(key, 0) - legacy.parameter_breakdown.get(key, 0)
              for key in candidate.keys() | legacy.parameter_breakdown.keys()}
    return {
        "analysis": copy.deepcopy(result),
        "comparison": {
            "baseline_engine": legacy.engine_version,
            "baseline_formula": legacy.formula_version,
            "baseline_parameters": legacy.parameters,
            "candidate_parameters": parts["total"],
            "parameter_delta": parts["total"] - legacy.parameters,
            "category_deltas": deltas,
            "matched": not any(deltas.values()) and parts["total"] == legacy.parameters,
        },
    }
