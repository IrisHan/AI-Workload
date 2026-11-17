"""Current engine capabilities derived from immutable model configuration."""
import json
from functools import lru_cache

from app.engines.model_analyzer import validate_legacy_config
from app.engines.model_structure import inspect_model

CAPABILITY_VERSION = "model-capabilities-1.1.0"


@lru_cache(maxsize=128)
def _capability(config_json):
    config = json.loads(config_json)
    try:
        validate_legacy_config(config)
        return True, ()
    except ValueError:
        pass
    try:
        result = inspect_model(config)
    except ValueError as exc:
        return False, (str(exc),)
    supported = bool((result.get("morphology") or {}).get("cost_supported"))
    notes = (
        "Workload uses morphology-workload-1.1.0 with operator-cost-2.2.0; not the dense GQA closed form."
        if supported else
        "Structure supported; closed-form Workload waits on remaining layer methods."
    )
    return supported, (notes,)


def current_capabilities(model):
    supported, notes = _capability(json.dumps(model.raw_config, sort_keys=True))
    return model.model_copy(update=dict(
        workload_supported=supported, support_notes=list(notes), capability_version=CAPABILITY_VERSION,
    ))
