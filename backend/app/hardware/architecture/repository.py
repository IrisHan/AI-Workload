import hashlib
import json
from functools import lru_cache
from pathlib import Path

from app.hardware.architecture.schemas import ArchitectureTemplate, ArchitectureVariant
from app.hardware.schemas import Source

DATA = Path(__file__).parent / "data" / "architecture-20260914.json"


def content_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        {k: v for k, v in value.items() if k != "content_sha256"},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


@lru_cache(maxsize=1)
def load_bundle():
    bundle = json.loads(DATA.read_text())
    if content_hash(bundle) != bundle["content_sha256"]:
        raise ValueError("Architecture bundle checksum mismatch")
    ids = {s["id"] for s in bundle["sources"]}
    for source in bundle["sources"]:
        Source.model_validate(source)
    for template in bundle["templates"]:
        ArchitectureTemplate.model_validate(template)
        if content_hash(template) != template["content_sha256"]:
            raise ValueError("Architecture template checksum mismatch")
        if any(not set(f["source_ids"]) <= ids for f in template["specs"]):
            raise ValueError("Unknown architecture source")
    for variant in bundle["variants"]:
        ArchitectureVariant.model_validate(variant)
        if any(not set(f["source_ids"]) <= ids for f in variant["facts"]):
            raise ValueError("Unknown variant source")
    return bundle
