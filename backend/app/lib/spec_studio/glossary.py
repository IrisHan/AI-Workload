from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

GLOSSARY_PATH = Path(__file__).resolve().parent / "glossary.yaml"


@lru_cache(maxsize=1)
def load_glossary() -> dict[str, Any]:
    if not GLOSSARY_PATH.exists():
        return {}
    data = yaml.safe_load(GLOSSARY_PATH.read_text()) or {}
    return data.get("entries", data)


def glossary_for(capability_ids: list[str]) -> dict[str, Any]:
    all_entries = load_glossary()
    out = {}
    for cid in capability_ids:
        if cid in all_entries:
            out[cid] = all_entries[cid]
    return out
