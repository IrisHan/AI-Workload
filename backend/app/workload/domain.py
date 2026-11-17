"""Frozen source envelope and canonical hashes shared by service and workers."""
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.engines.inference.ir import ENGINE_VERSION, ModelIR

VERSIONS = {
    "engine": ENGINE_VERSION, "schema": "workload-design-1.0", "adapter": "4",
    "partition": "framework-layout-v2", "memory": "2", "compute": "4",
    "communication": "3", "scheduler": "4", "search": "3", "speculative_cycle": "1",
    "explanation": "1",
}


def canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenSources:
    model: dict
    structure: dict
    hardware: dict
    ir: ModelIR
    topology: dict
    assumptions: dict
