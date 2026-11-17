from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class FrozenSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelSpec(FrozenSpec):
    id: str
    name: str
    provider: str
    repo_id: str
    revision: str
    architecture: str
    parameters: int
    layers: int
    hidden_size: int
    attention_heads: int
    kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    tie_word_embeddings: bool
    dtype: str
    is_moe: bool = False
    source_url: str
    retrieved_at: str
    config_sha256: str
    raw_config: dict[str, Any]
    version: str
    active_parameters: int | None = None
    parameter_scope: str = "main"
    analysis_formula: str = "dense-gqa-1.0.0"
    workload_supported: bool = True
    capability_version: str | None = None
    support_notes: list[str] = Field(default_factory=list)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class HardwareSpec(FrozenSpec):
    id: str
    name: str
    vendor: str
    memory_bytes: int
    memory_bandwidth_bytes_s: float
    bf16_tflops: float
    fp32_tflops: float
    nvlink_gb_s: float
    power_watts: float
    source_url: str
    retrieved_at: str
    version: str
    notes: str


class LayerAnalysis(FrozenSpec):
    name: str
    parameters: int
    input_shape: str
    output_shape: str


class ModelAnalysis(FrozenSpec):
    model_id: str
    parameters: int
    parameter_breakdown: dict[str, int]
    layers: list[LayerAnalysis]
    engine_version: str
    formula_version: str
    assumptions: list[str]


class StructureSnapshot(FrozenSpec):
    id: str
    model_id: str
    model_version: str
    engine_version: str
    formula_version: str
    created_at: str
    input_snapshot: dict[str, Any]
    analysis: dict[str, Any]
    comparison: dict[str, Any]


class StructureOptions(FrozenSpec):
    kv_dtype: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16"
    include_mtp: bool = Field(default=False, strict=True)


class ModelMemoryInput(FrozenSpec):
    batch_size: int = Field(default=1, ge=1, le=4096, strict=True)
    context_length: int = Field(default=4096, ge=1, le=16777216, strict=True)
    dtype: Literal["bf16", "fp32"] = "bf16"
    include_training_state: bool = Field(default=False, strict=True)


class StructureCostInput(FrozenSpec):
    phase: Literal["prefill", "decode"] = "decode"
    batch_size: int = Field(default=1, ge=1, le=4096, strict=True)
    sequence_length: int = Field(default=128, ge=1, le=16777216, strict=True)
    context_length: int = Field(default=128, ge=1, le=16777216, strict=True)
    dtype: Literal["bf16", "fp32"] = "bf16"
    hardware_id: str = Field(min_length=1, max_length=100)
