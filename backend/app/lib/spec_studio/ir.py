from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["official", "calc-from-official", "web-ref", "assumed"]


class AttentionKind(str, Enum):
    MHA = "MHA"
    GQA = "GQA"
    MQA = "MQA"
    MLA = "MLA"
    CSA = "CSA"
    HCA = "HCA"
    SWA = "SWA"
    UNKNOWN = "UNKNOWN"


class FfnKind(str, Enum):
    DENSE = "DENSE"
    MOE = "MOE"


class NormKind(str, Enum):
    RMSNORM = "RMSNorm"
    LAYERNORM = "LayerNorm"


class NormMode(str, Enum):
    PRE = "pre"
    POST = "post"
    SANDWICH = "sandwich"


class PositionalKind(str, Enum):
    ROPE = "RoPE"
    YARN = "YaRN"
    LLAMA3 = "RoPE-Llama3"
    ALIBI = "ALiBi"
    LEARNED = "learned"
    NOPE = "NoPE"


class ResidualKind(str, Enum):
    ADD = "add"
    MHC = "mHC"


class Evidence(BaseModel):
    signal: str
    config_key: str
    value: Any = None
    confidence: Confidence = "official"


class Capability(BaseModel):
    id: str
    label: str
    kind: str
    evidence: list[Evidence] = Field(default_factory=list)
    supported: bool = True
    notes: str | None = None


class MLAConfig(BaseModel):
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


class MoEConfig(BaseModel):
    n_routed: int
    top_k: int
    n_shared: int = 0
    expert_dim: int
    dense_prefix: int = 0
    scoring_func: str | None = None
    topk_method: str | None = None
    n_group: int | None = None
    topk_group: int | None = None


class Backbone(BaseModel):
    vocab_size: int
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    context_length: int
    activation: str
    attention_bias: bool = False
    qkv_bias: bool = False
    o_bias: bool = False
    mlp_bias: bool = False
    tied_embeddings: bool = False
    mtp_layers: int = 0
    sliding_window: int | None = None


class LayerSpec(BaseModel):
    index: int
    count: int = 1
    ranges: list[list[int]] = Field(default_factory=list)
    attn_kind: AttentionKind
    ffn_kind: FfnKind
    extras: tuple[str, ...] = ()
    params_total: int = 0
    params_active: int = 0
    attn_params: int = 0
    ffn_params_total: int = 0
    ffn_params_active: int = 0
    norm_params: int = 0
    role: str = "main"


class TensorModule(BaseModel):
    name: str
    shape: list[int]
    params: int
    dtype: str
    per_layer: bool = False
    notes: str | None = None


class CalcRecord(BaseModel):
    calc_id: str
    formula: str
    substitutions: dict[str, Any] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    result: int | float
    unit: str
    source: str
    confidence: Confidence
    notes: str | None = None


class KvLayout(BaseModel):
    kind: str
    formula: str
    bytes_per_token: int
    dtype_assumed: str
    notes: str | None = None


class ServingMetrics(BaseModel):
    """Phase-2 hook. Names align with AGENT_BOARD style keys."""

    nparams_total: int
    nparams_active: int
    nparams_active_body: int
    weight_bytes: int
    weight_bytes_gib: float
    flop_gemm_decode: float
    kv_layout: KvLayout
    include_mtp: bool = False
    weight_bytes_logical: int = 0
    weight_bytes_source: str = "estimated"
    index_total_size: int | None = None
    hub_used_storage: int | None = None


class Identity(BaseModel):
    name: str
    repo_id: str | None = None
    revision: str | None = None
    sha: str | None = None
    model_type: str
    architectures: list[str] = Field(default_factory=list)


class UnreadField(BaseModel):
    key: str
    value_preview: Any = None
    reason: str = "not consumed by matcher"


class GraphNode(BaseModel):
    id: str
    kind: str
    label: str
    params: int = 0
    capability_ids: list[str] = Field(default_factory=list)
    parent: str | None = None
    layer_index: int | None = None
    detail: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str


class GraphIR(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class Totals(BaseModel):
    total: int
    active: int
    active_body: int
    embed: int
    lm_head: int
    attn: int
    ffn_total: int
    ffn_active: int
    other: int
    mtp: int = 0
    bytes_by_dtype: dict[str, float] = Field(default_factory=dict)


class ArchIR(BaseModel):
    engine_version: str = "2.0.0"
    formula_version: str = "2.0.0"
    identity: Identity
    backbone: Backbone
    capabilities: list[Capability]
    attention_kind: AttentionKind
    ffn_kind: FfnKind
    norm_kind: NormKind
    norm_mode: NormMode
    positional: PositionalKind
    residual: ResidualKind
    mla: MLAConfig | None = None
    moe: MoEConfig | None = None
    layers: list[LayerSpec]
    modules: list[TensorModule]
    calcs: list[CalcRecord]
    totals: Totals
    serving: ServingMetrics
    warnings: list[str] = Field(default_factory=list)
    infos: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    unread_fields: list[UnreadField] = Field(default_factory=list)
    dtype_weight: str = "bf16"
    dtype_expert: str | None = None


class InspectResult(BaseModel):
    arch: ArchIR
    graph: GraphIR
    glossary: dict[str, Any] = Field(default_factory=dict)
