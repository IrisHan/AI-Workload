"""Planner-owned immutable execution and storage identities; no infrastructure."""
from dataclasses import dataclass

ENGINE_VERSION = "inference-planner-1.3.1"


class PlanningError(ValueError):
    def __init__(self, code, message, field_path=None, details=None):
        super().__init__(message)
        self.code = code
        self.field_path = field_path
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": str(self), "field_path": self.field_path,
                "retryable": False, "details": self.details}


@dataclass(frozen=True)
class Weight:
    storage_id: str
    module_id: str
    shape: tuple[int, ...]
    count: int
    bias: int
    dtype: str
    partition: str
    routed: bool
    alias_of: str | None = None
    checkpoint_paths: tuple[str, ...] = ()

    @property
    def numel(self):
        from math import prod
        return (prod(self.shape) + self.bias) * self.count


@dataclass(frozen=True)
class Operator:
    id: str
    method_id: str
    weight_refs: tuple[str, ...]
    inputs: tuple[str, ...]
    output: str
    source_node_id: str
    method_version: str = "1"


@dataclass(frozen=True)
class Layer:
    index: int
    instance_id: str
    template_id: str
    attention: str
    ffn: str
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    experts: int
    top_k: int
    window: int | None
    weights: tuple[Weight, ...]
    operators: tuple[Operator, ...]
    activation: str = "silu"


@dataclass(frozen=True)
class ModelIR:
    model_id: str
    structure_id: str
    revision: str
    config_sha256: str
    structure_engine: str
    structure_formula: str
    layers: tuple[Layer, ...]
    shared: tuple[Weight, ...]
    vocab: int
    hidden: int
    context_limit: int
    logical_numel: int


@dataclass(frozen=True)
class Candidate:
    tp: int
    pp: int
    ep: int
    dp: int
    boundaries: tuple[int, ...]
    layout: str = "orthogonal-tp-ep-v1"
    moe: bool = False

    @property
    def legacy(self):
        return self.layout == "orthogonal-tp-ep-v1"

    @property
    def attention_groups(self):
        return self.ep if self.legacy else self.dp if self.moe and self.layout == "vllm-v0.10.2" else 1

    @property
    def independent_replicas(self):
        return 1 if self.moe and self.layout == "vllm-v0.10.2" else self.dp

    @property
    def expert_tp(self):
        return self.tp * self.attention_groups // self.ep

    def expert_owner(self, rank):
        return (rank.e * self.tp + rank.t) // self.expert_tp

    @property
    def gpu_count(self):
        return self.tp * self.pp * self.attention_groups * self.independent_replicas

    @property
    def key(self):
        prefix = "" if self.legacy else self.layout + "-"
        return prefix + f"t{self.tp}-p{self.pp}-e{self.ep}-d{self.dp}-b" + "_".join(map(str, self.boundaries))

    def stage(self, layer):
        return sum(layer >= boundary for boundary in self.boundaries)


@dataclass(frozen=True)
class Rank:
    rank: int
    d: int
    p: int
    e: int
    t: int
    node: int
    gpu: int


@dataclass(frozen=True)
class Shard:
    storage_id: str
    module_id: str
    local_shape: tuple[int, ...]
    count: int
    bias: int
    bytes: int
    partition: str
    expert_start: int | None
    expert_stop: int | None
