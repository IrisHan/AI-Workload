"""Strict public input contracts for the inference deployment designer."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[int, Field(strict=True, ge=1, le=16777216)]
GroupSize = Annotated[int, Field(strict=True, ge=1, le=256)]
Identifier = Annotated[str, Field(min_length=1, max_length=100)]
Precision = Literal["bf16", "fp32"]


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)


class Load(Spec):
    prompt_tokens: Positive = 2048
    output_tokens: Positive = 128
    concurrency: Annotated[int, Field(ge=1, le=4096)] = 8
    request_rate: Annotated[float, Field(gt=0, le=1000000)] | None = 1.0
    request_count: Positive | None = None
    arrival: Literal["deterministic", "poisson"] | None = "deterministic"
    seed: Annotated[int, Field(ge=0, le=2147483647)] = 1
    duration_seconds: Annotated[float, Field(gt=0, le=3600)] = 30.0
    warmup_seconds: Annotated[float, Field(ge=0, le=3600)] = 0.0

    @model_validator(mode="after")
    def window(self):
        if self.warmup_seconds >= self.duration_seconds:
            raise ValueError("warmup_seconds must be less than duration_seconds")
        return self


class TopologyOverrides(Spec):
    intra_bandwidth_bytes_s: Annotated[float, Field(gt=0)] | None = None
    inter_bandwidth_bytes_s: Annotated[float, Field(gt=0)] | None = None
    nic_count: GroupSize | None = None
    nic_bandwidth_bytes_s: Annotated[float, Field(gt=0)] | None = None
    oversubscription: Annotated[float, Field(ge=1, le=100)] = 1.0
    link_efficiency: Annotated[float, Field(gt=0, le=1)] = 0.7
    inter_link_efficiency: Annotated[float, Field(gt=0, le=1)] = 1.0
    intra_latency_us: Annotated[float, Field(ge=0, le=1000000)] = 5.0
    inter_latency_us: Annotated[float, Field(ge=0, le=1000000)] = 10.0


class Resources(Spec):
    node_count: GroupSize = 1
    gpus_per_node: GroupSize = 8
    topology_template: Literal[
        "single_gpu-v1", "switched_sxm_reference-v1", "pcie_shared_root-v1", "rdma_reference-v1"
    ] = "switched_sxm_reference-v1"
    topology_overrides: TopologyOverrides = Field(default_factory=TopologyOverrides)

    @model_validator(mode="after")
    def budget(self):
        if self.node_count * self.gpus_per_node > 256:
            raise ValueError("GPU budget exceeds 256")
        if self.topology_template == "single_gpu-v1" and self.node_count * self.gpus_per_node != 1:
            raise ValueError("single_gpu-v1 requires exactly one GPU")
        return self


class Speculative(Spec):
    enabled: bool = False
    draft_tokens: Annotated[int, Field(ge=1, le=64)] = 5
    acceptance_rate: Annotated[float, Field(ge=0, le=1)] = 0.75
    draft_latency_ms: Annotated[float, Field(ge=0)] | None = None
    draft_memory_bytes: Annotated[int, Field(ge=0, le=9007199254740991)] | None = None
    include_prefill: bool = False


class Runtime(Spec):
    profile: Literal["analytical-continuous-v1", "analytical-sequential-v1"] = "analytical-continuous-v1"
    weight_dtype: Precision = "bf16"
    activation_dtype: Precision = "bf16"
    kv_dtype: Precision = "bf16"
    max_batch_sequences: Annotated[int, Field(ge=1, le=4096)] = 8
    max_batched_tokens: Positive = 2048
    prefill_chunk_tokens: Positive = 512
    continuous_batching: bool = True
    kv_page_tokens: Literal[16] = 16
    speculative: Speculative = Field(default_factory=Speculative)
    fairness_interval: Annotated[int, Field(ge=1, le=100)] = 8

    @model_validator(mode="after")
    def profile_consistency(self):
        if self.continuous_batching != (self.profile == "analytical-continuous-v1"):
            raise ValueError("continuous_batching must match runtime profile")
        if self.prefill_chunk_tokens > self.max_batched_tokens:
            raise ValueError("prefill chunk exceeds token budget")
        return self


class LayerLock(Spec):
    layer_index: Annotated[int, Field(ge=0, le=100000)] 
    allowed_tp: Annotated[list[GroupSize], Field(min_length=1, max_length=256)]


class ManualCandidate(Spec):
    tp: GroupSize
    pp: GroupSize
    ep: GroupSize
    dp: GroupSize
    boundaries: Annotated[list[Annotated[int, Field(ge=1)]], Field(max_length=255)]

    @model_validator(mode="after")
    def boundaries_count(self):
        if len(self.boundaries) != self.pp - 1 or sorted(set(self.boundaries)) != self.boundaries:
            raise ValueError("boundaries must be strictly increasing with pp-1 entries")
        if self.tp * self.pp * self.dp > 256:
            raise ValueError("candidate exceeds 256 GPUs")
        return self


class Parallelism(Spec):
    layout: Literal["orthogonal-tp-ep-v1", "vllm-v0.10.2", "sglang-v0.5.3"] = "orthogonal-tp-ep-v1"
    expert_parallel: bool = False
    mode: Literal["auto", "manual"] = "auto"
    allowed_tp: Annotated[list[GroupSize], Field(min_length=1, max_length=256)] | None = None
    allowed_pp: Annotated[list[GroupSize], Field(min_length=1, max_length=256)] | None = None
    allowed_ep: Annotated[list[GroupSize], Field(min_length=1, max_length=256)] | None = None
    dp_scaling: bool = False
    layer_locks: Annotated[list[LayerLock], Field(max_length=1024)] = Field(default_factory=list)
    pp_boundaries: Annotated[list[Annotated[int, Field(ge=1)]], Field(max_length=255)] | None = None
    manual_candidate: ManualCandidate | None = None

    @model_validator(mode="after")
    def manual(self):
        if (self.mode == "manual") != (self.manual_candidate is not None):
            raise ValueError("manual mode requires manual_candidate; auto mode must omit it")
        if self.pp_boundaries is not None and sorted(set(self.pp_boundaries)) != self.pp_boundaries:
            raise ValueError("pp_boundaries must be strictly increasing")
        return self


class Targets(Spec):
    ttft_ms: Annotated[float, Field(gt=0)] | None = None
    itl_ms: Annotated[float, Field(gt=0)] | None = None
    policy: Literal["soft", "hard"] = "soft"


OVERRIDE_DEFAULTS = {
    "activation_efficiency": (1.0, 0.01, 1.0),
    "kv_efficiency": (1.0, 0.01, 1.0),
    "memory_efficiency": (0.95, 0.01, 1.0),
    "runtime_fixed_bytes": (1073741824, 0, 9007199254740991),
    "safety_fraction": (0.10, 0.0, 0.5),
    "compute_efficiency": (0.5, 0.01, 1.0),
    "hbm_efficiency": (0.7, 0.01, 1.0),
    "launch_time_us": (5.0, 0.0, 1000000),
    "routing_imbalance": (1.25, 1.0, 256.0),
    "comm_reserve_bytes": (0, 0, 9007199254740991),
    "hbm_bandwidth_gb_s": (0, 0, 1000000),
}


class Override(Spec):
    path: str
    value: int | float
    reason: Annotated[str, Field(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def whitelist(self):
        if self.path not in OVERRIDE_DEFAULTS:
            raise ValueError("Only runtime assumption overrides are permitted")
        _, low, high = OVERRIDE_DEFAULTS[self.path]
        if not low <= self.value <= high:
            raise ValueError(f"{self.path} must be in [{low}, {high}]")
        if self.path.endswith("_bytes") and type(self.value) is not int:
            raise ValueError("Byte overrides require exact integers")
        if not self.reason.strip():
            raise ValueError("Override reason must not be blank")
        return self


class DesignInput(Spec):
    schema_version: Literal["workload-design-1.0"] = "workload-design-1.0"
    model_id: Identifier
    hardware_id: Identifier
    mode: Literal["online", "offline"] = "online"
    load: Load = Field(default_factory=Load)
    resources: Resources = Field(default_factory=Resources)
    runtime: Runtime = Field(default_factory=Runtime)
    parallelism: Parallelism = Field(default_factory=Parallelism)
    targets: Targets = Field(default_factory=Targets)
    overrides: Annotated[list[Override], Field(max_length=20)] = Field(default_factory=list)

    @model_validator(mode="after")
    def relationships(self):
        if self.mode == "online":
            if self.load.request_rate is None or self.load.arrival is None or self.load.request_count is not None:
                raise ValueError("online requires rate/arrival and no request_count")
        elif self.load.request_rate is not None or self.load.arrival is not None or self.load.request_count is None:
            raise ValueError("offline requires request_count and null rate/arrival")
        if self.runtime.max_batch_sequences > self.load.concurrency:
            raise ValueError("max_batch_sequences exceeds concurrency")
        if len({o.path for o in self.overrides}) != len(self.overrides):
            raise ValueError("Duplicate override path")
        return self


class Search(Spec):
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    max_candidates: Annotated[int, Field(ge=1, le=10000)] = 2000
    max_seconds: Annotated[float, Field(ge=1, le=120)] = 30.0
    event_budget: Annotated[int, Field(ge=1, le=1000000)] = 1000000


class CreateDesign(Spec):
    name: Annotated[str, Field(min_length=1, max_length=100)]
    input: DesignInput

    @model_validator(mode="after")
    def name_not_blank(self):
        if not self.name.strip():
            raise ValueError("Design name must not be blank")
        return self


class SaveRevision(Spec):
    input: DesignInput


class Preview(Spec):
    input: DesignInput
    client_revision: Annotated[int, Field(ge=0)]
    manual_candidate: ManualCandidate | None = None


class SubmitRun(Spec):
    design_revision_id: Identifier
    source_policy: Literal["current_engine", "pinned"] = "current_engine"
    structure_snapshot_id: Identifier | None = None
    search: Search = Field(default_factory=Search)

    @model_validator(mode="after")
    def pinned(self):
        if (self.source_policy == "pinned") != (self.structure_snapshot_id is not None):
            raise ValueError("pinned policy requires snapshot ID; current_engine must omit it")
        return self


class PlanRef(Spec):
    run_id: Identifier
    plan_id: Identifier


class Compare(Spec):
    plan_refs: Annotated[list[PlanRef], Field(min_length=2, max_length=4)]
