"""Versioned hardware catalog contract; SI units and explicit evidence."""
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Precision = Literal["fp32", "tf32", "fp16", "bf16", "fp8", "fp4", "int8"]
Confidence = Literal["official", "vendor_system", "user_reference", "assumed", "unknown"]
VERIFIED = {"official", "vendor_system"}


class Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Source(Spec):
    id: str
    title: str
    url: str | None = None
    retrieved_at: str
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    artifact: str


class Evidence(Spec):
    confidence: Confidence
    source_ids: list[str]
    note: str


class Compute(Spec):
    dense: float | None = Field(default=None, gt=0)
    sparse: float | None = Field(default=None, gt=0)
    unit: Literal["TFLOPS", "TOPS"] = "TFLOPS"
    status: Literal["supported", "unsupported", "unverified", "unknown"]
    evidence: Evidence

    @model_validator(mode="after")
    def check_status(self):
        if self.status in {"unknown", "unsupported"} and (self.dense is not None or self.sparse is not None):
            raise ValueError("Unknown/unsupported precision cannot carry a numeric peak")
        if self.status == "supported" and (self.dense is None or self.evidence.confidence not in VERIFIED):
            raise ValueError("Supported precision requires a verified dense peak")
        return self


class LinkSpec(Spec):
    name: str
    bandwidth_gb_s: float | None = Field(default=None, ge=0)
    direction: Literal["bidirectional", "unidirectional", "unspecified"]
    scope: str
    evidence: Evidence


class NodeContext(Spec):
    name: str
    accelerators: int | None = Field(default=None, gt=0)
    nic_gb_s_per_gpu: float | None = Field(default=None, ge=0)
    nic_direction: Literal["unidirectional"] = "unidirectional"
    full_mesh: bool | None = None
    evidence: Evidence


class Cost(Spec):
    cny_per_day: float | None = Field(default=None, ge=0)
    basis: Literal["user_reference", "capex", "cloud", "unknown"]
    as_of: str | None
    evidence: Evidence


class Education(Spec):
    title: str
    text: str
    source_ids: list[str]


class HardwareRecord(Spec):
    schema_version: Literal["hardware-1.0"] = "hardware-1.0"
    id: str = Field(max_length=100)
    sku: str
    name: str
    vendor: str
    family: str
    architecture: str
    form_factor: Literal["SXM", "PCIe", "Module"]
    memory_type: str
    memory_bytes: int = Field(gt=0)
    memory_bandwidth_bytes_s: float = Field(gt=0)
    bf16_tflops: float | None = Field(default=None, gt=0)
    fp32_tflops: float | None = Field(default=None, gt=0)
    nvlink_gb_s: float | None = Field(default=None, ge=0)
    power_watts: float | None = Field(default=None, gt=0)
    source_url: str
    retrieved_at: str
    version: str
    notes: str
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    compute: dict[Precision, Compute]
    interconnect: list[LinkSpec]
    node: NodeContext
    cost: Cost
    field_evidence: dict[str, Evidence]
    education: list[Education]
    sources: list[Source]
    caveats: list[str]
    legacy: bool = False

    @model_validator(mode="after")
    def consistent(self):
        for field, precision in (("bf16_tflops", "bf16"), ("fp32_tflops", "fp32")):
            peak = self.compute.get(precision)
            if getattr(self, field) != (peak.dense if peak else None):
                raise ValueError(f"{field} differs from canonical compute")
        ids = {source.id for source in self.sources}
        if len(ids) != len(self.sources):
            raise ValueError("Duplicate source ID")
        for key in ("memory_bytes", "memory_bandwidth_bytes_s", "power_watts"):
            if key not in self.field_evidence:
                raise ValueError(f"Missing field evidence: {key}")
        evidence = [*self.field_evidence.values(), self.node.evidence, self.cost.evidence]
        evidence += [c.evidence for c in self.compute.values()]
        evidence += [c.evidence for c in self.interconnect]
        for item in evidence:
            if not set(item.source_ids) <= ids:
                raise ValueError("Unknown evidence source")
            if item.confidence in VERIFIED and not item.source_ids:
                raise ValueError("Verified data requires a source")
        if any(not set(item.source_ids) <= ids for item in self.education):
            raise ValueError("Unknown education source")
        return self

    @computed_field
    @property
    def planner_precisions(self) -> list[str]:
        memory_verified = all(
            self.field_evidence[key].confidence in VERIFIED
            for key in ("memory_bytes", "memory_bandwidth_bytes_s")
        )
        return [
            p for p in ("bf16", "fp32")
            if memory_verified and p in self.compute and self.compute[p].status == "supported"
        ]


def payload_hash(payload: dict) -> str:
    content = {k: v for k, v in payload.items() if k not in {"content_sha256", "planner_precisions"}}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


class RooflinePoint(Spec):
    intensity: float
    performance_tops: float


class RooflineResult(Spec):
    hardware_id: str
    hardware_version: str
    hardware_sha256: str
    formula_version: str = "hardware-roofline-1.0.0"
    precision: Precision
    sparse: bool
    unit: Literal["TFLOPS", "TOPS"]
    peak_tops: float
    bandwidth_bytes_s: float
    ridge_point: float
    operating_point: RooflinePoint
    bottleneck: Literal["compute", "memory"]
    curve: list[RooflinePoint]
    assumptions: list[str]
