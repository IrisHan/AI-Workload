"""Public response envelopes; evolving engine detail is retained as JSON payload."""
from typing import Any, Literal

from pydantic import Field

from .schemas import DesignInput, Spec

RunStatus = Literal["queued", "running", "cancel_requested", "completed", "partial",
                    "cancelled", "failed", "interrupted"]


class RevisionDTO(Spec):
    id: str
    design_id: str
    revision_no: int
    input: DesignInput
    input_hash: str
    created_at: str


class DesignDTO(Spec):
    id: str
    name: str
    head_revision: int
    created_at: str
    updated_at: str
    etag: str
    revision: RevisionDTO
    runs: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str
    hardware_name: str


class RunDTO(Spec):
    id: str
    design_id: str
    revision_id: str
    design_input: DesignInput
    structure_snapshot_id: str
    status: RunStatus
    input_hash: str
    versions: dict[str, Any]
    queued_at: str
    started_at: str | None
    finished_at: str | None
    attempt: int
    progress: dict[str, Any]
    summary: dict[str, Any] | None
    error: dict[str, Any] | None
    source_updates: list[dict[str, Any]]


class PageDTO(Spec):
    items: list[dict[str, Any]]
    next_cursor: str | None = None


class DetailPageDTO(Spec):
    items: list[dict[str, Any]]
    total: int
    next_offset: int | None
    result_hash: str


class PlanDTO(Spec):
    id: str
    run_id: str
    rank: int
    candidate_key: str
    result_hash: str
    summary: dict[str, Any]


class PlansDTO(PageDTO):
    search_complete: bool
    minimum_proven: bool
    coverage: dict[str, Any]


class GraphDTO(Spec):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    source_structure_id: str
    plan_id: str
    scope: dict[str, Any]
    layout_hint: str
    truncated: bool
    drilldown: dict[str, Any] | None
    result_hash: str
    network: dict[str, Any] | None = None


class PreviewDTO(Spec):
    client_revision: int
    coverage: Literal["static_lower_bound", "constraints_only", "timeout"]
    needs_full_run: bool
    capacity_status: Literal["unknown", "fail"]
    lower_bound_bytes: int | None
    diagnostics: list[dict[str, Any]]
    elapsed_ms: float


class InspectionDTO(Spec):
    result_hash: str
    source_structure_id: str
    scope: dict[str, Any]
    resident: bool
    placement: dict[str, int]
    shards: list[dict[str, Any]]
    local_tokens: int
    resident_request_quota: int
    cache: dict[str, int]
    allocations: list[dict[str, Any]]
    activation_peak_bytes: int
    timing: dict[str, Any]
    explanation: dict[str, Any] | None = None


class CapabilitiesDTO(Spec):
    supported: bool
    issues: list[dict[str, Any]]
    model: dict[str, Any] | None
    hardware: dict[str, Any] | None
    topology_defaults: dict[str, Any]
    parallel_domains: dict[str, list[int]] | None
    parallel_domains_by_layout: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    layouts: list[str] = Field(default_factory=list)
    structure_id: str | None
    precisions: list[str]
    overrides: dict[str, Any]
    runtime_profiles: list[str]
    layout: str
    etp: str
    topology_templates: list[str]
    limits: dict[str, Any]


class CompareDTO(Spec):
    plans: list[PlanDTO]
    input_differences: list[dict[str, Any]]
    versions: list[dict[str, Any]]
    assumptions: list[dict[str, Any]]
