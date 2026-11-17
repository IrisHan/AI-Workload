from typing import Literal

from app.hardware.schemas import Source, Spec
from pydantic import Field


class ArchitectureFact(Spec):
    key: str
    label: str
    value: str
    scope: str
    status: Literal["official", "reference", "derived", "unknown", "catalog"]
    source_ids: list[str]
    note: str = ""


class ContentBlock(Spec):
    kind: Literal["text", "table"]
    text: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class ArchitectureSection(Spec):
    number: int = Field(ge=1, le=8)
    title: str
    notes: list[str]
    blocks: list[ContentBlock]
    reference_blocks: list[ContentBlock]
    reference_text: str


class ArchitectureTemplate(Spec):
    id: str
    name: str
    version: str
    content_sha256: str
    specs: list[ArchitectureFact]
    explanations: list[str]


class ArchitectureVariant(Spec):
    id: str
    architecture_id: str
    source_family: str | None
    facts: list[ArchitectureFact]
    notes: dict[str, list[str]]


class ArchitectureDocument(Spec):
    hardware_id: str
    hardware_version: str
    hardware_sha256: str
    architecture: ArchitectureTemplate
    variant: ArchitectureVariant
    document_version: str
    content_sha256: str
    headline_metrics: list[ArchitectureFact]
    memory_hierarchy: list[ArchitectureFact]
    sections: list[ArchitectureSection]
    sources: list[Source]
    related_skus: list[dict[str, str]]
    comparison: list[ContentBlock]
    caveats: list[str]
