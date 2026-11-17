"""Hardware-owned access to the existing immutable version table."""
import json
from pathlib import Path

from app.domain.specs import HardwareSpec
from app.hardware.schemas import HardwareRecord, payload_hash
from app.infrastructure.database import HardwareRow
from sqlalchemy import select
from sqlalchemy.orm import Session

DATA = Path(__file__).parent / "data"
LEGACY_ID = "nvidia-h100-sxm-80gb-v1"


def read_record(payload: dict) -> HardwareRecord:
    if "schema_version" in payload:
        record = HardwareRecord.model_validate(payload)
        if payload_hash(payload) != record.content_sha256:
            raise ValueError(f"Hardware content hash mismatch: {record.id}")
        return record
    old = HardwareSpec.model_validate(payload)
    if old.id != LEGACY_ID:
        raise ValueError("Unknown legacy hardware schema")
    # Legacy payload is never rewritten. Presentation enrichment comes from its frozen fields.
    ev = dict(confidence="official", source_ids=["legacy"], note=old.notes)
    unknown = dict(confidence="unknown", source_ids=[], note="Not recorded in legacy version.")
    payload = dict(
        schema_version="hardware-1.0", **old.model_dump(), sku="nvidia-h100-sxm-80gb",
        family="H100", architecture="Hopper", form_factor="SXM", memory_type="HBM3",
        content_sha256="0" * 64, legacy=True, caveats=["Legacy v1: original source artifact was not archived."],
        compute={p: dict(dense=v, sparse=None, unit="TFLOPS", status="supported", evidence=ev)
                 for p, v in (("bf16", old.bf16_tflops), ("fp32", old.fp32_tflops))},
        interconnect=[dict(name="NVLink", bandwidth_gb_s=old.nvlink_gb_s, direction="bidirectional",
                           scope="GPU interconnect", evidence=ev)],
        node=dict(name="Not recorded", accelerators=None, nic_gb_s_per_gpu=None, full_mesh=None, evidence=unknown),
        cost=dict(cny_per_day=None, basis="unknown", as_of=None, evidence=unknown),
        field_evidence={key: ev for key in ("memory_bytes", "memory_bandwidth_bytes_s", "power_watts")},
        education=[], sources=[dict(id="legacy", title="Legacy H100 source", url=old.source_url,
                                    retrieved_at=old.retrieved_at, sha256=None, artifact="not-archived")],
    )
    record = HardwareRecord.model_validate(payload)
    normalized = record.model_dump(exclude={"planner_precisions"})
    normalized["content_sha256"] = payload_hash(normalized)
    return HardwareRecord.model_validate(normalized)


class HardwareRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, identifier: str) -> HardwareRecord:
        from app.repositories.catalog import MissingResource

        row = self.session.get(HardwareRow, identifier)
        if row is None:
            raise MissingResource(f"Hardware {identifier} not found")
        return read_record(row.payload)

    def list(self, include_legacy=False) -> list[HardwareRecord]:
        records = [read_record(row.payload) for row in self.session.scalars(select(HardwareRow))]
        return sorted(
            [r for r in records if include_legacy or not r.legacy],
            key=lambda r: (r.sku != "nvidia-h100-sxm-80gb", r.legacy, r.vendor, r.family, r.form_factor),
        )

    def seed(self):
        files = sorted(DATA.glob("*.json"))
        if not files:
            raise RuntimeError("Packaged hardware catalog is missing")
        for path in files:
            payload = json.loads(path.read_text())
            record = read_record(payload)
            existing = self.session.get(HardwareRow, record.id)
            if existing is not None:
                if existing.payload != payload:
                    raise RuntimeError(f"Immutable hardware version conflict: {record.id}")
                continue
            self.session.add(HardwareRow(id=record.id, payload=payload))
        self.session.commit()

    def calculation_input(self, identifier: str):
        row = self.session.get(HardwareRow, identifier)
        if row is not None and "schema_version" not in row.payload:
            return HardwareSpec.model_validate(row.payload)
        return self.get(identifier)
