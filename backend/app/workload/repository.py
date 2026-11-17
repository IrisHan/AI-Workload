"""Transactional Workload store; publication and lifecycle changes are atomic."""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import delete, func, select, text

from .domain import content_hash
from .tables import DesignRow, DetailRow, PlanRow, RevisionRow, RunRow

ACTIVE = ("queued", "running", "cancel_requested")
TERMINAL = ("completed", "partial", "cancelled", "failed", "interrupted")


def timestamp(seconds=0):
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


class StoreError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.details = status, code, details or {}


def require(row, code):
    if row is None:
        raise StoreError(404, code, "Requested Workload resource does not exist")
    return row


def etag(design):
    return f'"{design.id}:{design.head_revision}"'


def match(design, expected):
    if expected != etag(design):
        raise StoreError(409, "REVISION_CONFLICT", "Design changed; preserve draft and reload or save as new",
                         {"current_etag": etag(design)})


def bind_explanation_scope(value, run_id, plan_id, input_hash):
    """Bind before hashing publication; result_hash remains the outer envelope to avoid a cycle."""
    if isinstance(value, dict):
        if value.get("version") == "calculation-trace-1" or "formula" in value and "scope" in value:
            value["scope"].update(run_id=run_id, plan_id=plan_id, input_hash=input_hash)
        for item in value.values():
            bind_explanation_scope(item, run_id, plan_id, input_hash)
    elif isinstance(value, list):
        for item in value:
            bind_explanation_scope(item, run_id, plan_id, input_hash)


class Repository:
    def __init__(self, session):
        self.session = session

    def lock(self):
        # All callers enter write methods with a clean, dedicated session.
        if self.session.bind.dialect.name == "sqlite":
            self.session.execute(text("BEGIN IMMEDIATE"))
        else:
            raise StoreError(503, "WORKER_UNAVAILABLE", "Shared PostgreSQL worker locking is not validated")

    def design(self, identifier):
        return require(self.session.get(DesignRow, identifier), "DESIGN_NOT_FOUND")

    def revision(self, identifier):
        return require(self.session.get(RevisionRow, identifier), "REVISION_NOT_FOUND")

    def run(self, identifier):
        return require(self.session.get(RunRow, identifier), "RUN_NOT_FOUND")

    def plan(self, run_id, plan_id):
        row = self.session.scalar(select(PlanRow).where(PlanRow.id == plan_id, PlanRow.run_id == run_id))
        return require(row, "PLAN_NOT_IN_RUN")

    def head(self, design):
        return self.session.scalar(select(RevisionRow).where(
            RevisionRow.design_id == design.id, RevisionRow.revision_no == design.head_revision))

    def idempotent(self, row_type, key, request_hash):
        row = self.session.scalar(select(row_type).where(row_type.idempotency_key == key))
        if row and row.request_hash != request_hash:
            raise StoreError(409, "IDEMPOTENCY_CONFLICT", "Idempotency-Key was used for a different request")
        return row

    def create_design(self, body, key):
        self.lock()
        digest = content_hash(body)
        existing = self.idempotent(DesignRow, key, digest)
        if existing:
            return existing, False
        now = timestamp()
        row = DesignRow(id=uuid4().hex, name=body["name"].strip(), head_revision=1,
                        created_at=now, updated_at=now, idempotency_key=key, request_hash=digest)
        self.session.add(row)
        self.session.flush()
        self._revision(row, body["input"])
        self.session.commit()
        return row, True

    def _revision(self, design, payload):
        row = RevisionRow(id=uuid4().hex, design_id=design.id, revision_no=design.head_revision,
                          model_id=payload["model_id"], hardware_id=payload["hardware_id"],
                          payload=payload, input_hash=content_hash(payload),
                          schema_version=payload["schema_version"], created_at=timestamp())
        self.session.add(row)
        return row

    def save_revision(self, identifier, payload, expected):
        self.lock()
        design = self.design(identifier)
        match(design, expected)
        design.head_revision += 1
        design.updated_at = timestamp()
        revision = self._revision(design, payload)
        self.session.commit()
        return design, revision

    def submit(self, revision_id, frozen, versions, key, request_hash):
        self.lock()
        existing = self.idempotent(RunRow, key, request_hash)
        if existing:
            return existing, False
        self.revision(revision_id)
        queued = self.session.scalar(select(func.count()).select_from(RunRow).where(RunRow.status == "queued"))
        if queued >= 4:
            raise StoreError(429, "QUEUE_FULL", "Local queue has four waiting runs")
        row = RunRow(id=uuid4().hex, revision_id=revision_id,
                     structure_snapshot_id=frozen["sources"]["structure"]["id"],
                     status="queued", idempotency_key=key, request_hash=request_hash,
                     input_hash=content_hash(frozen), frozen_input=frozen, versions=versions,
                     queued_at=timestamp(), progress={})
        self.session.add(row)
        self.session.commit()
        return row, True

    def cancel(self, identifier):
        self.lock()
        row = self.run(identifier)
        if row.status == "queued":
            row.status, row.finished_at = "cancelled", timestamp()
        elif row.status == "running":
            row.status = "cancel_requested"
        self.session.commit()
        return row

    def claim(self, owner):
        self.lock()
        now = timestamp()
        for row in self.session.scalars(select(RunRow).where(
                RunRow.status.in_(("running", "cancel_requested")), RunRow.lease_expires_at < now)):
            row.status, row.finished_at = "interrupted", now
            row.error = {"code": "LEASE_EXPIRED", "message": "Worker lease expired; retry preserves frozen input"}
        busy = self.session.scalar(select(func.count()).select_from(RunRow).where(
            RunRow.status.in_(("running", "cancel_requested"))))
        row = None if busy else self.session.scalar(select(RunRow).where(RunRow.status == "queued").order_by(
            RunRow.queued_at, RunRow.id).limit(1))
        if row:
            row.status, row.owner, row.started_at = "running", owner, now
            row.lease_expires_at, row.attempt = timestamp(15), row.attempt + 1
        self.session.commit()
        return row

    def heartbeat(self, identifier, owner, progress):
        self.lock()
        row = self.run(identifier)
        if row.owner != owner or row.status not in ("running", "cancel_requested"):
            return False
        row.lease_expires_at = timestamp(15)
        row.progress = progress
        self.session.commit()
        return row.status == "running"

    def finish(self, identifier, owner, result=None, error=None, interrupted=False):
        self.lock()
        row = self.run(identifier)
        if row.owner != owner or row.status not in ("running", "cancel_requested"):
            return False
        if interrupted:
            row.status, row.error = "interrupted", {"code": "WORKER_STOPPED", "message": "Worker stopped"}
        elif row.status == "cancel_requested":
            row.status = "cancelled"
        elif error:
            row.status, row.error = "failed", error
        elif result:
            for rank, plan in enumerate(result["plans"], 1):
                plan_id = uuid4().hex
                details = plan.pop("details", {})
                bind_explanation_scope(plan, identifier, plan_id, row.input_hash)
                bind_explanation_scope(details, identifier, plan_id, row.input_hash)
                digest = content_hash({"summary": plan, "details": details})
                self.session.add(PlanRow(id=plan_id, run_id=identifier, candidate_key=plan["id"],
                                         rank=rank, summary=plan, result_hash=digest))
                self.session.flush()
                for kind, items in details.items():
                    for chunk in range(0, len(items), 100):
                        payload = {"items": items[chunk:chunk + 100], "total": len(items)}
                        self.session.add(DetailRow(plan_id=plan_id, kind=kind, chunk_key=chunk // 100,
                                                   payload=payload, hash=content_hash(payload)))
            row.result_manifest = {k: v for k, v in result.items() if k != "plans"}
            row.result_manifest["plan_count"] = len(result["plans"])
            row.status = result["status"]
        else:
            row.status = "failed"
            row.error = {"code": "EMPTY_RESULT", "message": "Worker returned no result"}
        row.finished_at, row.lease_expires_at = timestamp(), None
        self.session.commit()
        return True

    def delete_design(self, identifier, expected):
        self.lock()
        design = self.design(identifier)
        match(design, expected)
        active = self.session.scalar(select(func.count()).select_from(RunRow).join(
            RevisionRow, RunRow.revision_id == RevisionRow.id).where(
                RevisionRow.design_id == identifier, RunRow.status.in_(ACTIVE)))
        if active:
            raise StoreError(409, "ACTIVE_RUN", "Cancel or finish active runs before deleting this design")
        self.session.execute(delete(DesignRow).where(DesignRow.id == identifier))
        self.session.commit()

    def details(self, plan_id, kind, offset=0, limit=50):
        chunks = self.session.scalars(select(DetailRow).where(
            DetailRow.plan_id == plan_id, DetailRow.kind == kind,
            DetailRow.chunk_key >= offset // 100,
            DetailRow.chunk_key <= (offset + limit - 1) // 100).order_by(DetailRow.chunk_key)).all()
        first = self.session.get(DetailRow, (plan_id, kind, 0))
        total = first.payload["total"] if first else 0
        items = [item for chunk in chunks for item in chunk.payload["items"]]
        items = items[offset % 100:offset % 100 + limit]
        return {"items": items, "total": total,
                "next_offset": offset + len(items) if offset + len(items) < total else None}
