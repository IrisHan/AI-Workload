"""Workload persistence, separately created only in isolated DBs until cutover."""
from app.infrastructure.database import Base
from sqlalchemy import JSON, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column


class DesignRow(Base):
    __tablename__ = "workload_designs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    head_revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64))


class RevisionRow(Base):
    __tablename__ = "workload_design_revisions"
    __table_args__ = (UniqueConstraint("design_id", "revision_no"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    design_id: Mapped[str] = mapped_column(ForeignKey("workload_designs.id", ondelete="CASCADE"), index=True)
    revision_no: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    hardware_id: Mapped[str] = mapped_column(ForeignKey("hardware_versions.id", ondelete="RESTRICT"))
    payload: Mapped[dict] = mapped_column(JSON)
    input_hash: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[str] = mapped_column(String(40))


class RunRow(Base):
    __tablename__ = "workload_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("workload_design_revisions.id", ondelete="CASCADE"), index=True)
    structure_snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("model_structure_snapshots.id", ondelete="RESTRICT"))
    status: Mapped[str] = mapped_column(String(30), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), unique=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    frozen_input: Mapped[dict] = mapped_column(JSON)
    versions: Mapped[dict] = mapped_column(JSON)
    queued_at: Mapped[str] = mapped_column(String(40), index=True)
    started_at: Mapped[str | None] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40))
    lease_expires_at: Mapped[str | None] = mapped_column(String(40))
    owner: Mapped[str | None] = mapped_column(String(64))
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[dict | None] = mapped_column(JSON)
    result_manifest: Mapped[dict | None] = mapped_column(JSON)


class PlanRow(Base):
    __tablename__ = "workload_plans"
    __table_args__ = (UniqueConstraint("run_id", "candidate_key"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("workload_runs.id", ondelete="CASCADE"), index=True)
    candidate_key: Mapped[str] = mapped_column(String(200))
    rank: Mapped[int] = mapped_column(Integer, index=True)
    summary: Mapped[dict] = mapped_column(JSON)
    result_hash: Mapped[str] = mapped_column(String(64))


class DetailRow(Base):
    __tablename__ = "workload_plan_details"
    plan_id: Mapped[str] = mapped_column(ForeignKey("workload_plans.id", ondelete="CASCADE"), primary_key=True)
    kind: Mapped[str] = mapped_column(String(30), primary_key=True)
    chunk_key: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    hash: Mapped[str] = mapped_column(String(64))


WORKLOAD_TABLES = [row.__table__ for row in (DesignRow, RevisionRow, RunRow, PlanRow, DetailRow)]
