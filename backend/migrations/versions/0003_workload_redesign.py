"""Replace legacy profiles with immutable inference designs and durable runs."""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "workload_designs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("head_revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False, unique=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
    )
    op.create_index("ix_workload_designs_created_at", "workload_designs", ["created_at"])
    op.create_table(
        "workload_design_revisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("design_id", sa.String(64), sa.ForeignKey("workload_designs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("revision_no", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(100), sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
                  nullable=False),
        sa.Column("hardware_id", sa.String(100), sa.ForeignKey("hardware_versions.id", ondelete="RESTRICT"),
                  nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(50), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("design_id", "revision_no"),
    )
    op.create_index("ix_workload_design_revisions_design_id", "workload_design_revisions", ["design_id"])
    op.create_table(
        "workload_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("revision_id", sa.String(64),
                  sa.ForeignKey("workload_design_revisions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("structure_snapshot_id", sa.String(64),
                  sa.ForeignKey("model_structure_snapshots.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False, unique=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("frozen_input", sa.JSON(), nullable=False),
        sa.Column("versions", sa.JSON(), nullable=False),
        sa.Column("queued_at", sa.String(40), nullable=False),
        sa.Column("started_at", sa.String(40)),
        sa.Column("finished_at", sa.String(40)),
        sa.Column("lease_expires_at", sa.String(40)),
        sa.Column("owner", sa.String(64)),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("progress", sa.JSON(), nullable=False),
        sa.Column("error", sa.JSON()),
        sa.Column("result_manifest", sa.JSON()),
    )
    for column in ("revision_id", "status", "queued_at"):
        op.create_index(f"ix_workload_runs_{column}", "workload_runs", [column])
    op.create_table(
        "workload_plans",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("workload_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("candidate_key", sa.String(200), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("result_hash", sa.String(64), nullable=False),
        sa.UniqueConstraint("run_id", "candidate_key"),
    )
    for column in ("run_id", "rank"):
        op.create_index(f"ix_workload_plans_{column}", "workload_plans", [column])
    op.create_table(
        "workload_plan_details",
        sa.Column("plan_id", sa.String(64), sa.ForeignKey("workload_plans.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("kind", sa.String(30), primary_key=True),
        sa.Column("chunk_key", sa.Integer(), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
    )
    op.drop_table("workload_profiles")
    op.drop_table("workloads")


def downgrade():
    raise RuntimeError("Workload history was intentionally removed; restore source-only data into a fresh schema")
