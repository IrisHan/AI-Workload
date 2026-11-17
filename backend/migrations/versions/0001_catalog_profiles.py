"""Immutable catalog versions, workloads and analysis snapshots."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("repo_id", sa.String(200), nullable=False),
        sa.Column("revision", sa.String(40), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.UniqueConstraint("repo_id", "revision"),
    )
    op.create_table(
        "hardware_versions",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_table(
        "workloads",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("model_id", sa.String(100), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("hardware_id", sa.String(100), sa.ForeignKey("hardware_versions.id"), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_workloads_model_id", "workloads", ["model_id"])
    op.create_index("ix_workloads_hardware_id", "workloads", ["hardware_id"])
    op.create_table(
        "workload_profiles",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("workload_id", sa.String(100), sa.ForeignKey("workloads.id"), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_workload_profiles_workload_id", "workload_profiles", ["workload_id"])


def downgrade():
    op.drop_table("workload_profiles")
    op.drop_table("workloads")
    op.drop_table("hardware_versions")
    op.drop_table("model_versions")
