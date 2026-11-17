"""Versioned model structure analysis; existing profiles remain unchanged."""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "model_structure_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_id", sa.String(100), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index("ix_model_structure_snapshots_model_id", "model_structure_snapshots", ["model_id"])


def downgrade():
    op.drop_table("model_structure_snapshots")
