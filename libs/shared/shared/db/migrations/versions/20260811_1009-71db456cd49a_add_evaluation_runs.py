"""add evaluation runs

Revision ID: 71db456cd49a
Revises: 0b071baee6b1
Create Date: 2026-08-11 10:09:35.647721

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "71db456cd49a"
down_revision: str | Sequence[str] | None = "0b071baee6b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Native Postgres enum introduced by this migration. `create_type=False` on the
# column stops `create_table` from auto-emitting CREATE TYPE; we create/drop it
# explicitly so the ordering is under our control (matching the other tables).
evaluation_run_status = postgresql.ENUM(
    "completed",
    "failed",
    name="evaluation_run_status",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    evaluation_run_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("status", evaluation_run_status, nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("llm_provider", sa.String(length=64), nullable=False),
        sa.Column("llm_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("results", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.String(length=2048), nullable=True),
        sa.Column("triggered_by", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["triggered_by"],
            ["users.id"],
            name=op.f("fk_evaluation_runs_triggered_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evaluation_runs")),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("evaluation_runs")
    evaluation_run_status.drop(op.get_bind(), checkfirst=True)
