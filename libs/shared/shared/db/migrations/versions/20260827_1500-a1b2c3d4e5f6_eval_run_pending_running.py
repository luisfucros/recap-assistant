"""eval run pending/running statuses + updated_at

Revision ID: a1b2c3d4e5f6
Revises: d0040f40180f
Create Date: 2026-08-27 15:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "d0040f40180f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add in-progress enum values and an ``updated_at`` column for the stuck sweep."""
    op.execute(sa.text("ALTER TYPE evaluation_run_status ADD VALUE IF NOT EXISTS 'pending'"))
    op.execute(sa.text("ALTER TYPE evaluation_run_status ADD VALUE IF NOT EXISTS 'running'"))
    op.add_column(
        "evaluation_runs",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop ``updated_at``. Postgres cannot cheaply remove enum values; leave them."""
    op.drop_column("evaluation_runs", "updated_at")
