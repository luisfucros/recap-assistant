"""add document updated_at

Revision ID: 7954563772ac
Revises: c51be0bcaffb
Create Date: 2026-08-12 13:23:38.917169

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7954563772ac"
down_revision: str | Sequence[str] | None = "c51be0bcaffb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Autogenerate also proposed dropping the LangGraph checkpointer's tables
# (checkpoints/checkpoint_blobs/checkpoint_writes/checkpoint_migrations) — a
# pre-existing, documented false positive, since those are managed by
# `python -m api.checkpointer`, never by Alembic. Only the real schema change,
# `documents.updated_at`, is kept here.


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "documents",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "updated_at")
