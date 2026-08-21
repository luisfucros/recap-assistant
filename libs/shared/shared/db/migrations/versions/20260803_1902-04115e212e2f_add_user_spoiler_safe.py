"""add user spoiler_safe

Revision ID: 04115e212e2f
Revises: a76b0cdc7292
Create Date: 2026-08-03 19:02:49.152834

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "04115e212e2f"
down_revision: str | Sequence[str] | None = "a76b0cdc7292"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default 'true' so existing rows adopt the spoiler-safe default rather
    # than a null; the app seeds new users from SPOILER_SAFE_DEFAULT explicitly.
    op.add_column(
        "users",
        sa.Column("spoiler_safe", sa.Boolean(), server_default="true", nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "spoiler_safe")
