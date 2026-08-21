"""add user is_admin

Revision ID: 0b071baee6b1
Revises: e46ef33e43ae
Create Date: 2026-08-11 09:52:21.756058

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0b071baee6b1"
down_revision: str | Sequence[str] | None = "e46ef33e43ae"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # server_default 'false' so existing rows adopt the non-admin default rather
    # than a null; no route ever sets this — it's flipped directly in the database.
    op.add_column(
        "users",
        sa.Column("is_admin", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "is_admin")
