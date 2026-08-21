"""add user preferred_language

Revision ID: 9b8aa5234228
Revises: 6ce405587cb5
Create Date: 2026-07-21 13:33:22.206284

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9b8aa5234228"
down_revision: str | Sequence[str] | None = "6ce405587cb5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The native PG enum backing users.preferred_language. `create_type=False` so the
# column DDL doesn't implicitly emit CREATE TYPE — we create/drop it explicitly
# (op.add_column won't create the type on its own; a bare Enum would then fail).
language_enum = postgresql.ENUM("en", "es", "de", "fr", "it", name="language", create_type=False)


def upgrade() -> None:
    """Upgrade schema."""
    language_enum.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "users",
        sa.Column("preferred_language", language_enum, server_default="en", nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "preferred_language")
    language_enum.drop(op.get_bind(), checkfirst=True)
