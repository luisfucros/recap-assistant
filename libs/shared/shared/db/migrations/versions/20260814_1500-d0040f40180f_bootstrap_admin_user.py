"""bootstrap admin user

Revision ID: d0040f40180f
Revises: 7954563772ac
Create Date: 2026-08-14 15:00:00.000000

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from shared.core.config import get_settings
from shared.core.passwords import build_password_hash

# revision identifiers, used by Alembic.
revision: str = "d0040f40180f"
down_revision: str | Sequence[str] | None = "7954563772ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Fixed, distinct from any other namespace this codebase derives deterministic
# ids from (e.g. the evaluation system user's), so the bootstrap admin's id is
# stable across re-runs of this migration without colliding with anything else.
_NAMESPACE = uuid.UUID("2f6a0b8e-2f7f-4d1a-9c3e-8b6a1d4e7f20")

# A lightweight snapshot of just the columns this migration touches — migrations
# don't import the live ORM model, so future changes to `User` never retroactively
# break this one.
_users = sa.table(
    "users",
    sa.column("id", sa.Uuid()),
    sa.column("email", sa.String()),
    sa.column("hashed_password", sa.String()),
    sa.column("display_name", sa.String()),
    sa.column("is_admin", sa.Boolean()),
)


def upgrade() -> None:
    """Seed one admin user from ``INITIAL_ADMIN_EMAIL``/``INITIAL_ADMIN_PASSWORD``.

    A silent no-op when either is unset (opt-in), or when a user already owns
    that email (idempotent — safe to run again with the same ``.env``).
    """
    settings = get_settings()
    email = settings.initial_admin_email
    password = settings.initial_admin_password
    if not email or not password:
        return

    bind = op.get_bind()
    exists = bind.execute(sa.select(_users.c.id).where(_users.c.email == email)).first()
    if exists is not None:
        return

    bind.execute(
        _users.insert().values(
            id=uuid.uuid5(_NAMESPACE, email),
            email=email,
            hashed_password=build_password_hash().hash(password.get_secret_value()),
            display_name="Admin",
            is_admin=True,
        )
    )


def downgrade() -> None:
    """Remove exactly the row this migration created, if any.

    Targets the deterministic id derived from the configured email (plus a
    matching ``is_admin`` flag) so a coincidentally-reused id or email never
    causes an unrelated admin to be deleted. A no-op if the email isn't
    configured at downgrade time.
    """
    settings = get_settings()
    email = settings.initial_admin_email
    if not email:
        return

    bind = op.get_bind()
    bind.execute(
        _users.delete().where(
            _users.c.id == uuid.uuid5(_NAMESPACE, email),
            _users.c.email == email,
            _users.c.is_admin.is_(True),
        )
    )
