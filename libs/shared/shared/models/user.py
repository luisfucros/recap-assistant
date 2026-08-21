"""The ``User`` model — the identity all per-user data is scoped to.

Password login and Google OAuth both map to one user row: ``hashed_password`` is
null for OAuth-only accounts, ``google_sub`` is null for password-only ones.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import Language
from shared.db.base import Base
from shared.models.types import LANGUAGE_TYPE


class User(Base):
    """A registered user."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    # Null for OAuth-only accounts (no local password).
    hashed_password: Mapped[str | None] = mapped_column(String(255))
    # Google's stable subject id; null for password-only accounts.
    google_sub: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    # The language the assistant chats in; independent of any document's language.
    preferred_language: Mapped[Language] = mapped_column(
        LANGUAGE_TYPE, default=Language.EN, server_default=Language.EN.value
    )
    # Global spoiler-safe default (FR-18). When true, retrieval/summaries are
    # hard-bounded to already-read pages; a per-document override lives on
    # ``reading_progress.spoiler_safe`` and a per-query override wins over both.
    spoiler_safe: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    # Grants access to admin-only routes (e.g. POST/GET /evaluations). No API path
    # sets this — it's flipped directly in the database, deliberately out of any
    # user-facing surface so a compromised account can never self-promote.
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
