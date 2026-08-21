"""Reading-state models: ``ReadingProgress`` and ``ReadingEvent``.

These express *where a user is* in each document and *how they got there*:

* :class:`ReadingProgress` is the mutable, per-``(user, document)`` position of
  record — current page, status, and the ``last_summarized_page`` high-water mark
  that the progress→summary→memory loop advances (FR-3.1). It is the agent's
  relational reading-state source and the default bound for read-range retrieval.
* :class:`ReadingEvent` is an **append-only** activity trail (never updated, only
  inserted) powering analytics — pace, streaks, pages-over-time (FR-17). Keeping
  history as events rather than only a mutable current page is what makes those
  metrics derivable and auditable.

Both carry ``user_id`` for per-user isolation and cascade-delete with the user
and the document, so a removed document leaves no orphaned progress or events.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import ReadingEventType, ReadingStatus
from shared.db.base import Base
from shared.models.types import READING_EVENT_TYPE_TYPE, READING_STATUS_TYPE


class ReadingProgress(Base):
    """A user's current position and status within one document.

    Exactly one row per ``(user_id, document_id)`` (enforced by a unique
    constraint), so updating a position is an upsert on that natural key rather
    than an ever-growing table.
    """

    __tablename__ = "reading_progress"
    __table_args__ = (
        # One progress row per user per document; also the upsert conflict target.
        UniqueConstraint("user_id", "document_id", name="uq_reading_progress_user_document"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    # The user's current 1-based page. Retrieval defaults to page_end <= this.
    current_page: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Recap high-water mark: the last page already folded into a summary memory.
    # The progress→summary→memory loop summarizes last_summarized_page..current_page
    # and then advances this (FR-3.1); starts at 0 (nothing summarized yet).
    last_summarized_page: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    status: Mapped[ReadingStatus] = mapped_column(
        READING_STATUS_TYPE,
        default=ReadingStatus.NOT_STARTED,
        server_default=ReadingStatus.NOT_STARTED.value,
    )
    # Per-document spoiler-safe override: NULL means "defer to the user default".
    # Resolving the effective setting (per-doc → user default → per-query) lands
    # with the spoiler-safe retrieval work (FR-18).
    spoiler_safe: Mapped[bool | None] = mapped_column(Boolean)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Bumped on every read/update so "recently accessed" can order by it.
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReadingEvent(Base):
    """An append-only record of one reading activity (analytics source, FR-17).

    Never updated after insert — pace/streaks/history are computed by aggregating
    these rows, so the mutable :class:`ReadingProgress` row stays a cheap
    read/update path while the full trail remains auditable.
    """

    __tablename__ = "reading_events"
    __table_args__ = (
        # Analytics scans a user's events over time; this is their access path.
        Index("ix_reading_events_user_occurred", "user_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )

    type: Mapped[ReadingEventType] = mapped_column(READING_EVENT_TYPE_TYPE)
    # Page span this event moved through (both null for non-positional events like
    # a bare status change or session marker).
    from_page: Mapped[int | None] = mapped_column(Integer)
    to_page: Mapped[int | None] = mapped_column(Integer)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
