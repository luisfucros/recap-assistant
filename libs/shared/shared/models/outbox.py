"""The transactional-outbox table driving cross-service async messaging.

The API writes a domain event (e.g. ``document.uploaded``) into this table **in
the same transaction** as the state change that produced it, rather than
publishing to the broker directly. This removes the dual-write race between the
Postgres commit and the queue publish: either both the state change and the
event are durable, or neither is. A relay (Celery beat, in the ingestion
service) polls unprocessed rows, enqueues a task per event, and stamps
``processed_at``. Delivery is at-least-once, so consumers must be idempotent.

This is an infrastructure table, not user-owned data — it is intentionally *not*
a :class:`~shared.repositories.base.UserScopedRepository` subject. Any user
scoping a consumer needs lives inside the event ``payload``.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base


class OutboxEvent(Base):
    """A durable domain event awaiting relay to the message broker."""

    __tablename__ = "outbox"
    __table_args__ = (
        # The relay polls the unprocessed backlog oldest-first; a partial index
        # keeps that scan cheap and shrinks as events are marked processed.
        Index(
            "ix_outbox_unprocessed",
            "created_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Dotted event name, e.g. "document.uploaded" — routes the relay's dispatch.
    event_type: Mapped[str] = mapped_column(String(255))
    # Id of the aggregate the event is about (e.g. the document id).
    aggregate_id: Mapped[uuid.UUID] = mapped_column()
    # Event body the consumer needs (includes user_id for scoping downstream).
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Null until the relay has enqueued this event; set once it is delivered.
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Relay delivery attempts, for observability and dead-lettering.
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
