"""``UsageEvent``: the append-only per-user cost trail (NFR-13).

Prometheus's ``recap_llm_tokens_total``/``recap_operation_seconds`` (the SLI
layer) deliberately never carry a ``user_id`` label — doing so would give one
time series per user and blow up cardinality. Per-user token spend and
tool-call counts are durable, low-volume, and queried by aggregation rather
than by dashboard, so they belong in Postgres instead, mirroring the
``reading_events`` append-only trail that already powers per-user analytics
(FR-17): never updated, only inserted and aggregated over.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import UsageEventType
from shared.db.base import Base
from shared.models.types import USAGE_EVENT_TYPE_TYPE


class UsageEvent(Base):
    """One unit of per-user cost: an LLM call's token counts, or one tool call.

    ``prompt_tokens``/``completion_tokens`` are set only for ``TOKEN_USAGE``;
    ``tool_name`` only for ``TOOL_CALL`` — the same nullable-by-kind shape
    ``ReadingEvent`` uses for its own two event kinds' page fields.
    """

    __tablename__ = "usage_events"
    __table_args__ = (
        # Aggregation scans a user's events over a trailing window; this is
        # that access path (mirrors ix_reading_events_user_occurred).
        Index("ix_usage_events_user_occurred", "user_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    type: Mapped[UsageEventType] = mapped_column(USAGE_EVENT_TYPE_TYPE)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    tool_name: Mapped[str | None] = mapped_column(String(100))

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
