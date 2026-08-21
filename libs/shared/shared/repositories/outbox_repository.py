"""Data access for the transactional outbox.

Unlike the per-user repositories, the outbox is an infrastructure table: the API
writes an event here inside the same transaction as the state change, and the
ingestion service's relay drains it. It is deliberately *not* user-scoped — any
scoping a consumer needs travels inside the event ``payload``.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.models.outbox import OutboxEvent


class OutboxRepository:
    """Append events and drain the unprocessed backlog, scoped to one session."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a DB session."""
        self._session = session

    async def add(
        self, *, event_type: str, aggregate_id: uuid.UUID, payload: dict[str, object]
    ) -> OutboxEvent:
        """Stage an event for relay (commit it with the state change it describes)."""
        event = OutboxEvent(event_type=event_type, aggregate_id=aggregate_id, payload=dict(payload))
        self._session.add(event)
        await self._session.flush()
        return event

    async def count_unprocessed(self) -> int:
        """Return the number of unprocessed events (the relay's queue depth)."""
        result = await self._session.execute(
            select(func.count()).select_from(OutboxEvent).where(OutboxEvent.processed_at.is_(None))
        )
        return int(result.scalar_one())

    async def fetch_unprocessed(self, *, limit: int = 100) -> Sequence[OutboxEvent]:
        """Return the oldest unprocessed events (relay reads these each tick)."""
        result = await self._session.execute(
            select(OutboxEvent)
            .where(OutboxEvent.processed_at.is_(None))
            .order_by(OutboxEvent.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()

    async def mark_processed(self, event_id: uuid.UUID) -> None:
        """Stamp an event delivered so the relay won't re-enqueue it."""
        await self._session.execute(
            update(OutboxEvent)
            .where(OutboxEvent.id == event_id)
            .values(processed_at=func.now(), attempts=OutboxEvent.attempts + 1)
        )
