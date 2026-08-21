"""Per-user data access for the usage-event cost trail (NFR-13).

Append-only, mirroring :class:`~shared.repositories.reading_repository.ReadingEventRepository`:
events are only ever inserted (via the inherited :meth:`add`) and read back for
aggregation; there is no update path.
"""

from collections.abc import Sequence
from datetime import datetime

from shared.models.usage import UsageEvent
from shared.repositories.base import UserScopedRepository


class UsageEventRepository(UserScopedRepository[UsageEvent]):
    """Owner-scoped, append-only access to the usage-event cost trail."""

    model = UsageEvent

    async def list_since(self, since: datetime, *, limit: int = 10_000) -> Sequence[UsageEvent]:
        """Return the user's usage events at or after ``since``, oldest first.

        The oldest-first order over a bounded window is what the usage
        aggregation (token totals, tool-call counts) folds over.
        """
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.occurred_at >= since)
            .order_by(self.model.occurred_at.asc())
            .limit(limit)
        )
        return result.scalars().all()


__all__ = ["UsageEventRepository"]
