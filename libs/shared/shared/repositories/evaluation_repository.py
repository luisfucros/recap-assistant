"""Data access for persisted evaluation runs.

Like the outbox, this is an infrastructure/system table rather than per-user
data — an evaluation run belongs to the app, not to the admin who triggered it
(see :class:`~shared.models.evaluation.EvaluationRun`) — so this is deliberately
not a :class:`~shared.repositories.base.UserScopedRepository` subject.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.errors import NotFoundError
from shared.models.evaluation import EvaluationRun


class EvaluationRunRepository:
    """Persist and fetch evaluation runs, scoped to one DB session."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to a DB session."""
        self._session = session

    async def add(self, run: EvaluationRun) -> EvaluationRun:
        """Persist a new run and flush so generated fields (e.g. ``id``) populate."""
        self._session.add(run)
        await self._session.flush()
        return run

    async def get(self, run_id: uuid.UUID) -> EvaluationRun | None:
        """Return a run by id, or ``None`` if it doesn't exist."""
        result = await self._session.execute(
            select(EvaluationRun).where(EvaluationRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def get_or_404(self, run_id: uuid.UUID) -> EvaluationRun:
        """Return a run by id or raise :class:`NotFoundError`."""
        run = await self.get(run_id)
        if run is None:
            raise NotFoundError()
        return run

    async def list_recent(self, *, limit: int = 10, offset: int = 0) -> Sequence[EvaluationRun]:
        """Return a page of runs, newest first (admin list)."""
        result = await self._session.execute(
            select(EvaluationRun)
            .order_by(EvaluationRun.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count(self) -> int:
        """Return the total number of evaluation runs (for pagination)."""
        result = await self._session.execute(select(func.count()).select_from(EvaluationRun))
        return int(result.scalar_one())
