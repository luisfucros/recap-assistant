"""Data access for persisted evaluation runs.

Like the outbox, this is an infrastructure/system table rather than per-user
data — an evaluation run belongs to the app, not to the admin who triggered it
(see :class:`~shared.models.evaluation.EvaluationRun`) — so this is deliberately
not a :class:`~shared.repositories.base.UserScopedRepository` subject.
"""

import uuid

from sqlalchemy import select
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
