"""Isolation foundation for per-user data access.

The project's load-bearing invariant is **strict per-user isolation**: every
relational query for user-owned data must be filtered by ``user_id``, and that
id must come from the authenticated request context — never from a client- or
LLM-supplied argument. :class:`UserScopedRepository` centralizes that so the
filter can't be forgotten: it is bound once at construction and injected into
every query the repository builds.

The ``users`` table itself is the identity boundary and is *not* user-scoped
(see :class:`~shared.repositories.user_repository.UserRepository`); this base is
for everything a user *owns* (documents, chunks, progress, memories, …).
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core.errors import NotFoundError
from shared.db.base import Base


class UserScopedRepository[ModelT: Base]:
    """Base repository whose every query is filtered by the owner's ``user_id``.

    Subclasses set the class attribute :attr:`model` to a mapped class that has
    both an ``id`` and a ``user_id`` column. The owning ``user_id`` is supplied
    once (from the authenticated context) and applied to every read, so a caller
    can never widen the scope by passing a different id into a query method.
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession, user_id: uuid.UUID) -> None:
        """Bind the repository to a DB session and the owning user's id."""
        self._session = session
        self._user_id = user_id

    @property
    def user_id(self) -> uuid.UUID:
        """The owner id every query in this repository is filtered by."""
        return self._user_id

    def _scoped_select(self) -> Select[tuple[ModelT]]:
        """A ``SELECT`` over :attr:`model` pre-filtered to the owning user.

        Every read builds on this, so the ``user_id`` predicate is impossible to
        omit — a missing filter would be a hard bug, not an unfiltered search.
        """
        return select(self.model).where(self.model.user_id == self._user_id)

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        """Return the owned entity by id, or ``None`` if it isn't the user's."""
        result = await self._session.execute(
            self._scoped_select().where(self.model.id == entity_id)
        )
        return result.scalar_one_or_none()

    async def get_or_404(self, entity_id: uuid.UUID) -> ModelT:
        """Return the owned entity by id or raise :class:`NotFoundError`.

        Not-owned and not-existing are deliberately indistinguishable to the
        caller (both 404), so the endpoint never leaks whether another user's
        resource exists.
        """
        entity = await self.get(entity_id)
        if entity is None:
            raise NotFoundError()
        return entity

    async def list(self, *, limit: int = 100, offset: int = 0) -> Sequence[ModelT]:
        """Return a page of the user's owned entities (newest-first if ordered)."""
        result = await self._session.execute(self._scoped_select().limit(limit).offset(offset))
        return result.scalars().all()

    async def add(self, entity: ModelT) -> ModelT:
        """Persist a new owned entity and flush so generated fields populate.

        The entity's ``user_id`` must already equal :attr:`user_id`; a mismatch
        is a programming error and is rejected rather than silently trusted.
        """
        if getattr(entity, "user_id", None) != self._user_id:
            raise ValueError("entity.user_id does not match the repository's owner")
        self._session.add(entity)
        await self._session.flush()
        return entity


def ensure_owned(entity: object | None, user_id: uuid.UUID) -> None:
    """Assert an already-loaded entity belongs to ``user_id`` (service-layer guard).

    Use at the service layer as a defense-in-depth re-check on objects that were
    fetched outside a :class:`UserScopedRepository` (e.g. by natural key). Raises
    :class:`NotFoundError` when the entity is missing or owned by someone else —
    404 rather than 403 so existence never leaks.
    """
    if entity is None or getattr(entity, "user_id", None) != user_id:
        raise NotFoundError()
