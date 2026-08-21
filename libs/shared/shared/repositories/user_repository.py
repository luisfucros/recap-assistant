"""Data access for the ``users`` table.

The users table is the identity boundary itself, so — unlike per-user data
repositories — its queries are keyed by email / google_sub / id rather than
filtered by ``user_id``.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.models.user import User


class UserRepository:
    """Async CRUD for users, scoped to one session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return await self._session.get(User, user_id)

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_google_sub(self, google_sub: str) -> User | None:
        result = await self._session.execute(select(User).where(User.google_sub == google_sub))
        return result.scalar_one_or_none()

    async def add(self, user: User) -> User:
        """Persist a new user and flush so its generated fields are populated."""
        self._session.add(user)
        await self._session.flush()
        return user
