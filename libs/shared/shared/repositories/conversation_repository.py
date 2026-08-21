"""Per-user data access for chat conversations and their messages.

Both repositories are :class:`~shared.repositories.base.UserScopedRepository`
subjects: every query is filtered by the owning ``user_id`` bound at
construction, so a caller can never read or widen another user's chat history.
The owner id always comes from the authenticated context, never from a client-
or LLM-supplied argument. Messages carry their own ``user_id`` (denormalized from
the conversation), so even a message query scoped by ``conversation_id`` is *also*
owner-filtered — a conversation id belonging to another user yields nothing.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select

from shared.models.conversation import Conversation, Message
from shared.repositories.base import UserScopedRepository


class ConversationRepository(UserScopedRepository[Conversation]):
    """Owner-scoped access to :class:`~shared.models.conversation.Conversation`."""

    model = Conversation

    async def list_recent(self, *, limit: int = 10, offset: int = 0) -> Sequence[Conversation]:
        """Return the user's conversations, most-recently-active first.

        Ordered by ``updated_at`` (bumped on each turn), matching the
        ``ix_conversations_user_updated`` index — the conversation-list access path.
        """
        result = await self._session.execute(
            self._scoped_select().order_by(self.model.updated_at.desc()).limit(limit).offset(offset)
        )
        return result.scalars().all()

    async def count(self) -> int:
        """Return the exact number of the user's conversations (for pagination)."""
        result = await self._session.execute(
            select(func.count()).select_from(self._scoped_select().subquery())
        )
        return int(result.scalar_one())

    async def delete(self, conversation: Conversation) -> None:
        """Delete an owned conversation row; its messages go via the DB FK cascade.

        The caller must have loaded ``conversation`` through this repository (so it
        is owner-scoped). Message rows are removed by the ``ON DELETE CASCADE``
        foreign key, not the ORM, mirroring how a document's chunks are dropped.
        """
        await self._session.delete(conversation)
        await self._session.flush()


class MessageRepository(UserScopedRepository[Message]):
    """Owner-scoped, append-only access to a conversation's messages."""

    model = Message

    async def list_by_conversation(
        self, conversation_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> Sequence[Message]:
        """Return a conversation's messages in chronological order (oldest first).

        Scoped by both ``user_id`` (via :meth:`_scoped_select`) and
        ``conversation_id``, so another user's conversation id can never surface
        messages. Oldest-first is the order a transcript renders in.
        """
        result = await self._session.execute(
            self._scoped_select()
            .where(self.model.conversation_id == conversation_id)
            .order_by(self.model.created_at.asc(), self.model.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def count_by_conversation(self, conversation_id: uuid.UUID) -> int:
        """Return the exact number of messages in the user's conversation."""
        result = await self._session.execute(
            select(func.count()).select_from(
                self._scoped_select()
                .where(self.model.conversation_id == conversation_id)
                .subquery()
            )
        )
        return int(result.scalar_one())


__all__ = ["ConversationRepository", "MessageRepository"]
