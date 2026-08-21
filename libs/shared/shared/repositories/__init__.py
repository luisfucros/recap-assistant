"""Data-access repositories; every user-owned query is scoped by ``user_id``."""

from shared.repositories.base import UserScopedRepository, ensure_owned
from shared.repositories.conversation_repository import (
    ConversationRepository,
    MessageRepository,
)
from shared.repositories.document_repository import ChunkRepository, DocumentRepository
from shared.repositories.evaluation_repository import EvaluationRunRepository
from shared.repositories.memory_repository import LongTermMemoryRepository
from shared.repositories.outbox_repository import OutboxRepository
from shared.repositories.reading_repository import (
    ReadingEventRepository,
    ReadingProgressRepository,
)
from shared.repositories.usage_repository import UsageEventRepository
from shared.repositories.user_repository import UserRepository

__all__ = [
    "ChunkRepository",
    "ConversationRepository",
    "DocumentRepository",
    "EvaluationRunRepository",
    "LongTermMemoryRepository",
    "MessageRepository",
    "OutboxRepository",
    "ReadingEventRepository",
    "ReadingProgressRepository",
    "UsageEventRepository",
    "UserRepository",
    "UserScopedRepository",
    "ensure_owned",
]
