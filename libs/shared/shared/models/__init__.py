"""SQLAlchemy ORM models — the single relational schema owner.

Import every model here so they register on ``Base.metadata`` (Alembic's
``env.py`` imports this package for autogenerate to see the full schema).
"""

from shared.models.conversation import Conversation, Message
from shared.models.document import Chunk, Document
from shared.models.evaluation import EvaluationRun
from shared.models.memory import LongTermMemory
from shared.models.outbox import OutboxEvent
from shared.models.reading import ReadingEvent, ReadingProgress
from shared.models.usage import UsageEvent
from shared.models.user import User

__all__ = [
    "Chunk",
    "Conversation",
    "Document",
    "EvaluationRun",
    "LongTermMemory",
    "Message",
    "OutboxEvent",
    "ReadingEvent",
    "ReadingProgress",
    "UsageEvent",
    "User",
]
