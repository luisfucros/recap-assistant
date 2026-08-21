"""Shared SQLAlchemy column types for the ORM models.

Native Postgres ``ENUM`` types are declared here as **single, shared objects**
so a given enum (e.g. ``language``) maps to exactly one Postgres type even when
several tables use it. Defining a second ``Enum(..., name="language")`` object
would make SQLAlchemy try to ``CREATE TYPE language`` twice; importing the one
object below avoids that. ``values_callable`` stores each member's *value*
(``"en"``) rather than its Python name (``"EN"``).
"""

from sqlalchemy import Enum as SAEnum

from shared.core.enums import (
    DocumentFormat,
    DocumentStatus,
    EvaluationRunStatus,
    Language,
    MemoryType,
    MessageRole,
    ReadingEventType,
    ReadingStatus,
    UsageEventType,
)


def _pg_enum[E](enum_cls: type[E], name: str) -> SAEnum:
    """Build a native Postgres enum type storing member values, not names."""
    return SAEnum(enum_cls, name=name, values_callable=lambda e: [m.value for m in e])


# One shared object per enum — import these into models rather than re-declaring.
LANGUAGE_TYPE = _pg_enum(Language, "language")
DOCUMENT_STATUS_TYPE = _pg_enum(DocumentStatus, "document_status")
DOCUMENT_FORMAT_TYPE = _pg_enum(DocumentFormat, "document_format")
READING_STATUS_TYPE = _pg_enum(ReadingStatus, "reading_status")
READING_EVENT_TYPE_TYPE = _pg_enum(ReadingEventType, "reading_event_type")
MESSAGE_ROLE_TYPE = _pg_enum(MessageRole, "message_role")
MEMORY_TYPE_TYPE = _pg_enum(MemoryType, "memory_type")
EVALUATION_RUN_STATUS_TYPE = _pg_enum(EvaluationRunStatus, "evaluation_run_status")
USAGE_EVENT_TYPE_TYPE = _pg_enum(UsageEventType, "usage_event_type")
