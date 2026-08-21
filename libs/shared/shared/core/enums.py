"""Shared domain enumerations.

Small, fixed vocabularies reused across models, APIs, and services. Keeping them
here (rather than redefining per layer) means the DB column, the Pydantic schema,
and the business logic all agree on exactly one set of allowed values.
"""

import enum


class Language(enum.StrEnum):
    """A supported content/UI language (ISO 639-1).

    Used by both ``users.preferred_language`` (the language the assistant chats
    in) and ``documents.language`` (a book's own language) — the two are
    independent. ``StrEnum`` so values serialize as the plain code (``"en"``).
    """

    EN = "en"
    ES = "es"
    DE = "de"
    FR = "fr"
    IT = "it"


class DocumentStatus(enum.StrEnum):
    """Lifecycle of an uploaded document through the ingestion pipeline.

    The API creates a document ``PENDING``; the ingestion worker moves it to
    ``PROCESSING``, then to the terminal ``INDEXED`` (chunks + vectors persisted)
    or ``FAILED`` (with a ``failure_reason``). The ``INDEXED`` transition is
    committed atomically with the chunk insert and only after the vector upsert
    succeeds, so this value never claims success for a partial result.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentFormat(enum.StrEnum):
    """Source format of an uploaded document.

    Only PDF is supported today; the enum exists so adding a format is a schema
    migration with a named value rather than a free-text column.
    """

    PDF = "pdf"


class ReadingStatus(enum.StrEnum):
    """Where a user stands in a document, tracked on ``reading_progress``.

    A row starts ``NOT_STARTED`` (or is implicitly so when absent), moves to
    ``READING`` once the user records a position, and reaches a terminal
    ``COMPLETED`` (read to the end) or ``CANCELLED`` (abandoned). This is the
    agent's relational reading-state source and drives the reading list.
    """

    NOT_STARTED = "not_started"
    READING = "reading"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MemoryType(enum.StrEnum):
    """A kind of long-term memory the assistant can store and recall.

    ``SUMMARY`` memories are keyed to a document + page range (the recap loop);
    the rest capture durable facts about the user: their ``PREFERENCE``s, learned
    ``CONCEPT``s and ``FACT``s, reading ``HABIT``s, and answered ``FAQ``s. Kept
    here as one shared vocabulary so the memory model, the classifier schema, and
    retrieval all agree on the allowed values.
    """

    PREFERENCE = "preference"
    SUMMARY = "summary"
    CONCEPT = "concept"
    FACT = "fact"
    HABIT = "habit"
    FAQ = "faq"


class MessageRole(enum.StrEnum):
    """The author of a persisted chat :class:`~shared.models.conversation.Message`.

    These are the canonical chat roles. The product-facing transcript writes
    ``USER`` (what the reader sent, normalized to text) and ``ASSISTANT`` (the
    agent's answer, or a guardrail refusal); ``SYSTEM`` and ``TOOL`` are part of
    the vocabulary so a fuller transcript can be represented without a schema
    change. The agent's *internal* message graph (tool-call/observation loop) is
    owned by the LangGraph checkpointer, not this table — see
    :class:`~shared.models.conversation.Message`.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ScratchpadKind(enum.StrEnum):
    """A kind of entry in the agent's turn-scoped scratchpad (FR-7.8).

    ``PLAN`` is the turn's plan (always recalled — it frames the turn); ``FINDING``
    captures something a tool step turned up; ``QUESTION`` an open question the
    agent is still resolving. Findings/questions are recalled only when relevant.
    """

    PLAN = "plan"
    FINDING = "finding"
    QUESTION = "question"


class ReadingEventType(enum.StrEnum):
    """A kind of entry in the append-only ``reading_events`` analytics trail.

    The trail is never updated, only inserted, so pace/streaks/history stay
    derivable and auditable (FR-17): ``POSITION_ADVANCED`` on a forward move,
    ``STATUS_CHANGED`` on a status transition, ``SESSION`` to mark a reading
    session, and ``COMPLETED`` when a document is finished.
    """

    POSITION_ADVANCED = "position_advanced"
    STATUS_CHANGED = "status_changed"
    SESSION = "session"
    COMPLETED = "completed"


class EvaluationRunStatus(enum.StrEnum):
    """Terminal outcome of one :class:`~shared.models.evaluation.EvaluationRun`.

    A run is always persisted with a terminal status — there is no ``PENDING``/
    ``RUNNING`` value because ``EvaluationService`` runs a dataset to completion
    (or failure) synchronously within the triggering request/CLI call rather
    than as a background job.
    """

    COMPLETED = "completed"
    FAILED = "failed"


class UsageEventType(enum.StrEnum):
    """A kind of entry in the append-only ``usage_events`` per-user cost trail.

    Durable, per-user counterpart to the low-cardinality Prometheus metrics
    (``recap_llm_tokens_total``, ``recap_operation_seconds``): those can never
    carry a ``user_id`` label without exploding the time-series count (NFR-13),
    so ``TOKEN_USAGE`` (an answer-model LLM call's prompt/completion tokens) and
    ``TOOL_CALL`` (one executed tool call, by name) are recorded here instead,
    keyed to a user the same way ``reading_events`` is.
    """

    TOKEN_USAGE = "token_usage"
    TOOL_CALL = "tool_call"
