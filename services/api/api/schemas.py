"""Pydantic request/response models for the HTTP layer.

These are the API's external contract — every request body is validated here
before it reaches a service, and responses are shaped here so internal ORM
fields (e.g. ``hashed_password``) can never leak. Email is normalized and given
a minimal shape check rather than depending on a full email-validation library.
"""

import re
import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.core.enums import (
    DocumentFormat,
    DocumentStatus,
    EvaluationRunStatus,
    Language,
    MemoryType,
    MessageRole,
    ReadingStatus,
)

# Pragmatic "something@something.tld" check — not RFC-complete, just enough to
# reject obvious garbage without pulling in an email-validation dependency.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Keep in sync with the security policy; argon2 handles long passwords fine.
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128


class _EmailIn(BaseModel):
    """Mixin: a normalized, minimally-validated email field."""

    email: str = Field(max_length=320, default="user@example.com")

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _EMAIL_RE.match(normalized):
            raise ValueError("must be a valid email address")
        return normalized


class RegisterRequest(_EmailIn):
    """Payload to create an email/password account."""

    password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    display_name: str | None = Field(default=None, max_length=255)


class LoginRequest(_EmailIn):
    """Payload to authenticate with email + password."""

    password: str = Field(min_length=1, max_length=_PASSWORD_MAX)


class AdminCreateUserRequest(_EmailIn):
    """Payload for an admin to directly create any user (regular or admin)."""

    password: str = Field(min_length=_PASSWORD_MIN, max_length=_PASSWORD_MAX)
    display_name: str | None = Field(default=None, max_length=255)
    is_admin: bool = False


class UserPublic(BaseModel):
    """Public view of a user — never includes the password hash or auth linkage."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str | None
    preferred_language: Language
    spoiler_safe: bool
    is_admin: bool


class UpdateMeRequest(BaseModel):
    """Partial update of the current user's profile (only provided fields change)."""

    display_name: str | None = Field(default=None, max_length=255)
    preferred_language: Language | None = None
    spoiler_safe: bool | None = None


class DocumentPublic(BaseModel):
    """Public view of a document — safe to return over the API.

    Excludes internal fields (``object_key``, ``content_sha256``, ``embed_model``,
    ``user_id``) that are storage/ownership details, not client concerns.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    title: str | None
    author: str | None
    format: DocumentFormat
    language: Language | None
    status: DocumentStatus
    failure_reason: str | None
    page_count: int | None
    created_at: datetime
    indexed_at: datetime | None


class DocumentPage(BaseModel):
    """A page of the user's documents (standard list envelope)."""

    items: list[DocumentPublic]
    total: int
    page: int
    page_size: int


class UpdateDocumentRequest(BaseModel):
    """Partial update of a document's user-editable metadata.

    Currently only the detected ``language`` can be overridden (e.g. when
    detection guessed wrong); only provided fields change.
    """

    language: Language | None = None


class ReadingProgressPublic(BaseModel):
    """Public view of a user's reading state for one document."""

    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    current_page: int
    last_summarized_page: int
    status: ReadingStatus
    # Per-document spoiler-safe override; null means "defer to the user default".
    spoiler_safe: bool | None
    last_accessed_at: datetime


class UpdateProgressRequest(BaseModel):
    """Partial update of reading state for a document (only provided fields change).

    ``current_page`` moves the position (auto-promoting status to *reading* /
    *completed*); ``status`` overrides it (e.g. cancel/reopen); ``spoiler_safe``
    sets the per-document override, where an explicit ``null`` clears it back to
    the user default.
    """

    current_page: int | None = Field(default=None, ge=0)
    status: ReadingStatus | None = None
    spoiler_safe: bool | None = None


class ReadingListResponse(BaseModel):
    """The user's tracked documents grouped by reading status (most-recent first)."""

    reading: list[ReadingProgressPublic]
    completed: list[ReadingProgressPublic]
    cancelled: list[ReadingProgressPublic]


class PagesOnDay(BaseModel):
    """Pages read on a single calendar day (a point in the pages-over-time series)."""

    day: date
    pages: int


class AnalyticsSummary(BaseModel):
    """A user's reading analytics over a trailing window (FR-17).

    Derived from the append-only reading-event trail plus current status counts;
    computed server-side, cached, and scoped to the requesting user.
    """

    window_days: int
    pages_read: int
    active_days: int
    pace_pages_per_day: float
    current_streak_days: int
    longest_streak_days: int
    documents_started: int
    documents_completed: int
    documents_cancelled: int
    pages_over_time: list[PagesOnDay]


class ToolCallCount(BaseModel):
    """How many times one tool was called (a breakdown line in usage)."""

    tool: str
    count: int


class UsageSummary(BaseModel):
    """A user's LLM token spend and tool-call counts over a trailing window (NFR-13).

    Derived from the append-only usage-event trail; computed server-side,
    cached, and scoped to the requesting user — the durable, per-user
    counterpart to the low-cardinality Prometheus metrics (which never carry a
    ``user_id`` label).
    """

    window_days: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: int
    tool_calls_by_tool: list[ToolCallCount]


# Accepted attachment media types (FR-19), grouped so the ``kind`` and the
# ``mime_type`` are cross-checked. Kept deliberately narrow (an allowlist) — a
# reading assistant only needs common voice-note and image formats, and a tight
# list is one less thing an attacker can probe. Precise byte-size limits are
# applied against ``Settings`` at the route boundary; the coarse ceiling here just
# bounds a single request's memory before settings are consulted.
_AUDIO_MIMES = frozenset(
    {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/webm",
        "audio/ogg",
    }
)
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_MEDIA_HARD_MAX_BYTES = 25 * 1024 * 1024
_MAX_PARTS = 8


class ChatMediaPart(BaseModel):
    """One non-text attachment in a chat turn (FR-19), base64-encoded on the wire.

    ``kind`` and ``mime_type`` are cross-validated so an ``audio`` part can't smuggle
    an image type (or vice versa); ``data`` is decoded from base64 during validation
    (invalid base64 ⇒ 422). The originals are archived and normalized to text by the
    agent's ``normalize_input`` step before anything reasons over them.
    """

    kind: Literal["audio", "image"]
    mime_type: str = Field(max_length=100)
    data: Base64Bytes

    @model_validator(mode="after")
    def _check_media(self) -> "ChatMediaPart":
        allowed = _AUDIO_MIMES if self.kind == "audio" else _IMAGE_MIMES
        if self.mime_type.lower() not in allowed:
            raise ValueError(f"unsupported {self.kind} mime type: {self.mime_type}")
        if not self.data:
            raise ValueError("attachment is empty")
        if len(self.data) > _MEDIA_HARD_MAX_BYTES:
            raise ValueError("attachment exceeds the maximum allowed size")
        return self


class ChatRequest(BaseModel):
    """A chat turn: the user's message, any attachments, and the thread to continue.

    Supply ``message`` (typed text), ``parts`` (audio/image attachments, FR-19), or
    both — at least one is required. Omit ``conversation_id`` to start a new
    conversation; the response/stream reports the id so the client can continue it.
    """

    message: str = Field(default="", max_length=8000)
    parts: list[ChatMediaPart] = Field(default_factory=list, max_length=_MAX_PARTS)
    conversation_id: uuid.UUID | None = Field(default=None)

    @model_validator(mode="after")
    def _require_content(self) -> "ChatRequest":
        if not self.message.strip() and not self.parts:
            raise ValueError("provide a message or at least one attachment")
        return self


class ToolStepPublic(BaseModel):
    """One tool the agent called this turn and the observation it returned."""

    name: str
    args: dict
    result: str


class ChatResponse(BaseModel):
    """The non-streamed result of a chat turn.

    ``blocked`` marks a guardrail refusal (``answer`` holds the polite reason and
    ``tool_steps`` is empty); ``interrupt`` is set instead when the turn paused
    for the user (HITL) — ``answer``/``tool_steps`` are then empty and the turn
    continues via ``POST /chat/{conversation_id}/resume``. Its shape varies by
    the ``kind`` key it always carries: ``tool_approval`` (``tool_call``,
    ``reason``), ``page_range_confirm`` (``document_id``, ``document_title``,
    ``proposal``), ``ask_pages_read`` (``document_id``, ``document_title``,
    ``reason``), or ``spoiler_warning`` (``document_id``, ``document_title``,
    ``current_page``, ``reason``). Otherwise ``answer`` is the grounded reply.
    ``trace_id`` correlates the turn to its trace when tracing is enabled
    (``null`` when Langfuse is not configured).
    """

    conversation_id: uuid.UUID
    answer: str
    blocked: bool
    tool_steps: list[ToolStepPublic]
    trace_id: str | None = None
    interrupt: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    """The user's answer/decision for whatever the paused turn is asking (HITL).

    Which fields matter depends on the pending interrupt's ``kind`` (see
    :class:`ChatResponse`): a ``tool_approval`` uses ``decision`` (``args`` too
    when editing); a ``page_range_confirm`` uses ``decision`` (``page_start``/
    ``page_end`` too when editing); an ``ask_pages_read`` answers with just
    ``page_start``/``page_end`` (no ``decision``); a ``spoiler_warning`` uses
    ``decision`` (``approve`` reveals the flagged answer, anything else keeps it
    withheld). Fields irrelevant to the pending interrupt are ignored — the
    paused node reads only what it expects.
    """

    decision: Literal["approve", "deny", "edit"] | None = Field(default=None)
    args: dict[str, Any] | None = Field(default=None)
    page_start: int | None = Field(default=None)
    page_end: int | None = Field(default=None)

    @model_validator(mode="after")
    def _validate(self) -> "ResumeRequest":
        # An all-``None`` payload serializes to `{}`, which LangGraph's
        # ``Command(resume=...)`` treats as "no resume value supplied" — it
        # re-interrupts with the identical pending payload instead of raising,
        # which would look like the request silently did nothing. Rejecting it
        # here gives a clear 422 instead of that confusing no-op.
        if (
            self.decision is None
            and self.args is None
            and self.page_start is None
            and self.page_end is None
        ):
            raise ValueError("provide a decision, args, or a page range to resume with")
        if self.decision == "edit" and self.args is None and self.page_start is None:
            raise ValueError("'edit' requires args (tool approval) or a page range to edit")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be >= page_start")
        return self


class ConversationPublic(BaseModel):
    """A chat thread in the conversation list."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationPage(BaseModel):
    """A page of the user's conversations (standard list envelope)."""

    items: list[ConversationPublic]
    total: int
    page: int
    page_size: int


class MessagePublic(BaseModel):
    """One persisted message in a conversation's transcript."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    tool_calls: dict | None
    created_at: datetime


class MessagePage(BaseModel):
    """A page of a conversation's messages (chronological; standard envelope)."""

    items: list[MessagePublic]
    total: int
    page: int
    page_size: int


class MemoryPublic(BaseModel):
    """Public view of a stored long-term memory (the privacy view/delete surface, FR-4.5).

    Excludes ``user_id`` (implicit — always the caller's own) and
    ``embedding_id`` (an internal vector-store pointer, not a client concern).
    ``document_id``/``page_start``/``page_end`` are null except for a
    ``summary``-type memory, which is keyed to a page range.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: MemoryType
    content: str
    document_id: uuid.UUID | None
    page_start: int | None
    page_end: int | None
    created_at: datetime


class MemoryPage(BaseModel):
    """A page of the user's stored memories (standard list envelope)."""

    items: list[MemoryPublic]
    total: int
    page: int
    page_size: int


class RecommendationPublic(BaseModel):
    """One explainable recommendation (FR-5) — from the reader's own library, or the web.

    ``document_id``/``author`` are set only for a library recommendation;
    ``url`` only for a web-sourced one — the two paths share this one shape so
    the client doesn't need to know which produced a given item.
    """

    model_config = ConfigDict(from_attributes=True)

    title: str
    reason: str
    document_id: uuid.UUID | None
    author: str | None
    url: str | None
    score: float | None


class RecommendationsResponse(BaseModel):
    """Explainable recommendations for one user.

    Not the standard paginated list envelope: this is a bounded, computed
    top-N (like ``/analytics``), not a persisted collection to page through.
    """

    items: list[RecommendationPublic]


class EvaluationRunRequest(BaseModel):
    """Trigger an evaluation run against a named, versioned dataset (FR-12)."""

    dataset_name: str = Field(min_length=1, max_length=255)


class EvaluationDatasetPublic(BaseModel):
    """One shipped evaluation dataset (name + immutable version)."""

    name: str
    version: str


class EvaluationDatasetList(BaseModel):
    """Shipped datasets the admin UI can enqueue (not paginated — the set is tiny)."""

    items: list[EvaluationDatasetPublic]


class EvaluationRunPublic(BaseModel):
    """One persisted evaluation run: what it ran with, and its scores.

    ``results`` holds every case's individual scores; ``summary`` the run-level
    aggregate — both opaque JSON here since their shape is scorer-defined, not
    part of the API's typed contract.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_name: str
    dataset_version: str
    status: EvaluationRunStatus
    prompt_version: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    results: dict[str, Any]
    summary: dict[str, Any]
    error: str | None
    created_at: datetime


class EvaluationRunPage(BaseModel):
    """A page of evaluation runs (standard list envelope)."""

    items: list[EvaluationRunPublic]
    total: int
    page: int
    page_size: int
