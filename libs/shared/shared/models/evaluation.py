"""The ``EvaluationRun`` model — a persisted record of one dataset evaluation.

An evaluation run is a system-level operation an admin triggers against a
versioned dataset (FR-12), not per-user data the run was triggered *for* — so,
like :class:`~shared.models.outbox.OutboxEvent`, this is deliberately **not** a
:class:`~shared.repositories.base.UserScopedRepository` subject. ``triggered_by``
is only an audit pointer to the admin who ran it.

There is no in-progress status: the API's ``EvaluationService`` runs a dataset
to completion (or failure) synchronously within the triggering request/CLI
call, so every row lands with a terminal
:class:`~shared.core.enums.EvaluationRunStatus`.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from shared.core.enums import EvaluationRunStatus
from shared.db.base import Base
from shared.models.types import EVALUATION_RUN_STATUS_TYPE


class EvaluationRun(Base):
    """One run of a versioned dataset through retrieval + the agent, scored."""

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_name: Mapped[str] = mapped_column(String(255))
    dataset_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[EvaluationRunStatus] = mapped_column(EVALUATION_RUN_STATUS_TYPE)

    # The prompt/model/embedding identifiers this run actually answered with, so
    # two runs are comparable across a prompt or provider change (FR-12.3).
    prompt_version: Mapped[str] = mapped_column(String(64))
    llm_provider: Mapped[str] = mapped_column(String(64))
    llm_model: Mapped[str] = mapped_column(String(255))
    embedding_model: Mapped[str] = mapped_column(String(255))

    # Per-case scores (keyed by case id) and the run-level aggregate.
    results: Mapped[dict[str, Any]] = mapped_column(JSONB)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Set only when status is FAILED (e.g. the dataset failed to load).
    error: Mapped[str | None] = mapped_column(String(2048))

    triggered_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
