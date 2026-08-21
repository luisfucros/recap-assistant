"""Evaluation routes: trigger and inspect dataset runs (FR-12), admin-only.

Both routes require :data:`~api.deps.AdminUser` — an evaluation run seeds and
scores fixture data under a dedicated system user, not the caller's own
library, so there is no per-caller data to scope these to; gating is purely an
operational-access control, not the isolation invariant the rest of the API
enforces.
"""

import uuid

from fastapi import APIRouter

from api.deps import (
    AdminUser,
    DbSession,
    EvaluationRunRepositoryDep,
    EvaluationServiceDep,
    UserRepositoryDep,
)
from api.schemas import EvaluationRunPublic, EvaluationRunRequest

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/run", response_model=EvaluationRunPublic, summary="Run an evaluation dataset")
async def run_evaluation(
    request: EvaluationRunRequest,
    admin: AdminUser,
    session: DbSession,
    users: UserRepositoryDep,
    runs: EvaluationRunRepositoryDep,
    evaluation_service: EvaluationServiceDep,
) -> EvaluationRunPublic:
    """Run every case in the named dataset through retrieval + the agent, score it, and persist the run.

    Raises:
        NotFoundError: ``dataset_name`` doesn't match a shipped dataset (404).
    """
    run = await evaluation_service.run_evaluation(
        dataset_name=request.dataset_name,
        session=session,
        users=users,
        runs=runs,
        triggered_by=admin.id,
    )
    return EvaluationRunPublic.model_validate(run)


@router.get("/{run_id}", response_model=EvaluationRunPublic, summary="Get an evaluation run")
async def get_evaluation_run(
    run_id: uuid.UUID,
    admin: AdminUser,
    runs: EvaluationRunRepositoryDep,
) -> EvaluationRunPublic:
    """Return a previously run evaluation's scores by id.

    Raises:
        NotFoundError: no run with this id exists (404).
    """
    run = await runs.get_or_404(run_id)
    return EvaluationRunPublic.model_validate(run)
