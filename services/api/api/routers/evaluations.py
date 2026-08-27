"""Evaluation routes: enqueue, list, and inspect dataset runs (FR-12), admin-only.

Scoring is off this request path (FR-12.5): ``POST /run`` persists ``pending``
and returns 202; the eval worker writes ``completed``/``failed``. Clients poll
``GET /evaluations`` / ``GET /evaluations/{id}``.
"""

import uuid

from fastapi import APIRouter, Query, status

from api.deps import (
    AdminUser,
    DbSession,
    EvaluationRunRepositoryDep,
    EvaluationServiceDep,
)
from api.evaluation.datasets.loader import list_datasets
from api.schemas import (
    EvaluationDatasetList,
    EvaluationDatasetPublic,
    EvaluationRunPage,
    EvaluationRunPublic,
    EvaluationRunRequest,
)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post(
    "/run",
    response_model=EvaluationRunPublic,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue an evaluation dataset run",
)
async def run_evaluation(
    request: EvaluationRunRequest,
    admin: AdminUser,
    session: DbSession,
    runs: EvaluationRunRepositoryDep,
    evaluation_service: EvaluationServiceDep,
) -> EvaluationRunPublic:
    """Persist a pending run and enqueue scoring. 404 if the dataset is unknown.

    Raises:
        NotFoundError: ``dataset_name`` doesn't match a shipped dataset (404).
    """
    run = await evaluation_service.enqueue_evaluation(
        dataset_name=request.dataset_name,
        session=session,
        runs=runs,
        triggered_by=admin.id,
    )
    return EvaluationRunPublic.model_validate(run)


@router.get("/datasets", response_model=EvaluationDatasetList, summary="List evaluation datasets")
async def list_evaluation_datasets(_admin: AdminUser) -> EvaluationDatasetList:
    """Return shipped dataset names and versions for the admin run picker."""
    items = [
        EvaluationDatasetPublic(name=dataset.name, version=dataset.version)
        for dataset in list_datasets()
    ]
    return EvaluationDatasetList(items=items)


@router.get("", response_model=EvaluationRunPage, summary="List evaluation runs")
async def list_evaluation_runs(
    _admin: AdminUser,
    runs: EvaluationRunRepositoryDep,
    page: int = Query(1, ge=1, description="1-based page number."),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (max 100)."),
) -> EvaluationRunPage:
    """Return a page of evaluation runs, newest first."""
    offset = (page - 1) * page_size
    items = await runs.list_recent(limit=page_size, offset=offset)
    total = await runs.count()
    return EvaluationRunPage(
        items=[EvaluationRunPublic.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{run_id}", response_model=EvaluationRunPublic, summary="Get an evaluation run")
async def get_evaluation_run(
    run_id: uuid.UUID,
    _admin: AdminUser,
    runs: EvaluationRunRepositoryDep,
) -> EvaluationRunPublic:
    """Return a previously run evaluation's scores by id.

    Raises:
        NotFoundError: no run with this id exists (404).
    """
    run = await runs.get_or_404(run_id)
    return EvaluationRunPublic.model_validate(run)
