"""Stuck-eval-run sweep (Celery beat safety net on the eval app only).

``enqueue_evaluation`` commits ``pending`` then ``.delay``s. If Redis is down
for that dispatch, or a worker dies mid-``running``, nothing else re-drives
the row. This tick re-enqueues ``pending``/``running`` rows older than
``EVAL_STUCK_THRESHOLD_SECONDS``. ``execute_evaluation`` is a no-op on an
already-terminal row, so a late overlapping task is safe once scoring finished.
"""

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from loguru import logger
from sqlalchemy import select

from api.eval_worker.base_task import AsyncTask
from api.eval_worker.celery_app import app
from api.eval_worker.resources import get_eval_resources
from api.resources import Resources
from shared.core.config import get_settings
from shared.core.enums import EvaluationRunStatus
from shared.models.evaluation import EvaluationRun

_STUCK_STATUSES = (EvaluationRunStatus.PENDING, EvaluationRunStatus.RUNNING)

Dispatch = Callable[[uuid.UUID], None]


def _dispatch(run_id: uuid.UUID) -> None:
    from api.eval_worker.tasks import run_evaluation_task

    run_evaluation_task.delay(str(run_id))


async def find_stuck_runs(
    resources: Resources, *, now: datetime, stuck_after_seconds: int
) -> Sequence[uuid.UUID]:
    """Return ids of ``pending``/``running`` rows stale past the threshold."""
    cutoff = now - timedelta(seconds=stuck_after_seconds)
    async with resources.sessionmaker() as session:
        result = await session.execute(
            select(EvaluationRun.id).where(
                EvaluationRun.status.in_(_STUCK_STATUSES),
                EvaluationRun.updated_at < cutoff,
            )
        )
        return [row[0] for row in result.all()]


async def sweep_stuck_runs(
    resources: Resources,
    *,
    now: datetime,
    stuck_after_seconds: int,
    dispatch: Dispatch = _dispatch,
) -> int:
    """Re-enqueue every stuck run found; return how many were dispatched."""
    stuck = await find_stuck_runs(resources, now=now, stuck_after_seconds=stuck_after_seconds)
    if not stuck:
        logger.debug("evaluation.sweep: no stuck runs")
        return 0
    for run_id in stuck:
        logger.bind(run_id=str(run_id)).warning(
            "evaluation.sweep: re-enqueuing stuck run past {}s", stuck_after_seconds
        )
        dispatch(run_id)
    logger.info("evaluation.sweep: re-enqueued {} runs", len(stuck))
    return len(stuck)


@app.task(base=AsyncTask, bind=True, name="eval.sweep_stuck_runs")
def sweep_stuck_runs_task(self: AsyncTask) -> int:
    """Beat-scheduled sweep tick."""
    settings = get_settings()
    return self.run_async(
        sweep_stuck_runs(
            get_eval_resources(),
            now=datetime.now(tz=UTC),
            stuck_after_seconds=settings.eval_stuck_threshold_seconds,
        )
    )
