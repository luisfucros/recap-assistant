"""Celery task: execute one evaluation run by id."""

import uuid
from typing import Any

from api.eval_worker.base_task import AsyncTask, run_on_process_loop
from api.eval_worker.celery_app import app
from api.eval_worker.resources import get_eval_resources
from shared.core.enums import EvaluationRunStatus
from shared.repositories import EvaluationRunRepository, UserRepository


async def _execute(run_id: uuid.UUID) -> None:
    resources = get_eval_resources()
    async with resources.sessionmaker() as session:
        await resources.evaluation_service.execute_evaluation(
            run_id=run_id,
            session=session,
            users=UserRepository(session),
            runs=EvaluationRunRepository(session),
        )


async def _mark_failed(run_id: uuid.UUID, reason: str) -> None:
    """Safety net if the task raises after Celery has given up."""
    resources = get_eval_resources()
    async with resources.sessionmaker() as session:
        runs = EvaluationRunRepository(session)
        run = await runs.get(run_id)
        if run is None or run.status in (
            EvaluationRunStatus.COMPLETED,
            EvaluationRunStatus.FAILED,
        ):
            return
        run.status = EvaluationRunStatus.FAILED
        run.error = reason[:2048]
        await session.commit()


class RunEvaluationTask(AsyncTask):
    """Marks a still-in-progress run ``failed`` if Celery exhausts the task."""

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        super().on_failure(exc, task_id, args, kwargs, einfo)
        raw = args[0] if args else kwargs.get("run_id")
        if raw is None:
            return
        run_on_process_loop(_mark_failed(uuid.UUID(str(raw)), str(exc)))


@app.task(base=RunEvaluationTask, bind=True, name="eval.run_evaluation")
def run_evaluation_task(self: RunEvaluationTask, run_id: str) -> None:
    """Score one ``EvaluationRun`` (enqueued after the pending row is committed)."""
    self.run_async(_execute(uuid.UUID(run_id)))
