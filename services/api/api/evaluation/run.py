"""CLI entrypoint for running an evaluation dataset (FR-12.3, FR-12.5).

    python -m api.evaluation.run [dataset_name]
    python -m api.evaluation.run [dataset_name] --sync

Default: enqueue a pending run (same path as ``POST /evaluations/run``) and
wait until the eval worker writes a terminal status. ``--sync`` scores
in-process (no worker). Exits non-zero only if the run failed, never on a
low score.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from api.resources import Resources
from shared.core.enums import EvaluationRunStatus
from shared.models.evaluation import EvaluationRun

_POLL_SECONDS = 2.0
_TERMINAL = (EvaluationRunStatus.COMPLETED, EvaluationRunStatus.FAILED)


def _print_run(run: EvaluationRun) -> None:
    print(f"run {run.id} — {run.dataset_name}@{run.dataset_version} — {run.status.value}")
    print(f"  prompt={run.prompt_version} model={run.llm_provider}:{run.llm_model}")
    print(f"  embedding={run.embedding_model}")
    if run.status is EvaluationRunStatus.FAILED:
        print(f"  error: {run.error}")
        return
    for key, value in (run.summary or {}).items():
        print(f"  {key}: {value}")


async def _wait_for_terminal(resources: Resources, run_id: uuid.UUID) -> EvaluationRun:
    from shared.repositories import EvaluationRunRepository

    while True:
        async with resources.sessionmaker() as session:
            run = await EvaluationRunRepository(session).get_or_404(run_id)
            if run.status in _TERMINAL:
                return run
        await asyncio.sleep(_POLL_SECONDS)


async def _run(dataset_name: str, *, sync: bool) -> int:
    from shared.core.config import get_settings
    from shared.repositories import EvaluationRunRepository, UserRepository

    resources = Resources(get_settings())
    try:
        async with resources.sessionmaker() as session:
            service = resources.evaluation_service
            users = UserRepository(session)
            runs = EvaluationRunRepository(session)
            if sync:
                run = await service.run_evaluation(
                    dataset_name=dataset_name,
                    session=session,
                    users=users,
                    runs=runs,
                )
            else:
                pending = await service.enqueue_evaluation(
                    dataset_name=dataset_name,
                    session=session,
                    runs=runs,
                )
                print(f"enqueued {pending.id} ({pending.status.value}); waiting for worker…")
                run = await _wait_for_terminal(resources, pending.id)
    finally:
        await resources.aclose()

    _print_run(run)
    return 1 if run.status is EvaluationRunStatus.FAILED else 0


def main() -> None:
    """Parse args and run (the entry point ``python -m api.evaluation.run`` invokes)."""
    parser = argparse.ArgumentParser(description="Run an evaluation dataset (FR-12.5).")
    parser.add_argument(
        "dataset_name", nargs="?", default="sample_v1", help="Dataset name (default: sample_v1)."
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Score in-process instead of waiting for the eval worker.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.dataset_name, sync=args.sync)))


if __name__ == "__main__":
    main()
