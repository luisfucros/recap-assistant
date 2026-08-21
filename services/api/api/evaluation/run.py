"""CLI entrypoint for running an evaluation dataset (FR-12.3).

    python -m api.evaluation.run [dataset_name]

Builds the same ``Resources`` composition root the app's lifespan does, runs
one dataset through ``EvaluationService``, prints its scores, and exits
non-zero only if the run itself failed to complete — never on a low score;
promoting a fixed dataset to a pass/fail CI gate is a later milestone.
"""

import argparse
import asyncio
import sys

from shared.core.enums import EvaluationRunStatus


async def _run(dataset_name: str) -> int:
    from api.resources import Resources
    from shared.core.config import get_settings
    from shared.repositories import EvaluationRunRepository, UserRepository

    resources = Resources(get_settings())
    try:
        async with resources.sessionmaker() as session:
            run = await resources.evaluation_service.run_evaluation(
                dataset_name=dataset_name,
                session=session,
                users=UserRepository(session),
                runs=EvaluationRunRepository(session),
            )
    finally:
        await resources.aclose()

    print(f"run {run.id} — {run.dataset_name}@{run.dataset_version} — {run.status.value}")
    print(f"  prompt={run.prompt_version} model={run.llm_provider}:{run.llm_model}")
    print(f"  embedding={run.embedding_model}")
    if run.status is EvaluationRunStatus.FAILED:
        print(f"  error: {run.error}")
        return 1
    for key, value in run.summary.items():
        print(f"  {key}: {value}")
    return 0


def main() -> None:
    """Parse args and run (the entry point ``python -m api.evaluation.run`` invokes)."""
    parser = argparse.ArgumentParser(description="Run an evaluation dataset (FR-12.3).")
    parser.add_argument(
        "dataset_name", nargs="?", default="sample_v1", help="Dataset name (default: sample_v1)."
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.dataset_name)))


if __name__ == "__main__":
    main()
