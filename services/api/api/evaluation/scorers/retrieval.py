"""Retrieval-quality scorers (FR-12.2): hit rate, recall, and MRR.

Pure functions over one case's retrieved vs. expected chunk ids — no I/O, so
they're unit-testable on hand-crafted lists. ``EvaluationService`` averages
each metric across a run's cases for the run-level summary.
"""

from collections.abc import Sequence


def hit_rate(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    """1.0 if any expected chunk was retrieved at all, else 0.0.

    A case with no expected chunks declared can't be judged, so it scores 0.0
    rather than raising — a dataset-authoring gap shows up as a low score, not
    a crashed run.
    """
    if not expected_ids:
        return 0.0
    return 1.0 if set(retrieved_ids) & set(expected_ids) else 0.0


def recall(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    """Fraction of the expected chunks that were retrieved."""
    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    return len(expected & set(retrieved_ids)) / len(expected)


def mean_reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    """``1 / rank`` of the first retrieved chunk that was expected, else 0.0.

    Retrieval order matters here (unlike hit rate/recall): a relevant chunk
    buried at the bottom of the results scores worse than one ranked first.
    """
    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank
    return 0.0
