"""Unit tests for the evaluation scorers (FR-12.2): pure math, no I/O."""

import pytest
from api.evaluation.scorers.judge import EvaluationJudgment
from api.evaluation.scorers.retrieval import hit_rate, mean_reciprocal_rank, recall
from pydantic import ValidationError


@pytest.mark.unit
class TestHitRate:
    def test_scores_zero_with_no_expected_chunks(self) -> None:
        assert hit_rate(["a", "b"], []) == 0.0

    def test_scores_one_when_any_expected_chunk_is_retrieved(self) -> None:
        assert hit_rate(["a", "b", "c"], ["c", "z"]) == 1.0

    def test_scores_zero_when_no_expected_chunk_is_retrieved(self) -> None:
        assert hit_rate(["a", "b"], ["z"]) == 0.0

    def test_scores_zero_on_empty_retrieval(self) -> None:
        assert hit_rate([], ["a"]) == 0.0


@pytest.mark.unit
class TestRecall:
    def test_scores_zero_with_no_expected_chunks(self) -> None:
        assert recall(["a"], []) == 0.0

    def test_scores_full_recall_when_all_expected_chunks_are_retrieved(self) -> None:
        assert recall(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_scores_partial_recall(self) -> None:
        assert recall(["a"], ["a", "b"]) == 0.5

    def test_scores_zero_on_no_overlap(self) -> None:
        assert recall(["a"], ["z"]) == 0.0

    def test_ignores_retrieved_chunks_that_were_not_expected(self) -> None:
        # Extra, irrelevant hits don't hurt recall — only missing expected ones do.
        assert recall(["a", "junk1", "junk2"], ["a"]) == 1.0


@pytest.mark.unit
class TestMeanReciprocalRank:
    def test_scores_zero_with_no_expected_chunks(self) -> None:
        assert mean_reciprocal_rank(["a"], []) == 0.0

    def test_scores_one_when_first_result_is_expected(self) -> None:
        assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0

    def test_scores_half_when_second_result_is_the_first_expected_hit(self) -> None:
        assert mean_reciprocal_rank(["z", "a"], ["a"]) == 0.5

    def test_scores_zero_when_expected_chunk_is_never_retrieved(self) -> None:
        assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_uses_the_first_matching_rank_when_multiple_expected_chunks_hit(self) -> None:
        assert mean_reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


@pytest.mark.unit
class TestEvaluationJudgment:
    def test_accepts_valid_scores(self) -> None:
        judgment = EvaluationJudgment(faithfulness=0.9, relevance=0.8, citation_ok=True)

        assert judgment.faithfulness == 0.9
        assert judgment.citation_ok is True
        assert judgment.reasoning == ""

    def test_rejects_a_faithfulness_score_above_one(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationJudgment(faithfulness=1.5, relevance=0.5, citation_ok=True)

    def test_rejects_a_negative_relevance_score(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationJudgment(faithfulness=0.5, relevance=-0.1, citation_ok=True)
