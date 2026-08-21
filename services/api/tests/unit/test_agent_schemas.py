"""Unit tests for the agent's structured node-output schemas (FR-7.9)."""

import pytest
from api.agent.schemas import (
    Complexity,
    GuardrailDecision,
    MemoryClassification,
    PageRangeProposal,
    PlannerDecision,
    SpoilerCheckDecision,
)
from pydantic import ValidationError

from shared.core.enums import MemoryType

pytestmark = pytest.mark.unit


def test_planner_decision_accepts_valid_complexity_and_plan() -> None:
    decision = PlannerDecision(
        complexity="standard", needs_tools=True, tool_plan=["retrieve_chunks"]
    )
    assert decision.complexity is Complexity.STANDARD
    assert decision.needs_tools is True
    assert decision.tool_plan == ["retrieve_chunks"]


def test_planner_decision_defaults_empty_tool_plan() -> None:
    decision = PlannerDecision(complexity="simple", needs_tools=False)
    assert decision.tool_plan == []


def test_planner_decision_rejects_unknown_complexity() -> None:
    with pytest.raises(ValidationError):
        PlannerDecision(complexity="trivial", needs_tools=False)


def test_guardrail_decision_shape() -> None:
    decision = GuardrailDecision(on_topic=False, safe=True, reason="off topic")
    assert decision.on_topic is False
    assert decision.safe is True
    assert decision.reason == "off topic"


def test_memory_classification_summary_carries_page_range() -> None:
    classification = MemoryClassification(type="summary", salient=True, page_start=10, page_end=24)
    assert classification.type is MemoryType.SUMMARY
    assert (classification.page_start, classification.page_end) == (10, 24)


def test_memory_classification_non_summary_defaults_null_pages() -> None:
    classification = MemoryClassification(type="preference", salient=True)
    assert classification.type is MemoryType.PREFERENCE
    assert classification.page_start is None and classification.page_end is None


def test_memory_classification_content_defaults_null() -> None:
    classification = MemoryClassification(type="preference", salient=False)
    assert classification.content is None


def test_memory_classification_carries_a_salient_facts_content() -> None:
    classification = MemoryClassification(
        type="fact", salient=True, content="Is 34 years old and works as a teacher."
    )
    assert classification.content == "Is 34 years old and works as a teacher."


def test_memory_classification_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        MemoryClassification(type="notes", salient=True)


def test_page_range_proposal_requires_bounds_and_reason() -> None:
    proposal = PageRangeProposal(page_start=1, page_end=30, proposal_reason="read since last recap")
    assert (proposal.page_start, proposal.page_end) == (1, 30)
    with pytest.raises(ValidationError):
        PageRangeProposal(page_start=1, page_end=30)  # missing proposal_reason


def test_spoiler_check_decision_flags_risk_with_a_reason() -> None:
    decision = SpoilerCheckDecision(spoiler_risk=True, reason="reveals the ending")
    assert decision.spoiler_risk is True
    assert decision.reason == "reveals the ending"


def test_spoiler_check_decision_defaults_reason_empty() -> None:
    decision = SpoilerCheckDecision(spoiler_risk=False)
    assert decision.reason == ""
