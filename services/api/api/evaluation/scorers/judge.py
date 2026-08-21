"""The LLM-as-judge answer-quality scorer (FR-12.2).

``EvaluationJudgment`` is the structured output of the ``evaluation_judge@v1``
prompt (rendered and invoked by ``EvaluationService``, the same
render-then-``.ainvoke`` pattern the agent graph's own structured judges use)
— kept here, next to the retrieval scorers, rather than in ``api.agent.schemas``
since it judges a finished answer for evaluation, not an in-flight turn.
"""

from pydantic import BaseModel, Field


class EvaluationJudgment(BaseModel):
    """One LLM-as-judge verdict on an agent answer against its context."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    citation_ok: bool
    reasoning: str = ""
