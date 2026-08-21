"""Recommendations route: explainable suggestions from the reader's own library (FR-5).

Calls only :meth:`RecommendationService.recommend_from_library` — the internal,
never-gated signal (reading history + library similarity + long-term-memory
preferences). The external, web-sourced path is reachable only through the
agent's ``recommend`` tool during a chat turn, where HITL approval applies; a
plain GET here has no interrupt/resume channel to ask for that approval over.
"""

from fastapi import APIRouter, Query

from api.deps import (
    CurrentUser,
    DocumentRepositoryDep,
    MemoryRepositoryDep,
    MemoryServiceDep,
    ProgressRepositoryDep,
    ProgressServiceDep,
    RecommendationServiceDep,
)
from api.schemas import RecommendationPublic, RecommendationsResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=RecommendationsResponse, summary="Get reading recommendations")
async def get_recommendations(
    user: CurrentUser,
    documents: DocumentRepositoryDep,
    progress: ProgressRepositoryDep,
    progress_service: ProgressServiceDep,
    memories: MemoryRepositoryDep,
    memory_service: MemoryServiceDep,
    recommendation_service: RecommendationServiceDep,
    limit: int = Query(5, ge=1, le=10, description="Maximum recommendations to return."),
) -> RecommendationsResponse:
    """Return explainable recommendations from the caller's own library (FR-5.1/5.3).

    Combines reading history (completed/in-progress documents), semantic
    similarity across the caller's library, and stated long-term-memory
    preferences/habits — never an external call, so nothing here needs HITL
    approval. Returns an empty list when there's no history/preference yet to
    recommend from, rather than a guess.
    """
    items = await recommendation_service.recommend_from_library(
        user_id=user.id,
        documents=documents,
        progress_repo=progress,
        progress_service=progress_service,
        memories=memories,
        memory_service=memory_service,
        limit=limit,
    )
    return RecommendationsResponse(
        items=[RecommendationPublic.model_validate(item) for item in items]
    )
