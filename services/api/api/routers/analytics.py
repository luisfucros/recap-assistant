"""Reading-analytics route: the caller's own pace, streaks, and history (FR-17).

Thin handler over :class:`~api.services.analytics_service.AnalyticsService`. The
user id comes from the access-token cookie (``CurrentUser``) and keys both the
computation and its cache, so a user only ever sees their own analytics.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Query

from api.deps import (
    AnalyticsServiceDep,
    CurrentUser,
    ProgressRepositoryDep,
    ReadingEventRepositoryDep,
)
from api.schemas import AnalyticsSummary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("", response_model=AnalyticsSummary, summary="Get your reading analytics")
async def get_analytics(
    user: CurrentUser,
    analytics_service: AnalyticsServiceDep,
    events: ReadingEventRepositoryDep,
    progress: ProgressRepositoryDep,
    window_days: int = Query(30, ge=1, le=365, description="Trailing window in days."),
) -> AnalyticsSummary:
    """Return the caller's reading pace, streaks, and pages-over-time.

    Computed over the trailing ``window_days`` from the reading-event trail and
    current status counts, and served from a short-lived per-user cache.
    """
    return await analytics_service.get_analytics(
        user_id=user.id,
        events=events,
        progress=progress,
        today=datetime.now(UTC).date(),
        window_days=window_days,
    )
