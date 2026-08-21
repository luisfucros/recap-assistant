"""Liveness endpoint used by compose/orchestration healthchecks."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health probe response."""

    status: str


@router.get("/health", summary="Liveness probe", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Return a static ``ok`` status so orchestrators can detect the process is up.

    This is intentionally dependency-free; deeper readiness checks (DB, Qdrant,
    Redis reachable) are added with the lifespan/singletons wiring.
    """
    return HealthStatus(status="ok")
