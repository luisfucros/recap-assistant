"""Async SQLAlchemy engine construction, shared by the app and the migration runner.

Both the API service (in its lifespan) and Alembic (in ``migrations/env.py``) need
an ``AsyncEngine`` built from the same configuration; centralizing it here avoids
divergence in driver/pool settings between runtime and migrations.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from shared.core.config import Settings, get_settings


def create_database_engine(settings: Settings | None = None) -> AsyncEngine:
    """Build an ``AsyncEngine`` from settings.

    Args:
        settings: Optional settings override; defaults to the process-wide
            singleton. Passing an explicit instance keeps this testable.

    Returns:
        A new ``AsyncEngine`` bound to ``DATABASE_URL``. ``pool_pre_ping`` guards
        against stale connections after a DB restart / idle timeout.
    """
    settings = settings or get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True, future=True)
