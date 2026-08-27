"""Process-wide ``Resources`` for the eval Celery worker."""

from functools import lru_cache

from api.resources import Resources
from shared.core.config import get_settings


@lru_cache
def get_eval_resources() -> Resources:
    """Return the process-wide API ``Resources`` (built once, cached)."""
    return Resources(get_settings())
