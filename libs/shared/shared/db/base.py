"""Declarative base and shared metadata for all ORM models.

Every ORM model inherits from :class:`Base`, so ``Base.metadata`` is the single
source of truth for the relational schema — it is what Alembic autogenerate
diffs against. The explicit naming convention makes constraint/index names
deterministic (server-generated names differ across engines and break
autogenerate), which keeps migrations stable and reviewable.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

# Deterministic names for indexes/constraints so autogenerate produces stable,
# reviewable migrations (and downgrades can find objects by name).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models; owns the shared, conventionally-named metadata."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
