"""Unit tests for the checkpointer DSN conversion (no infrastructure).

The pool/saver themselves need a real Postgres and are covered in the integration
tier; here we pin the pure transformation that lets one configured
``DATABASE_URL`` feed both the asyncpg SQLAlchemy engine and the psycopg-based
checkpointer.
"""

import pytest
from api.checkpointer import to_psycopg_dsn

pytestmark = pytest.mark.unit


def test_strips_asyncpg_driver_suffix() -> None:
    assert (
        to_psycopg_dsn("postgresql+asyncpg://postgres:pw@db:5432/recap")
        == "postgresql://postgres:pw@db:5432/recap"
    )


def test_leaves_plain_postgresql_url_unchanged() -> None:
    assert (
        to_psycopg_dsn("postgresql://postgres:pw@db:5432/recap")
        == "postgresql://postgres:pw@db:5432/recap"
    )


def test_strips_any_driver_suffix() -> None:
    assert to_psycopg_dsn("postgresql+psycopg://u@h/d") == "postgresql://u@h/d"


def test_preserves_query_parameters() -> None:
    assert (
        to_psycopg_dsn("postgresql+asyncpg://u:p@h:5432/d?sslmode=require")
        == "postgresql://u:p@h:5432/d?sslmode=require"
    )


def test_rejects_a_non_url() -> None:
    with pytest.raises(ValueError, match="not a database URL"):
        to_psycopg_dsn("just-a-host:5432")
