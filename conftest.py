"""Shared fixtures for the integration & functional tiers (real infra).

These tiers run against the throwaway services in ``docker-compose.test.yml``
(Postgres/Qdrant/MinIO on offset ports). When that stack is not reachable the
fixtures **skip** rather than fail, so ``pytest -m unit`` still runs anywhere and
``pytest -m integration`` gives a clear "start the test stack" message.

Design notes:
- Schema is created once per session via a *synchronous* fixture that drives its
  own ``asyncio.run`` — this sidesteps event-loop-scope pitfalls (asyncpg
  connections are bound to the loop that opened them). Per-test async fixtures
  then open their own engine in the test's loop.
- Isolation is per test: relational tables are truncated after each test; the
  Qdrant test collection is dropped around each test that uses it.
"""

import asyncio
import contextlib
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Connection targets — overridable by env, defaulting to docker-compose.test.yml.
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/recap_test"
)
TEST_QDRANT_URL = os.getenv("TEST_QDRANT_URL", "http://localhost:6335")
TEST_S3_ENDPOINT_URL = os.getenv("TEST_S3_ENDPOINT_URL", "http://localhost:9002")
TEST_JWT_SECRET = "integration-secret-please-change-000000000000"
TEST_BUCKET = "recap-test"
TEST_CHUNKS_COLLECTION = "document_chunks_test"

# Relational tables to truncate between tests (children first is unnecessary with
# CASCADE, but the list is the schema's real tables — never test-only ones).
_TRUNCATE_TABLES = (
    "reading_events",
    "reading_progress",
    "chunks",
    "documents",
    "outbox",
    "evaluation_runs",
    "users",
)


def make_test_settings(**overrides):  # noqa: ANN003, ANN201 — Settings, kwargs vary per test
    """Build a ``Settings`` pointed at the test infra (overridable per test)."""
    from shared.core.config import Settings

    base = {
        "database_url": TEST_DATABASE_URL,
        "qdrant_url": TEST_QDRANT_URL,
        "qdrant_chunks_collection": TEST_CHUNKS_COLLECTION,
        "s3_endpoint_url": TEST_S3_ENDPOINT_URL,
        "s3_access_key_id": "minioadmin",
        "s3_secret_access_key": "minioadmin",
        "s3_bucket": TEST_BUCKET,
        "jwt_secret": TEST_JWT_SECRET,
        "cookie_secure": False,
        # Skip startup warm-up in tests: keep boot fast and avoid building a real
        # embedder / probing infra during app-lifespan (functional) tests.
        "warm_up_on_start": False,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture(scope="session")
def test_settings():  # noqa: ANN201
    """Session-wide settings for the real-infra tiers."""
    return make_test_settings()


@pytest.fixture(scope="session")
def _schema_ready(test_settings) -> None:  # noqa: ANN001
    """Create the schema once (drop+create), or skip the tier if Postgres is down."""
    import shared.models  # noqa: F401 — registers every model on Base.metadata
    from shared.db.base import Base

    async def _setup() -> None:
        engine = create_async_engine(test_settings.database_url)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    try:
        asyncio.run(_setup())
    except (SQLAlchemyError, OSError) as exc:
        pytest.skip(f"test Postgres not reachable ({exc}); run docker-compose.test.yml")


@pytest.fixture
async def db_engine(test_settings, _schema_ready):  # noqa: ANN001, ANN201
    """A per-test async engine; truncates all tables on teardown for isolation.

    Truncation lives here (not in ``db_session``) so every DB-touching path —
    including tests that open their own sessions via ``db_sessionmaker`` — is
    isolated, whether or not it requested a ``db_session``.
    """
    engine = create_async_engine(test_settings.database_url)
    try:
        yield engine
    finally:
        with contextlib.suppress(SQLAlchemyError):
            async with engine.begin() as conn:
                await conn.execute(
                    text(f"TRUNCATE {', '.join(_TRUNCATE_TABLES)} RESTART IDENTITY CASCADE")
                )
        await engine.dispose()


@pytest.fixture
async def db_sessionmaker(db_engine):  # noqa: ANN001, ANN201
    """A sessionmaker bound to the per-test engine (``expire_on_commit=False``)."""
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def db_session(db_sessionmaker):  # noqa: ANN001, ANN201
    """Yield a single session for tests that want one (isolation via ``db_engine``)."""
    async with db_sessionmaker() as session:
        yield session


@pytest.fixture
async def qdrant_client(test_settings):  # noqa: ANN001, ANN201
    """A Qdrant client with the test collection cleaned around the test; skip if down."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=test_settings.qdrant_url, check_compatibility=False)
    try:
        await client.get_collections()
    except Exception as exc:
        await client.close()
        pytest.skip(f"test Qdrant not reachable ({exc}); run docker-compose.test.yml")

    collection = test_settings.qdrant_chunks_collection
    with contextlib.suppress(Exception):
        await client.delete_collection(collection)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            await client.delete_collection(collection)
        await client.close()


@pytest.fixture
async def storage(test_settings):  # noqa: ANN001, ANN201
    """A storage provider with the test bucket ensured; skip if MinIO is down."""
    import aioboto3
    from botocore.exceptions import BotoCoreError, ClientError

    from shared.providers.storage import build_storage_provider

    session = aioboto3.Session()
    try:
        async with session.client(
            "s3",
            endpoint_url=test_settings.s3_endpoint_url,
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            region_name=test_settings.s3_region,
        ) as s3:
            with contextlib.suppress(ClientError):  # already-exists is fine
                await s3.create_bucket(Bucket=test_settings.s3_bucket)
    except (BotoCoreError, OSError) as exc:
        pytest.skip(f"test MinIO not reachable ({exc}); run docker-compose.test.yml")

    return build_storage_provider(test_settings)


class FakeEmbedder:
    """Deterministic in-process embedder for pipeline tests (no external API).

    Real embeddings are an external boundary, so the integration tier mocks them
    while exercising real Postgres/Qdrant/MinIO. Vectors are stable per text so
    re-runs are reproducible.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts, *, batch_size=None) -> list[list[float]]:  # noqa: ANN001
        vectors = []
        for index, _text in enumerate(texts):
            # A simple, deterministic non-zero vector; distinct per position.
            base = float(index + 1)
            vectors.append([base + j for j in range(self._dim)])
        return vectors


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A deterministic embedder for pipeline tests."""
    return FakeEmbedder()
