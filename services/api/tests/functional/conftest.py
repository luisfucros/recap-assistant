"""Shared fixtures for functional (HTTP-level) tests.

One app and one ``TestClient`` are built for the whole functional session. This
is deliberate: the Prometheus instrumentator registers its metrics on the global
registry, so constructing multiple apps in a single process would double-register
and crash. A single shared app also mirrors the real one-app-per-process runtime.

The app is configured with a JWT secret and insecure cookies so auth flows work
over the test client's plain-HTTP transport. The only mocked boundary is the DB:
an in-memory fake user repository + a no-op session, wired via ``user_repo``.
"""

import uuid
from collections.abc import Iterator

import pytest
from api.app import create_app
from api.deps import get_db_session, get_rate_limit_service, get_user_repository
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.core.config import Settings
from shared.models.user import User

# A ≥32-byte secret keeps PyJWT from warning about weak keys.
TEST_JWT_SECRET = "test-secret-please-change-000000000000"


class FakeUserRepository:
    """In-memory stand-in for the DB-backed ``UserRepository``."""

    def __init__(self) -> None:
        self._by_id: dict[uuid.UUID, User] = {}
        self._by_email: dict[str, User] = {}
        self._by_google_sub: dict[str, User] = {}

    async def get_by_id(self, user_id: uuid.UUID) -> User | None:
        return self._by_id.get(user_id)

    async def get_by_email(self, email: str) -> User | None:
        return self._by_email.get(email)

    async def get_by_google_sub(self, google_sub: str) -> User | None:
        return self._by_google_sub.get(google_sub)

    async def add(self, user: User) -> User:
        if user.id is None:
            user.id = uuid.uuid4()
        self._index(user)
        return user

    def _index(self, user: User) -> None:
        self._by_id[user.id] = user
        self._by_email[user.email] = user
        if user.google_sub:
            self._by_google_sub[user.google_sub] = user


class _StubSession:
    """A DB session whose commit/rollback are no-ops (nothing to persist).

    ``commit`` re-indexes any users mutated in-place (e.g. a Google account
    linked to an existing email) so the fake repo reflects the write, mirroring
    what a real flush/commit would make visible.
    """

    def __init__(self, repo: FakeUserRepository) -> None:
        self._repo = repo

    async def commit(self) -> None:
        for user in list(self._repo._by_id.values()):
            self._repo._index(user)

    async def rollback(self) -> None: ...


@pytest.fixture(scope="session")
def app() -> FastAPI:
    """The single FastAPI app shared across functional tests."""
    return create_app(
        Settings(
            _env_file=None,
            backend_cors_origins="http://localhost:5173",
            jwt_secret=TEST_JWT_SECRET,
            cookie_secure=False,
        )
    )


@pytest.fixture(scope="session")
def client(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` with the app's lifespan (startup/shutdown) active."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user_repo(app: FastAPI, client: TestClient) -> Iterator[FakeUserRepository]:
    """Override the DB boundary with a fresh fake per test; reset cookies + overrides."""
    fake = FakeUserRepository()

    async def _stub_session():
        yield _StubSession(fake)

    app.dependency_overrides[get_user_repository] = lambda: fake
    app.dependency_overrides[get_db_session] = _stub_session
    client.cookies.clear()
    try:
        yield fake
    finally:
        app.dependency_overrides.pop(get_user_repository, None)
        app.dependency_overrides.pop(get_db_session, None)
        client.cookies.clear()


class _NeverRateLimited:
    """A rate limiter that never trips — the functional-test default.

    Real Redis may happen to be reachable from this process (e.g. a dev stack
    running alongside the test run) or not; either way, route tests other than
    the dedicated rate-limit tests shouldn't be flaky based on incidental Redis
    state shared across the session-scoped ``TestClient``/app.
    """

    async def enforce(self, **_: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _default_rate_limiting(app: FastAPI) -> None:
    """Disable rate limiting by default; a test can override further if it wants
    to assert 429 behavior (see the dedicated rate-limit tests)."""
    app.dependency_overrides[get_rate_limit_service] = lambda: _NeverRateLimited()
