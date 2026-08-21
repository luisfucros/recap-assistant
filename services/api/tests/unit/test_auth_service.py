"""Unit tests for AuthService (real crypto; DB mocked via a fake repository)."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from api.services.auth_service import (
    AuthService,
    DuplicateUserError,
    InvalidCredentialsError,
    InvalidTokenError,
    TokenPair,
)

from shared.core.config import Settings
from shared.core.enums import Language
from shared.models.user import User

# A ≥32-byte secret keeps PyJWT from warning about weak keys.
_SECRET = "test-secret-please-change-000000000000"


def _service() -> AuthService:
    return AuthService(Settings(_env_file=None, jwt_secret=_SECRET, access_token_ttl_minutes=15))


class FakeUserRepository:
    """In-memory stand-in for UserRepository (the DB boundary)."""

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
        self._by_id[user.id] = user
        self._by_email[user.email] = user
        if user.google_sub:
            self._by_google_sub[user.google_sub] = user
        return user


# --- passwords -------------------------------------------------------------


@pytest.mark.unit
def test_password_hash_roundtrip() -> None:
    svc = _service()
    hashed = svc.hash_password("hunter2")
    assert hashed != "hunter2"
    assert svc.verify_password("hunter2", hashed) is True
    assert svc.verify_password("wrong", hashed) is False


# --- tokens ----------------------------------------------------------------


@pytest.mark.unit
def test_access_token_roundtrips_to_user_id() -> None:
    svc = _service()
    uid = uuid.uuid4()
    tokens = svc.issue_tokens(uid)
    assert isinstance(tokens, TokenPair)
    assert svc.decode_token(tokens.access_token, expected_type="access") == uid
    assert svc.decode_token(tokens.refresh_token, expected_type="refresh") == uid


@pytest.mark.unit
def test_token_type_mismatch_rejected() -> None:
    svc = _service()
    tokens = svc.issue_tokens(uuid.uuid4())
    with pytest.raises(InvalidTokenError, match="expected a refresh token"):
        svc.decode_token(tokens.access_token, expected_type="refresh")


@pytest.mark.unit
def test_expired_token_rejected() -> None:
    svc = _service()
    expired = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": "access",
            "iat": datetime.now(tz=UTC) - timedelta(hours=2),
            "exp": datetime.now(tz=UTC) - timedelta(hours=1),
        },
        _SECRET,
        algorithm="HS256",
    )
    with pytest.raises(InvalidTokenError):
        svc.decode_token(expired, expected_type="access")


@pytest.mark.unit
def test_wrong_signature_rejected() -> None:
    svc = _service()
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "type": "access"}, "someone-elses-secret", algorithm="HS256"
    )
    with pytest.raises(InvalidTokenError):
        svc.decode_token(forged, expected_type="access")


# --- flows (fake repo) -----------------------------------------------------


@pytest.mark.unit
async def test_register_hashes_password_and_persists() -> None:
    svc = _service()
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2")
    assert user.email == "ada@example.com"
    assert user.hashed_password not in (None, "hunter2")
    assert await repo.get_by_email("ada@example.com") is user


@pytest.mark.unit
async def test_register_applies_default_language() -> None:
    svc = AuthService(Settings(_env_file=None, jwt_secret=_SECRET, default_language=Language.DE))
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2")
    assert user.preferred_language is Language.DE


@pytest.mark.unit
async def test_register_seeds_spoiler_safe_from_settings() -> None:
    svc = AuthService(Settings(_env_file=None, jwt_secret=_SECRET, spoiler_safe_default=False))
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2")
    assert user.spoiler_safe is False


@pytest.mark.unit
async def test_register_defaults_to_non_admin() -> None:
    svc = _service()
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2")
    assert user.is_admin is False


@pytest.mark.unit
async def test_register_can_create_an_admin() -> None:
    svc = _service()
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2", is_admin=True)
    assert user.is_admin is True


@pytest.mark.unit
async def test_register_duplicate_email_raises() -> None:
    svc = _service()
    repo = FakeUserRepository()
    await svc.register(repo, email="ada@example.com", password="hunter2")
    with pytest.raises(DuplicateUserError):
        await svc.register(repo, email="ada@example.com", password="other")


@pytest.mark.unit
async def test_authenticate_success_and_failure() -> None:
    svc = _service()
    repo = FakeUserRepository()
    await svc.register(repo, email="ada@example.com", password="hunter2")

    assert (await svc.authenticate(repo, email="ada@example.com", password="hunter2")).email == (
        "ada@example.com"
    )
    with pytest.raises(InvalidCredentialsError):
        await svc.authenticate(repo, email="ada@example.com", password="wrong")
    with pytest.raises(InvalidCredentialsError):
        await svc.authenticate(repo, email="nobody@example.com", password="hunter2")


@pytest.mark.unit
async def test_oauth_only_user_cannot_password_login() -> None:
    svc = _service()
    repo = FakeUserRepository()
    await repo.add(User(email="oauth@example.com", google_sub="g-123", hashed_password=None))
    with pytest.raises(InvalidCredentialsError):
        await svc.authenticate(repo, email="oauth@example.com", password="anything")


@pytest.mark.unit
async def test_refresh_rotates_tokens() -> None:
    svc = _service()
    repo = FakeUserRepository()
    user = await svc.register(repo, email="ada@example.com", password="hunter2")
    refresh = svc.issue_tokens(user.id).refresh_token

    rotated = await svc.refresh(repo, refresh)
    assert svc.decode_token(rotated.access_token, expected_type="access") == user.id
    # An access token cannot be used to refresh.
    with pytest.raises(InvalidTokenError):
        await svc.refresh(repo, rotated.access_token)


@pytest.mark.unit
async def test_refresh_unknown_user_rejected() -> None:
    svc = _service()
    repo = FakeUserRepository()
    orphan = svc.issue_tokens(uuid.uuid4()).refresh_token
    with pytest.raises(InvalidTokenError):
        await svc.refresh(repo, orphan)


# --- google oauth ----------------------------------------------------------


@pytest.mark.unit
async def test_google_creates_new_oauth_only_user() -> None:
    svc = _service()
    repo = FakeUserRepository()
    user = await svc.authenticate_google(
        repo, google_sub="g-1", email="ada@example.com", display_name="Ada"
    )
    assert user.google_sub == "g-1"
    assert user.hashed_password is None  # OAuth-only account has no password
    assert await repo.get_by_google_sub("g-1") is user


@pytest.mark.unit
async def test_google_returns_existing_user_by_sub() -> None:
    svc = _service()
    repo = FakeUserRepository()
    first = await svc.authenticate_google(repo, google_sub="g-1", email="ada@example.com")
    # A second sign-in with the same Google id resolves to the same account.
    again = await svc.authenticate_google(repo, google_sub="g-1", email="ada@example.com")
    assert again is first


@pytest.mark.unit
async def test_google_links_to_existing_password_account() -> None:
    svc = _service()
    repo = FakeUserRepository()
    existing = await svc.register(repo, email="ada@example.com", password="hunter2")
    assert existing.google_sub is None

    linked = await svc.authenticate_google(repo, google_sub="g-1", email="ada@example.com")
    # Same row, now carrying the Google id — password login still works too.
    assert linked is existing
    assert linked.google_sub == "g-1"
    assert linked.hashed_password is not None
