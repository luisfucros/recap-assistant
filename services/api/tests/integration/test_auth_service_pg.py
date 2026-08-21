"""Integration tests for ``AuthService`` against a real Postgres.

Covers the DB-touching flows (register/authenticate/refresh/Google linking) with
real crypto and a real ``UserRepository`` — the unit tier mocks the DB, so this
is where the email-uniqueness and account-linking behavior is verified for real.
The JWT/argon2 crypto is genuine; nothing external is involved.
"""

import pytest
from api.services.auth_service import (
    AuthService,
    DuplicateUserError,
    InvalidCredentialsError,
)

from shared.core.enums import Language
from shared.repositories import UserRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def auth(test_settings) -> AuthService:
    return AuthService(test_settings)


async def test_register_persists_user_with_default_language(auth, db_session) -> None:
    repo = UserRepository(db_session)
    user = await auth.register(
        repo, email="ada@example.com", password="hunter2!", display_name="Ada"
    )
    await db_session.commit()

    fetched = await repo.get_by_email("ada@example.com")
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.hashed_password is not None and fetched.hashed_password != "hunter2!"
    assert fetched.preferred_language is Language.EN


async def test_register_duplicate_email_raises(auth, db_session) -> None:
    repo = UserRepository(db_session)
    await auth.register(repo, email="dup@example.com", password="hunter2!")
    await db_session.commit()

    with pytest.raises(DuplicateUserError):
        await auth.register(repo, email="dup@example.com", password="other-pass!")


async def test_authenticate_roundtrip(auth, db_session) -> None:
    repo = UserRepository(db_session)
    await auth.register(repo, email="log@example.com", password="hunter2!")
    await db_session.commit()

    user = await auth.authenticate(repo, email="log@example.com", password="hunter2!")
    assert user.email == "log@example.com"

    with pytest.raises(InvalidCredentialsError):
        await auth.authenticate(repo, email="log@example.com", password="wrong")


async def test_refresh_issues_working_tokens(auth, db_session) -> None:
    repo = UserRepository(db_session)
    user = await auth.register(repo, email="ref@example.com", password="hunter2!")
    await db_session.commit()

    tokens = auth.issue_tokens(user.id)
    rotated = await auth.refresh(repo, tokens.refresh_token)
    # The rotated access token decodes back to the same subject.
    assert auth.decode_token(rotated.access_token, expected_type="access") == user.id


async def test_google_links_existing_email_then_matches_by_sub(auth, db_session) -> None:
    repo = UserRepository(db_session)
    await auth.register(repo, email="both@example.com", password="hunter2!")
    await db_session.commit()

    # First Google sign-in links the google_sub onto the existing password account.
    linked = await auth.authenticate_google(
        repo, google_sub="google-123", email="both@example.com", display_name="Both"
    )
    await db_session.commit()
    assert linked.google_sub == "google-123"

    # A later sign-in matches by google_sub (same user, no duplicate created).
    again = await auth.authenticate_google(repo, google_sub="google-123", email="both@example.com")
    assert again.id == linked.id
