"""Authentication: password hashing, JWT issue/verify, register/login/refresh.

Passwords are hashed with argon2 (via ``pwdlib``); access/refresh tokens are
signed JWTs (via ``PyJWT``) carrying the user id in ``sub`` and a ``type`` claim
so an access token can't be used where a refresh token is required. Refresh
rotation issues a fresh access+refresh pair on each refresh.

This service is transport-agnostic: routers translate its exceptions into HTTP
status codes, and pass in a ``UserRepository`` (the DB boundary).
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import jwt
from pwdlib import PasswordHash

from shared.core.config import Settings
from shared.core.passwords import build_password_hash
from shared.models.user import User
from shared.repositories.user_repository import UserRepository

TokenType = Literal["access", "refresh"]


class AuthError(Exception):
    """Base class for authentication errors."""


class AuthConfigError(AuthError):
    """Auth is misconfigured (e.g. no signing secret)."""


class DuplicateUserError(AuthError):
    """Registration attempted with an email that already exists."""


class InvalidCredentialsError(AuthError):
    """Email/password did not match a user."""


class InvalidTokenError(AuthError):
    """A token was missing, malformed, expired, or of the wrong type."""


@dataclass(frozen=True, slots=True)
class TokenPair:
    """An issued access + refresh token pair."""

    access_token: str
    refresh_token: str


class AuthService:
    """Password + JWT authentication, independent of the web layer."""

    def __init__(self, settings: Settings, *, password_hash: PasswordHash | None = None) -> None:
        if settings.jwt_secret is None:
            raise AuthConfigError("JWT_SECRET must be set to issue/verify tokens")
        self._secret = settings.jwt_secret.get_secret_value()
        self._algorithm = settings.jwt_algorithm
        self._access_ttl = timedelta(minutes=settings.access_token_ttl_minutes)
        self._refresh_ttl = timedelta(days=settings.refresh_token_ttl_days)
        self._default_language = settings.default_language
        self._default_spoiler_safe = settings.spoiler_safe_default
        # Argon2 by default; injectable for tests.
        self._pwd = password_hash or build_password_hash()

    # --- passwords --------------------------------------------------------- #
    def hash_password(self, password: str) -> str:
        return self._pwd.hash(password)

    def verify_password(self, password: str, hashed: str) -> bool:
        return self._pwd.verify(password, hashed)

    # --- tokens ------------------------------------------------------------ #
    def issue_tokens(self, user_id: uuid.UUID) -> TokenPair:
        """Issue a fresh access + refresh pair for ``user_id``."""
        return TokenPair(
            access_token=self._encode(user_id, "access", self._access_ttl),
            refresh_token=self._encode(user_id, "refresh", self._refresh_ttl),
        )

    def decode_token(self, token: str, *, expected_type: TokenType) -> uuid.UUID:
        """Validate a token and return its subject (user id).

        Raises:
            InvalidTokenError: bad signature, expired, malformed, or wrong type.
        """
        try:
            payload = jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except jwt.PyJWTError as exc:
            raise InvalidTokenError("invalid or expired token") from exc
        if payload.get("type") != expected_type:
            raise InvalidTokenError(f"expected a {expected_type} token")
        try:
            return uuid.UUID(payload["sub"])
        except (KeyError, ValueError) as exc:
            raise InvalidTokenError("token has no valid subject") from exc

    def _encode(self, user_id: uuid.UUID, token_type: TokenType, ttl: timedelta) -> str:
        now = datetime.now(tz=UTC)
        payload = {"sub": str(user_id), "type": token_type, "iat": now, "exp": now + ttl}
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    # --- flows (need the DB) ---------------------------------------------- #
    async def register(
        self,
        repo: UserRepository,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        is_admin: bool = False,
    ) -> User:
        """Create a password user. Raises ``DuplicateUserError`` if email exists.

        ``is_admin`` defaults false for public self-registration; only the
        admin-only user-creation route passes ``True``.
        """
        if await repo.get_by_email(email) is not None:
            raise DuplicateUserError(f"a user with email {email!r} already exists")
        user = User(
            email=email,
            hashed_password=self.hash_password(password),
            display_name=display_name,
            preferred_language=self._default_language,
            spoiler_safe=self._default_spoiler_safe,
            is_admin=is_admin,
        )
        return await repo.add(user)

    async def authenticate(self, repo: UserRepository, *, email: str, password: str) -> User:
        """Return the user for valid credentials, else ``InvalidCredentialsError``."""
        user = await repo.get_by_email(email)
        if user is None or user.hashed_password is None:
            raise InvalidCredentialsError("invalid email or password")
        if not self.verify_password(password, user.hashed_password):
            raise InvalidCredentialsError("invalid email or password")
        return user

    async def authenticate_google(
        self,
        repo: UserRepository,
        *,
        google_sub: str,
        email: str,
        display_name: str | None = None,
    ) -> User:
        """Resolve (or provision) the user behind a verified Google identity.

        Link order: match on ``google_sub`` first (the stable Google id); else
        attach the Google identity to an existing password account with the same
        email (so a user who signed up with a password can also use Google); else
        create a new OAuth-only account (no password). The caller is responsible
        for committing the session.
        """
        user = await repo.get_by_google_sub(google_sub)
        if user is not None:
            return user
        user = await repo.get_by_email(email)
        if user is not None:
            user.google_sub = google_sub
            return user
        return await repo.add(
            User(
                email=email,
                google_sub=google_sub,
                hashed_password=None,
                display_name=display_name,
                preferred_language=self._default_language,
                spoiler_safe=self._default_spoiler_safe,
                is_admin=False,
            )
        )

    async def refresh(self, repo: UserRepository, refresh_token: str) -> TokenPair:
        """Validate a refresh token and issue a rotated access + refresh pair."""
        user_id = self.decode_token(refresh_token, expected_type="refresh")
        if await repo.get_by_id(user_id) is None:
            raise InvalidTokenError("token subject is not a known user")
        return self.issue_tokens(user_id)
