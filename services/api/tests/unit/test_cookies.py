"""Unit tests for auth-cookie flags and scoping (no HTTP server needed)."""

import pytest
from api.security import ACCESS_COOKIE, REFRESH_COOKIE, clear_auth_cookies, set_auth_cookies
from api.services.auth_service import TokenPair
from fastapi import Response

from shared.core.config import Settings

pytestmark = pytest.mark.unit


def _set_cookie_headers(response: Response) -> list[str]:
    return [v.decode() for k, v in response.raw_headers if k == b"set-cookie"]


def test_cookies_are_httponly_and_carry_the_tokens() -> None:
    response = Response()
    settings = Settings(_env_file=None, cookie_secure=True, cookie_samesite="lax")

    set_auth_cookies(response, TokenPair(access_token="acc", refresh_token="ref"), settings)
    headers = _set_cookie_headers(response)
    joined = "\n".join(headers)

    assert len(headers) == 2
    assert f"{ACCESS_COOKIE}=acc" in joined
    assert f"{REFRESH_COOKIE}=ref" in joined
    # httpOnly on both (JS can't read them); Secure because cookie_secure=True.
    assert joined.lower().count("httponly") == 2
    assert joined.lower().count("secure") == 2


def test_refresh_cookie_is_path_scoped_to_auth() -> None:
    response = Response()
    settings = Settings(_env_file=None, api_v1_prefix="/api/v1")

    set_auth_cookies(response, TokenPair(access_token="acc", refresh_token="ref"), settings)
    refresh_header = next(h for h in _set_cookie_headers(response) if h.startswith(REFRESH_COOKIE))
    access_header = next(h for h in _set_cookie_headers(response) if h.startswith(ACCESS_COOKIE))

    assert "Path=/api/v1/auth" in refresh_header
    assert "Path=/" in access_header


def test_insecure_cookies_when_disabled() -> None:
    response = Response()
    settings = Settings(_env_file=None, cookie_secure=False)

    set_auth_cookies(response, TokenPair(access_token="a", refresh_token="r"), settings)

    assert "secure" not in "\n".join(_set_cookie_headers(response)).lower()


def test_clear_expires_both_cookies() -> None:
    response = Response()
    settings = Settings(_env_file=None)

    clear_auth_cookies(response, settings)
    headers = _set_cookie_headers(response)

    assert len(headers) == 2
    # delete_cookie expires immediately (Max-Age=0).
    assert all("Max-Age=0" in h for h in headers)
