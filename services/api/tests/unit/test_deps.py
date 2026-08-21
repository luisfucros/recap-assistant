"""Unit tests for standalone dependency logic in ``api.deps``.

Most dependencies in this module are thin repository/service factories with no
branching worth a dedicated test; :func:`require_admin` is the exception.
"""

import uuid

import pytest
from api.deps import require_admin

from shared.core.errors import AuthorizationError
from shared.models.user import User


def _user(*, is_admin: bool) -> User:
    return User(id=uuid.uuid4(), email="reader@example.com", is_admin=is_admin)


@pytest.mark.unit
async def test_require_admin_passes_through_an_admin_user() -> None:
    admin = _user(is_admin=True)

    assert await require_admin(admin) is admin


@pytest.mark.unit
async def test_require_admin_rejects_a_non_admin_user_with_403() -> None:
    reader = _user(is_admin=False)

    with pytest.raises(AuthorizationError) as exc_info:
        await require_admin(reader)

    assert exc_info.value.status_code == 403
