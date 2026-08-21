"""Unit tests for the domain error hierarchy and standard error body."""

import pytest

from shared.core.errors import AppError, ConflictError, ErrorResponse, NotFoundError

pytestmark = pytest.mark.unit


def test_base_app_error_defaults():
    error = AppError()
    assert error.status_code == 500
    assert error.code == "INTERNAL_ERROR"


def test_subclass_carries_status_and_code():
    error = NotFoundError()
    assert error.status_code == 404
    assert error.code == "NOT_FOUND"


def test_message_override_preserved_in_str():
    error = NotFoundError("no such document")
    assert str(error) == "no such document"
    assert error.code == "NOT_FOUND"


def test_per_instance_code_override():
    error = ConflictError("duplicate", code="DUPLICATE_DOCUMENT")
    assert error.code == "DUPLICATE_DOCUMENT"
    assert error.status_code == 409


def test_error_response_shape_is_detail_and_code():
    body = ErrorResponse(detail="x", code="Y").model_dump()
    assert set(body) == {"detail", "code"}
