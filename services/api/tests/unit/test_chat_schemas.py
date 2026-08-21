"""Unit tests for the chat request schema, incl. multimodal attachments (FR-19).

Pure Pydantic validation — the boundary contract for ``POST /chat``: a turn needs
some content (text or an attachment), base64 media decodes, and the ``kind`` and
``mime_type`` are cross-checked against the allowlist so a part can't smuggle a
disallowed or mismatched type past the door.
"""

import base64

import pytest
from api.schemas import ChatMediaPart, ChatRequest, ResumeRequest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_text_only_request_is_valid() -> None:
    req = ChatRequest(message="who narrates?")
    assert req.message == "who narrates?"
    assert req.parts == []


def test_audio_part_decodes_from_base64() -> None:
    req = ChatRequest(
        message="",
        parts=[{"kind": "audio", "mime_type": "audio/wav", "data": _b64(b"RIFF....")}],
    )
    assert req.parts[0].data == b"RIFF...."
    assert req.parts[0].kind == "audio"


def test_empty_turn_is_rejected() -> None:
    # No text and no attachments — nothing to answer.
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_attachment_only_turn_is_valid() -> None:
    req = ChatRequest(parts=[{"kind": "image", "mime_type": "image/png", "data": _b64(b"\x89PNG")}])
    assert not req.message.strip()
    assert req.parts[0].kind == "image"


def test_kind_and_mime_must_agree() -> None:
    # An "audio" part carrying an image mime type is rejected.
    with pytest.raises(ValidationError):
        ChatMediaPart(kind="audio", mime_type="image/png", data=_b64(b"x"))


def test_disallowed_mime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMediaPart(kind="image", mime_type="image/tiff", data=_b64(b"x"))


def test_empty_attachment_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatMediaPart(kind="audio", mime_type="audio/wav", data=_b64(b""))


def test_too_many_parts_is_rejected() -> None:
    part = {"kind": "image", "mime_type": "image/png", "data": _b64(b"x")}
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", parts=[part] * 9)


# --- ResumeRequest (HITL) ----------------------------------------------------- #


def test_resume_approve_is_valid() -> None:
    req = ResumeRequest(decision="approve")
    assert req.decision == "approve"


def test_resume_page_range_answer_needs_no_decision() -> None:
    req = ResumeRequest(page_start=1, page_end=42)
    assert req.decision is None


def test_resume_empty_payload_is_rejected() -> None:
    # Would serialize to `{}`, which LangGraph's Command(resume=...) treats as
    # "no resume value" and silently re-interrupts instead of progressing.
    with pytest.raises(ValidationError):
        ResumeRequest()


def test_resume_edit_without_args_or_page_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResumeRequest(decision="edit")


def test_resume_edit_with_args_is_valid() -> None:
    req = ResumeRequest(decision="edit", args={"query": "new query"})
    assert req.args == {"query": "new query"}


def test_resume_edit_with_page_range_is_valid() -> None:
    req = ResumeRequest(decision="edit", page_start=5, page_end=30)
    assert (req.page_start, req.page_end) == (5, 30)


def test_resume_inverted_page_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ResumeRequest(page_start=30, page_end=5)
