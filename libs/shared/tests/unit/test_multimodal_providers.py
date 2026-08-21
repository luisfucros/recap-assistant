"""Unit tests for the multimodal input providers (FR-19).

The hosted OpenAI transcriber/describer are tested against a fake ``AsyncOpenAI``-
shaped client (no network, mirrors ``test_providers.py``'s embedder pattern); the
factories are tested for config-driven selection and for a clear error when a
hosted key is missing. The local HuggingFace providers are tested only for their
"install the local extra" guard — the heavy ML stack isn't present in the unit
environment (and shouldn't be, for a <2s unit suite).
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from shared.core.config import Settings
from shared.providers.errors import ProviderConfigError
from shared.providers.transcription import (
    HuggingFaceTranscriber,
    OpenAIWhisperTranscriber,
    build_transcriber,
)
from shared.providers.vision import (
    OpenAIVisionDescriber,
    build_image_describer,
)

pytestmark = pytest.mark.unit


def _settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _fake_transcription_client(text: str) -> SimpleNamespace:
    """A minimal stand-in for ``AsyncOpenAI`` covering ``audio.transcriptions.create``."""
    create = AsyncMock(return_value=SimpleNamespace(text=text))
    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def _fake_chat_client(content: str) -> SimpleNamespace:
    """A minimal stand-in for ``AsyncOpenAI`` covering ``chat.completions.create``."""
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    create = AsyncMock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# --- hosted transcription ---------------------------------------------------- #


async def test_openai_transcriber_returns_text_and_uploads_audio() -> None:
    client = _fake_transcription_client("Odysseus sailed home.")
    transcriber = OpenAIWhisperTranscriber(api_key="sk-x", model="whisper-1", client=client)

    out = await transcriber.transcribe(b"RIFF....", mime_type="audio/wav")

    assert out == "Odysseus sailed home."
    # Uses the dedicated transcriptions endpoint, not a chat-completions call —
    # OpenAI treats plain speech-to-text as a distinct API from chat/vision.
    client.audio.transcriptions.create.assert_awaited_once_with(
        model="whisper-1", file=("audio.wav", b"RIFF....", "audio/wav")
    )


@pytest.mark.parametrize(
    ("mime_type", "expected_filename"),
    [
        ("audio/wav", "audio.wav"),
        ("audio/x-wav", "audio.wav"),
        ("audio/mpeg", "audio.mpeg"),
        ("audio/mp3", "audio.mp3"),
        ("audio/mp4", "audio.mp4"),
        ("audio/m4a", "audio.m4a"),
        ("audio/x-m4a", "audio.m4a"),
        ("audio/webm", "audio.webm"),
        ("audio/ogg", "audio.ogg"),
    ],
)
async def test_openai_transcriber_names_the_upload_with_a_real_extension(
    mime_type: str, expected_filename: str
) -> None:
    # OpenAI's transcriptions endpoint infers the audio format from the
    # uploaded filename's extension, not the multipart content-type — an
    # extensionless name (the pre-fix behavior) fails every upload with
    # "Invalid file format" regardless of what mime_type says.
    client = _fake_transcription_client("hello")
    transcriber = OpenAIWhisperTranscriber(api_key="sk-x", client=client)

    await transcriber.transcribe(b"data", mime_type=mime_type)

    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["file"][0] == expected_filename


# --- hosted vision ----------------------------------------------------------- #


async def test_openai_vision_describer_returns_description() -> None:
    client = _fake_chat_client("A photo of an open book.")
    describer = OpenAIVisionDescriber(api_key="sk-x", model="gpt-4.1-mini", client=client)

    out = await describer.describe(b"\x89PNG\r\n", mime_type="image/png")

    assert out == "A photo of an open book."
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4.1-mini"
    # The image rides as a base64 data URI in the chat payload.
    content = kwargs["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)


# --- local vision via Ollama (OpenAI-compatible, no real key) ---------------- #


async def test_ollama_vision_describer_targets_base_url_without_a_key() -> None:
    client = _fake_chat_client("A book cover.")
    describer = OpenAIVisionDescriber(
        model="llava", api_key="ollama", base_url="http://ollama:11434/v1", client=client
    )

    out = await describer.describe(b"\x89PNG\r\n", mime_type="image/png")

    assert out == "A book cover."


def test_build_image_describer_wires_ollama_base_url_and_placeholder_key() -> None:
    # Ollama's endpoint needs no real key, but the SDK client requires some
    # string — "ollama" is the same placeholder convention as `api/llm.py`.
    local = build_image_describer(
        _settings(vision_provider="ollama", ollama_base_url="http://ollama:11434/v1")
    )
    assert isinstance(local, OpenAIVisionDescriber)
    assert str(local._client.base_url).rstrip("/") == "http://ollama:11434/v1"
    assert local._client.api_key == "ollama"


def test_build_image_describer_uses_configured_ollama_api_key_when_set() -> None:
    # OLLAMA_API_KEY set ⇒ Ollama Cloud's hosted vision models, real key used.
    local = build_image_describer(
        _settings(
            vision_provider="ollama",
            ollama_base_url="https://ollama.com/v1",
            ollama_api_key="sk-cloud-key",
        )
    )
    assert local._client.api_key == "sk-cloud-key"


# --- factories --------------------------------------------------------------- #


def test_build_transcriber_selects_provider() -> None:
    assert isinstance(
        build_transcriber(_settings(transcription_provider="openai", openai_api_key="sk-x")),
        OpenAIWhisperTranscriber,
    )


def test_build_transcriber_huggingface_without_extra_raises() -> None:
    # The local Whisper model loads at construction (like the embedder), so the
    # missing-extra guard fires when the factory builds it — not on first use. The
    # 'local extra' message confirms it took the HuggingFace branch.
    with pytest.raises(ProviderConfigError, match=r"local. extra"):
        build_transcriber(_settings(transcription_provider="huggingface"))


def test_build_transcriber_requires_key_for_hosted() -> None:
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_transcriber(_settings(transcription_provider="openai"))


def test_build_image_describer_selects_provider() -> None:
    hosted = build_image_describer(_settings(vision_provider="openai", openai_api_key="sk-x"))
    assert isinstance(hosted, OpenAIVisionDescriber)

    # The local option is the same class pointed at Ollama's endpoint, no key.
    local = build_image_describer(
        _settings(vision_provider="ollama", ollama_base_url="http://ollama:11434/v1")
    )
    assert isinstance(local, OpenAIVisionDescriber)


def test_build_image_describer_requires_key_for_hosted() -> None:
    with pytest.raises(ProviderConfigError, match="OPENAI_API_KEY"):
        build_image_describer(_settings(vision_provider="openai"))


# --- local transcription guard when the ML stack is absent ------------------- #


def test_local_transcriber_errors_without_the_local_extra() -> None:
    # The pipeline is built at construction, so the missing-extra guard fires there
    # (a warm-up build surfaces a bad local config at boot, not mid-turn).
    with pytest.raises(ProviderConfigError, match=r"local. extra"):
        HuggingFaceTranscriber(model="openai/whisper-base")


async def test_local_transcriber_uses_injected_pipeline() -> None:
    # An injected pipeline exercises the local transcriber without the ML stack:
    # construction skips the heavy load and inference is offloaded to a thread.
    def fake_pipe(audio: bytes) -> dict:
        return {"text": "hello world"}

    transcriber = HuggingFaceTranscriber(model="openai/whisper-base", pipe=fake_pipe)
    assert await transcriber.transcribe(b"audio", mime_type="audio/wav") == "hello world"
