"""Unit tests for the API's LangChain chat-model factory.

No network: LangChain chat models construct without making API calls, so we build
real instances and assert on type/config. Settings use ``_env_file=None`` so the
local ``.env`` never leaks in.
"""

import pytest
from api.llm import (
    build_answer_models,
    build_chat_model,
    build_resilient_chat_model,
    build_resilient_structured_model,
    context_window_for,
    model_id_for,
    model_label,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable, RunnableWithFallbacks
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from shared.core.config import Settings
from shared.providers.errors import ProviderConfigError


class _Decision(BaseModel):
    """A tiny schema to drive structured-output construction in tests."""

    ok: bool


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.mark.unit
def test_build_chat_model_anthropic() -> None:
    model = build_chat_model(_settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-5"
    # Raw model retains bind_tools for the agent to use.
    assert hasattr(model, "bind_tools")


@pytest.mark.unit
def test_build_chat_model_openai() -> None:
    model = build_chat_model(_settings(llm_provider="openai", openai_api_key="k"))
    assert isinstance(model, ChatOpenAI)


@pytest.mark.unit
def test_build_chat_model_ollama_uses_base_url_no_key() -> None:
    settings = _settings(llm_provider="ollama", ollama_base_url="http://ollama:11434/v1")
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "http://ollama:11434/v1"
    assert model.openai_api_key.get_secret_value() == "ollama"  # local placeholder


@pytest.mark.unit
def test_build_chat_model_ollama_uses_configured_api_key_when_set() -> None:
    settings = _settings(
        llm_provider="ollama",
        ollama_base_url="https://ollama.com/v1",
        ollama_api_key="sk-cloud-key",
    )
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_key.get_secret_value() == "sk-cloud-key"


@pytest.mark.unit
def test_build_chat_model_cheap_tier_and_override() -> None:
    # Cheap tier default.
    cheap = build_chat_model(
        _settings(llm_provider="anthropic", anthropic_api_key="k"), tier="cheap"
    )
    assert cheap.model == "claude-haiku-4-5-20251001"
    # Explicit override wins.
    overridden = build_chat_model(
        _settings(llm_provider="anthropic", anthropic_api_key="k", llm_model="custom-x")
    )
    assert overridden.model == "custom-x"


@pytest.mark.unit
def test_build_chat_model_missing_key_raises() -> None:
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        build_chat_model(_settings(llm_provider="anthropic", anthropic_api_key=None))


@pytest.mark.unit
def test_resilient_model_without_fallbacks_is_not_wrapped() -> None:
    # No fallbacks configured → a retry runnable, not a fallback chain.
    runnable = build_resilient_chat_model(_settings(llm_provider="openai", openai_api_key="k"))
    assert not isinstance(runnable, RunnableWithFallbacks)


@pytest.mark.unit
def test_resilient_model_with_fallbacks_composes_chain() -> None:
    runnable = build_resilient_chat_model(
        _settings(
            llm_provider="anthropic",
            anthropic_api_key="k",
            openai_api_key="k2",
            llm_fallback_providers="openai",
        )
    )
    assert isinstance(runnable, RunnableWithFallbacks)
    assert len(runnable.fallbacks) == 1


@pytest.mark.unit
def test_unknown_fallback_provider_raises() -> None:
    with pytest.raises(ProviderConfigError, match="fallback provider"):
        build_resilient_chat_model(
            _settings(
                llm_provider="anthropic", anthropic_api_key="k", llm_fallback_providers="nope"
            )
        )


@pytest.mark.unit
def test_resilient_structured_model_wraps_the_base_model() -> None:
    # Regression: with_structured_output is a BaseChatModel method, so it must wrap
    # the raw model *before* retry/fallback. Building it on a resilient runnable
    # (which lacks with_structured_output) used to raise AttributeError.
    runnable = build_resilient_structured_model(
        _settings(llm_provider="openai", openai_api_key="k"), _Decision, tier="cheap"
    )
    assert isinstance(runnable, Runnable)
    # No fallbacks configured → a retry runnable, not a fallback chain.
    assert not isinstance(runnable, RunnableWithFallbacks)


@pytest.mark.unit
def test_resilient_structured_model_composes_fallbacks() -> None:
    runnable = build_resilient_structured_model(
        _settings(
            llm_provider="openai",
            openai_api_key="k",
            ollama_base_url="http://ollama:11434/v1",
            llm_fallback_providers="ollama",
        ),
        _Decision,
    )
    assert isinstance(runnable, RunnableWithFallbacks)
    assert len(runnable.fallbacks) == 1


@pytest.mark.unit
def test_build_answer_models_returns_raw_primary_and_no_fallbacks_by_default() -> None:
    # Returned raw so the graph can bind_tools before composing resilience; with no
    # LLM_FALLBACK_PROVIDERS configured the fallback list is empty.
    primary, fallbacks = build_answer_models(_settings(llm_provider="openai", openai_api_key="k"))
    assert isinstance(primary, ChatOpenAI)
    assert hasattr(primary, "bind_tools")  # still tool-bindable (not yet wrapped)
    assert fallbacks == []


@pytest.mark.unit
def test_build_answer_models_returns_raw_fallback_models_in_order() -> None:
    primary, fallbacks = build_answer_models(
        _settings(
            llm_provider="anthropic",
            anthropic_api_key="k",
            openai_api_key="k2",
            ollama_base_url="http://ollama:11434/v1",
            llm_fallback_providers="openai,ollama",
        )
    )
    assert isinstance(primary, ChatAnthropic)
    # Raw (un-wrapped) fallbacks, in the configured order, each tool-bindable.
    assert [type(m) for m in fallbacks] == [ChatOpenAI, ChatOpenAI]
    assert all(hasattr(m, "bind_tools") for m in fallbacks)


# --- trace-span labeling helpers ---------------------------------------------- #


@pytest.mark.unit
def test_model_id_for_resolves_the_active_provider_and_tier() -> None:
    settings = _settings(llm_provider="anthropic", anthropic_api_key="k")
    assert model_id_for(settings, "default") == "claude-sonnet-5"
    assert model_id_for(settings, "cheap") == "claude-haiku-4-5-20251001"


@pytest.mark.unit
def test_model_id_for_honors_an_explicit_override() -> None:
    settings = _settings(llm_provider="anthropic", anthropic_api_key="k", llm_model="custom-x")
    assert model_id_for(settings, "default") == "custom-x"


@pytest.mark.unit
def test_model_label_reads_provider_and_model_off_a_raw_chat_model() -> None:
    model = build_chat_model(_settings(llm_provider="anthropic", anthropic_api_key="k"))
    assert model_label(model) == "anthropic-chat:claude-sonnet-5"


@pytest.mark.unit
def test_model_label_falls_back_on_an_unrecognized_object() -> None:
    # No `.model`/`.model_name` to read — name defaults to "unknown"; no
    # `._llm_type` either — provider falls back to the type name. Never raises.
    assert model_label(object()) == "object:unknown"


@pytest.mark.unit
def test_model_label_never_raises_when_introspection_errors() -> None:
    class _Explodes:
        @property
        def model(self) -> str:
            raise RuntimeError("boom")

    assert model_label(_Explodes()) == "unknown"


@pytest.mark.unit
def test_context_window_for_resolves_the_configured_per_provider_window() -> None:
    settings = _settings(llm_context_window_anthropic=42_000)
    assert context_window_for(settings, "anthropic") == 42_000
    assert context_window_for(settings, "openai") == settings.llm_context_window_openai
    assert context_window_for(settings, "ollama") == settings.llm_context_window_ollama


@pytest.mark.unit
def test_context_window_for_unknown_provider_raises() -> None:
    with pytest.raises(ProviderConfigError):
        context_window_for(_settings(), "mistral")
