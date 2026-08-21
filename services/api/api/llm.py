"""LLM chat models for the agent, built from settings.

The agent runs on **LangChain chat models**: LangGraph binds tools to them,
streams their output, and composes fallbacks — so this module returns configured
``BaseChatModel``/``Runnable`` objects rather than a bespoke client. Provider
selection, per-tier model ids, retries, and cross-provider fallbacks are all
config-driven, and Ollama is reached through the OpenAI-compatible client (so one
integration covers OpenAI and local models).

Two entry points, because tool-calling and resilience interact:
- ``build_chat_model`` returns a raw ``BaseChatModel`` that still exposes
  ``bind_tools``. The agent binds tools to it and applies fallbacks over the
  *bound* models (fallbacks must wrap tool-bound models, so that composition
  belongs to the agent).
- ``build_resilient_chat_model`` returns a retry + fallback ``Runnable`` for the
  non-tool calls (complexity classifier, summarizer, guardrail checks).
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from shared.core.config import Settings
from shared.providers._config import require_secret, resolve_ollama_api_key
from shared.providers.errors import ProviderConfigError

ModelTier = Literal["default", "cheap"]
LLMProviderName = Literal["anthropic", "openai", "ollama"]

# Built-in per-tier model ids; overridable via LLM_MODEL / LLM_MODEL_CHEAP.
_DEFAULTS: dict[str, dict[ModelTier, str]] = {
    "anthropic": {"default": "claude-sonnet-5", "cheap": "claude-haiku-4-5-20251001"},
    "openai": {"default": "gpt-5-mini-2025-08-07", "cheap": "gpt-4.1-mini"},
    "ollama": {"default": "llama3.1", "cheap": "llama3.1"},
}

_ANTHROPIC_MAX_TOKENS = 4096


def _model_id(settings: Settings, provider: str, tier: ModelTier) -> str:
    override = settings.llm_model if tier == "default" else settings.llm_model_cheap
    return override or _DEFAULTS[provider][tier]


def model_id_for(settings: Settings, tier: ModelTier) -> str:
    """The resolved model id for ``tier`` on the *active* (``LLM_PROVIDER``) provider.

    Used to label trace spans for nodes that run on a wrapped resilience
    ``Runnable`` (structured-output guardrail/planner) rather than a raw
    ``BaseChatModel``, where :func:`model_label` can't introspect the model.
    """
    return _model_id(settings, settings.llm_provider, tier)


_CONTEXT_WINDOW_SETTINGS: dict[str, str] = {
    "anthropic": "llm_context_window_anthropic",
    "openai": "llm_context_window_openai",
    "ollama": "llm_context_window_ollama",
}


def context_window_for(settings: Settings, provider: str) -> int:
    """The configured context-window size (tokens) for ``provider`` (compaction).

    Used by :class:`~api.services.compaction_service.CompactionService` to decide
    when a session's running token count has grown "long" relative to the
    *active* provider's window, not a fixed number (FR-4.1.2).
    """
    attr = _CONTEXT_WINDOW_SETTINGS.get(provider)
    if attr is None:
        raise ProviderConfigError(f"Unknown LLM provider {provider!r}")
    return getattr(settings, attr)


def model_label(model: BaseChatModel) -> str:
    """Best-effort ``provider:model`` label for a raw chat model, for tracing.

    Different LangChain integrations name their model field differently
    (``ChatAnthropic.model`` vs. ``ChatOpenAI.model_name``), and ``_llm_type`` is
    an informal convention, not a guaranteed API — so this never raises; a
    labeling failure must never break a turn, only make its trace less detailed.
    """
    try:
        name = getattr(model, "model", None) or getattr(model, "model_name", None) or "unknown"
        provider = getattr(model, "_llm_type", type(model).__name__)
        return f"{provider}:{name}"
    except Exception:
        return "unknown"


def _base_chat_model(settings: Settings, provider: str, tier: ModelTier) -> BaseChatModel:
    """Build a single (un-wrapped) LangChain chat model — retains ``bind_tools``."""
    model = _model_id(settings, provider, tier)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=require_secret(
                settings.anthropic_api_key, "ANTHROPIC_API_KEY", "anthropic LLM"
            ),
            max_tokens=_ANTHROPIC_MAX_TOKENS,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=require_secret(settings.openai_api_key, "OPENAI_API_KEY", "openai LLM"),
        )
    if provider == "ollama":
        from langchain_openai import ChatOpenAI

        # Local Ollama ignores the key (a placeholder is used); OLLAMA_API_KEY
        # lets this same client reach Ollama Cloud's hosted models instead.
        return ChatOpenAI(
            model=model,
            api_key=resolve_ollama_api_key(settings.ollama_api_key),
            base_url=settings.ollama_base_url,
        )
    raise ProviderConfigError(f"Unknown LLM provider {provider!r}")


def _fallback_provider_names(settings: Settings) -> list[str]:
    """Parse ``LLM_FALLBACK_PROVIDERS`` (CSV) into an ordered, de-duplicated list.

    The primary provider is dropped, and each name is validated so a typo fails
    fast rather than silently disabling fallback.
    """
    names: list[str] = []
    for raw in settings.llm_fallback_providers.split(","):
        name = raw.strip().lower()
        if not name or name == settings.llm_provider or name in names:
            continue
        if name not in _DEFAULTS:
            raise ProviderConfigError(f"Unknown LLM fallback provider {name!r}")
        names.append(name)
    return names


def build_chat_model(settings: Settings, *, tier: ModelTier = "default") -> BaseChatModel:
    """Return the primary chat model for ``tier`` (raw; supports ``bind_tools``)."""
    return _base_chat_model(settings, settings.llm_provider, tier)


def build_answer_models(
    settings: Settings, *, tier: ModelTier = "default"
) -> tuple[BaseChatModel, list[BaseChatModel]]:
    """Return the raw primary answer model and its raw per-provider fallbacks.

    Returned **raw** (un-wrapped) on purpose: the agent must ``bind_tools`` to each
    model *before* composing retry/fallbacks, because ``bind_tools`` is a
    ``BaseChatModel`` method that the retry/fallback ``Runnable`` wrappers don't
    expose — so the order has to be bind-then-wrap. Composing the fallback chain
    therefore belongs to the graph (which knows the per-turn tools, see
    :func:`~api.agent.graph.build_agent_graph`); this factory only resolves the
    primary + ordered fallback providers from settings.
    """
    primary = build_chat_model(settings, tier=tier)
    fallbacks = [
        _base_chat_model(settings, name, tier) for name in _fallback_provider_names(settings)
    ]
    return primary, fallbacks


def build_resilient_chat_model(settings: Settings, *, tier: ModelTier = "default") -> Runnable:
    """Return a retry + fallback chat runnable for non-tool calls.

    The primary and each configured fallback get ``with_retry`` (transient-error
    resilience); fallbacks are tried in order if the primary keeps failing. Not
    for tool-calling paths — bind tools first, then compose fallbacks there.
    """
    attempts = 1 + settings.llm_max_retries
    primary = build_chat_model(settings, tier=tier).with_retry(stop_after_attempt=attempts)
    fallbacks = [
        _base_chat_model(settings, name, tier).with_retry(stop_after_attempt=attempts)
        for name in _fallback_provider_names(settings)
    ]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_resilient_structured_model(
    settings: Settings, schema: type[BaseModel], *, tier: ModelTier = "default"
) -> Runnable:
    """Return a retry + fallback runnable that emits a validated ``schema`` instance.

    Structured output must be applied to each *raw* chat model **before** resilience
    is composed: ``with_structured_output`` is a ``BaseChatModel`` method, while the
    retry/fallback wrappers expose only ``Runnable`` methods (calling it on a
    resilient model raises ``AttributeError``). So we build
    ``base.with_structured_output(schema).with_retry(...)`` per provider, then
    compose cross-provider fallbacks over those — mirroring
    :func:`build_resilient_chat_model`, but for the structured-output nodes
    (guardrail judge, planner).
    """
    attempts = 1 + settings.llm_max_retries

    def structured(provider: str) -> Runnable:
        return (
            _base_chat_model(settings, provider, tier)
            .with_structured_output(schema)
            .with_retry(stop_after_attempt=attempts)
        )

    primary = structured(settings.llm_provider)
    fallbacks = [structured(name) for name in _fallback_provider_names(settings)]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
