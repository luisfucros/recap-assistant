"""Application configuration.

A single Pydantic ``Settings`` model, shared by both services via ``libs/shared``,
that reads every value from the environment (or a local ``.env``). Secrets are
typed as ``SecretStr`` so they never render in logs or reprs, and are left
optional so local/dev/test can boot without them; a validator enforces the
security-critical ones only when ``ENVIRONMENT=prod`` — that way a misconfigured
production deploy fails fast instead of running insecurely.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.core.enums import Language

Environment = Literal["local", "dev", "test", "prod"]


class Settings(BaseSettings):
    """Typed application settings sourced from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ------------------------------------------------------- #
    app_name: str = "recap"
    environment: Environment = "local"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    # Fallback language for new users and for documents whose language can't be
    # detected (one of the supported set).
    default_language: Language = Language.EN
    # Default spoiler-safe setting for new users (FR-18). When on, retrieval and
    # summaries are hard-bounded to already-read pages so unread content can't
    # leak; a user can flip it globally or override it per document/per query.
    spoiler_safe_default: bool = True
    # Force-build heavy singletons (and open infra connections) at startup rather
    # than on the first request/task, trading a slower boot for a fast first
    # request. Disable in tests/CI where boot speed matters and infra may be absent.
    warm_up_on_start: bool = True

    # --- Auth / security --------------------------------------------------- #
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 7
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    # Comma-separated list of allowed browser origins; exposed as ``allowed_origins``.
    backend_cors_origins: str = "http://localhost:5173"

    # --- Bootstrap admin (optional) ----------------------------------------- #
    # Seeds one admin user via a one-time, idempotent data migration (run by the
    # `migrate` container). Leave both unset to skip seeding entirely — safe to
    # leave configured permanently across every future `alembic upgrade head`.
    initial_admin_email: str | None = None
    initial_admin_password: SecretStr | None = None

    # --- Google OAuth ------------------------------------------------------ #
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    google_redirect_uri: str | None = None
    # Where to send the browser after a successful OAuth login (the SPA).
    frontend_url: str = "http://localhost:5173"

    # --- PostgreSQL -------------------------------------------------------- #
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/recap"

    # --- Qdrant ------------------------------------------------------------ #
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_chunks_collection: str = "document_chunks"
    qdrant_memory_collection: str = "long_term_memory"

    # --- Redis / Celery ---------------------------------------------------- #
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    # Port the ingestion worker serves its Prometheus /metrics on.
    ingestion_metrics_port: int = 9808

    # --- Uploads ----------------------------------------------------------- #
    # Hard cap on an uploaded document's size, enforced while streaming at the
    # API boundary (default 50 MiB). Bounds memory and storage abuse.
    max_upload_bytes: int = 50 * 1024 * 1024

    # --- Ingestion pipeline ------------------------------------------------ #
    # Structure-aware chunking, measured in whitespace-delimited words (a stable,
    # tokenizer-free proxy for token windows). Overlap carries context across
    # chunk boundaries so a passage split mid-idea is still retrievable.
    chunk_size_words: int = 512
    chunk_overlap_words: int = 64
    # Outbox relay (Celery beat): how often to poll, and how many events per tick.
    outbox_relay_interval_seconds: float = 5.0
    outbox_relay_batch_size: int = 100
    # Bounded exponential backoff for a transiently-failing ingestion task.
    ingest_max_retries: int = 3
    # Beat-scheduled safety net: how often to sweep for, and how stale a document
    # must be, to re-enqueue a `pending`/`processing` row whose ingestion task was
    # never actually observed to finish (e.g. its enqueued message was lost after
    # the outbox marked it dispatched). Comfortably above the time a healthy run
    # (including `ingest_max_retries` backoff) takes, so it never fires on work
    # that's merely still in progress.
    ingest_sweep_interval_seconds: float = 60.0
    ingest_stuck_threshold_seconds: int = 900

    # --- Object storage (S3 / MinIO) --------------------------------------- #
    s3_endpoint_url: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_bucket: str = "recap"
    s3_region: str = "us-east-1"

    # --- LLM provider ------------------------------------------------------ #
    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    # Unset ⇒ local Ollama (no auth, a placeholder key is used). Set to reach
    # Ollama Cloud's hosted models instead (pair with a cloud OLLAMA_BASE_URL).
    ollama_api_key: SecretStr | None = None
    # Model IDs per tier; leave unset to use each provider's built-in defaults.
    llm_model: str | None = None
    llm_model_cheap: str | None = None
    # Resilience: retry attempts on transient errors, and an ordered CSV of
    # provider names to fall back to (e.g. "openai,ollama"). Empty ⇒ no fallback.
    llm_max_retries: int = 2
    llm_fallback_providers: str = ""

    # --- Retrieval --------------------------------------------------------- #
    # Default number of chunks a semantic search returns (before near-duplicate
    # collapse); the agent/tools may request fewer per call.
    retrieval_top_k: int = 8

    # --- Analytics --------------------------------------------------------- #
    # How long a user's computed reading analytics are cached in Redis before a
    # recompute; keeps the hot read/update path cheap (FR-17).
    analytics_cache_ttl_seconds: int = 300
    # How long a user's computed token-spend/tool-call usage summary is cached in
    # Redis before a recompute (NFR-13) — same rationale as analytics above.
    usage_cache_ttl_seconds: int = 300

    # --- Agent compaction (FR-4.1) ----------------------------------------- #
    # Per-provider context windows (tokens) — conservative floors for the model
    # families this project ships against; override if a chosen model's actual
    # window differs. Used only to decide when a session's running token count
    # is "long" enough to auto-compact, not to enforce a hard request limit.
    llm_context_window_anthropic: int = 200_000
    llm_context_window_openai: int = 128_000
    llm_context_window_ollama: int = 128_000
    # Fraction of the active context window at which a session auto-compacts,
    # leaving headroom for the next turn's prompt, tools, and completion.
    compaction_threshold_ratio: float = 0.75

    # --- Agent scratchpad -------------------------------------------------- #
    # TTL for the agent's turn-scoped working memory in Redis (FR-7.8). Kept short
    # — it's ephemeral per-turn notes (plan/findings/questions), distinct from
    # conversation state and long-term memory; it expires soon after the turn.
    scratchpad_ttl_seconds: int = 1800

    # --- Embeddings -------------------------------------------------------- #
    # `ollama` runs on the same OpenAI-compatible endpoint as the local LLM/vision
    # providers (OLLAMA_BASE_URL, no key) — `embedding_model_local` is the pulled
    # Ollama embedding model (e.g. `nomic-embed-text`).
    embeddings_provider: Literal["openai", "voyage", "huggingface", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"  # hosted; local providers use their own field
    embedding_model_local: str = "nomic-embed-text"
    embed_batch_size: int = 64
    # Override the vector dimension when it can't be inferred from the model.
    embedding_dim: int | None = None
    voyage_api_key: SecretStr | None = None

    # --- Multimodal input (FR-19) ----------------------------------------- #
    # Audio→text and image→text run behind pluggable providers, hosted or local,
    # so the whole chat path can run with no hosted API. Hosted uses the OpenAI
    # audio/vision APIs (reusing OPENAI_API_KEY); local uses a HuggingFace Whisper
    # model (audio) / captioning model (image) via `transformers` (offline).
    transcription_provider: Literal["openai", "huggingface"] = "openai"
    transcription_model: str = "whisper-1"  # hosted; local overrides to a HF id
    transcription_model_local: str = "openai/whisper-base"
    # Vision runs on an OpenAI-compatible chat API: hosted OpenAI, or a local Ollama
    # vision model over OLLAMA_BASE_URL (no key, no torch install) — same split as
    # the LLM. `vision_model_local` is the pulled Ollama model (e.g. `llava`).
    vision_provider: Literal["openai", "ollama"] = "openai"
    vision_model: str = "gpt-4.1-mini"  # hosted; local overrides to the Ollama model
    vision_model_local: str = "llava"
    # Hard cap on a single decoded chat attachment (audio clip / image), enforced
    # at the API boundary (default 15 MiB). Bounds memory and provider-cost abuse;
    # documents (much larger) go through the upload path, not chat.
    chat_media_max_bytes: int = 15 * 1024 * 1024

    # --- Web search -------------------------------------------------------- #
    web_search_provider: Literal["brave", "tavily"] = "brave"
    brave_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    # --- Rate limiting ------------------------------------------------------ #
    # Fixed-window request caps, enforced in Redis so the count is shared across
    # every API replica. Auth (register/login/refresh) is keyed per client IP —
    # there's no authenticated user yet, and this is exactly the credential-
    # stuffing/brute-force surface. Chat (a turn-triggering route) is keyed per
    # user — it bounds per-account LLM-cost abuse, not shared-IP traffic.
    rate_limit_auth_max: int = 10
    rate_limit_auth_window_seconds: int = 60
    rate_limit_chat_max: int = 20
    rate_limit_chat_window_seconds: int = 60

    # --- Langfuse (optional tracing) --------------------------------------- #
    langfuse_host: str | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

    @property
    def allowed_origins(self) -> list[str]:
        """CORS origins as a list (parsed from the comma-separated env value)."""
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    @property
    def google_oauth_configured(self) -> bool:
        """True only when the Google OAuth client is fully configured."""
        return bool(
            self.google_client_id and self.google_client_secret and self.google_redirect_uri
        )

    @property
    def tracing_enabled(self) -> bool:
        """True only when all Langfuse credentials are present (else tracing is a no-op)."""
        return bool(self.langfuse_host and self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def is_production(self) -> bool:
        """True when running in the production environment."""
        return self.environment == "prod"

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> "Settings":
        """Fail fast if a production deploy is missing security-critical config."""
        if self.environment != "prod":
            return self
        problems: list[str] = []
        if not self.jwt_secret:
            problems.append("jwt_secret is required in production")
        if not self.allowed_origins or "*" in self.allowed_origins:
            problems.append("backend_cors_origins must be explicit (no wildcard) in production")
        if not self.cookie_secure:
            problems.append("cookie_secure must be true in production")
        if problems:
            raise ValueError("; ".join(problems))
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached; inject as a dependency)."""
    return Settings()
