# shared

The library imported by **both** services (`api` and `ingestion`). It holds the
cross-cutting building blocks so the services stay thin and never depend on each
other. Neither service imports the other; `shared` is the only shared code.

## Responsibilities

- **`core`** — `Settings` (Pydantic, env-sourced; secrets as `SecretStr`), structured PII-redacting logging, shared error types, and the password-hashing factory (`build_password_hash`) used by the API's `AuthService` and the bootstrap-admin data migration alike.
- **`models`** — SQLAlchemy ORM models (the relational schema).
- **`db`** — the declarative `Base` + naming convention, the async engine factory, and Alembic migrations (the single schema owner).
- **`repositories`** — data-access classes; every query is scoped by `user_id`.
- **`providers`** — pluggable, config-selected provider abstractions behind `Protocol`s: `Embedder` (OpenAI / Voyage / HuggingFace-local / Ollama), `StorageProvider` (S3 / MinIO), `WebSearchProvider` (Brave / Tavily), with `build_*` factories. (The LLM chat model lives in the API service, on LangChain.)
- **`observability`** — always-on Prometheus metrics (`time_operation`, `record_tokens`, `render_metrics`) and optional, no-op-capable Langfuse tracing (`build_tracer`).
- **`prompt`** — the versioned prompt registry (`get_prompt_registry`, resolve by `name@version`); no inline prompt strings.
- **`ingestion_core`** — pure parse/chunk/embed logic reused by the ingestion service.

## Key entry points

- `shared.core.config.get_settings()` — the process-wide settings singleton.
- `shared.db.create_database_engine(settings)` / `shared.db.Base` — DB engine + ORM base.
- `shared.providers.build_embedder / build_storage_provider / build_web_search_provider`.
- `shared.observability.build_tracer`, `time_operation`, `render_metrics`.
- `shared.prompt.get_prompt_registry()`.

## Test in isolation

```bash
uv run pytest -m unit libs/shared
```

Migrations (from the repo root):

```bash
uv run alembic -c libs/shared/alembic.ini upgrade head   # requires a reachable database
```
