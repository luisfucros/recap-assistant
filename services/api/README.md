# api — Assistant / API service

FastAPI service that fronts the assistant. Latency-sensitive and stateless; it
holds no business logic in routers (router → service → repository/provider) and
enforces auth + per-user scoping in the service/repository layers. The agent
(LangGraph, built in later milestones) runs on LangChain chat models.

## Responsibilities

- HTTP API under `/api/v1` (health today; auth, documents, chat, progress, memory, recommendations as milestones land).
- The LangGraph agent and its tools (retrieval, memory, progress, summarize, web search) with streaming and human-in-the-loop.
- Validating uploads and writing the ingestion outbox event (the actual parse/embed work is the ingestion service's job).

## Key entry points

- **`api.app:create_app`** — the application factory (ASGI target, used with `--factory`).
- **`api.lifespan`** — builds and disposes singletons in a `Resources` container on `app.state` (DB engine, Redis, Qdrant, prompt registry, tracer; lazy storage/embedder).
- **`api.deps`** — FastAPI dependencies to reach those singletons (`DbSession`, `QdrantDep`, `RedisDep`, `PromptsDep`, `EmbedderDep`).
- **`api.llm`** — LangChain chat-model factory (`build_chat_model`, `build_resilient_chat_model`); provider selection + fallbacks/retries.
- **`api.checkpointer`** — the durable LangGraph checkpointer (agent short-term memory) over Postgres via psycopg, keyed by conversation id. `build_pool` yields the app's connection pool; `python -m api.checkpointer` is the one-shot that creates the checkpoint tables (run once, like `alembic upgrade head` — never on replica startup; wired as the `checkpointer-setup` compose service).
- **`api.routers.chat`** — the chat surface: `POST /chat` (non-streaming), `POST /chat/stream` (Server-Sent Events) and `WS /chat/ws` (many turns per connection) both stream tool steps → answer tokens → `done` (or a lone `blocked`, or a lone `interrupt` when a gated tool call pauses for approval) via a shared transport-agnostic frame generator; `POST /chat/{conversation_id}/resume` continues a paused turn with an approve/edit/deny decision; `GET /conversations` + `GET /conversations/{id}/messages` serve history. Turns accept typed text and/or audio/image attachments (`parts`, base64). Delegates to `AgentService` (the run) and `ConversationService` (the transcript).
- **`api.services.multimodal_service`** — `MultimodalNormalizer`: for a chat turn's audio/image attachments, archives each original content-addressed in object storage and derives text (transcript/caption) via the config-selected transcriber/vision providers. The agent's `normalize_input` node folds that text into the message before guardrails, so the pipeline stays text-only.
- **`api.services.memory_service`** — `MemoryService`: writes salient (preference/fact/habit/faq) and page-range summary memories, joining Postgres (content) and the `long_term_memory` Qdrant collection (embedding + filter metadata); semantic/typed/page-range retrieval with the owning `user_id` injected server-side and a spoiler-safe `max_page_end` bound; view/delete for the memory panel.
- **`api.metrics`** — instruments HTTP metrics and exposes `/metrics`.
- **`api.routers.admin`** — `POST /admin/users`, `AdminUser`-gated: create a regular or admin account directly, without self-registration. A first admin is seeded by a one-time, idempotent data migration from `INITIAL_ADMIN_EMAIL`/`INITIAL_ADMIN_PASSWORD` (see `.env.example`) rather than any API route.

## Run in isolation

```bash
uv run uvicorn api.app:create_app --factory --port 8000
# → http://localhost:8000/api/v1/health, /api/v1/docs, /metrics
```

Nothing connects to infrastructure at startup (clients connect lazily), so this
runs without Postgres/Redis/Qdrant up.

## Test in isolation

```bash
uv run pytest services/api          # unit + functional (functional needs no external infra)
```

## Container

```bash
docker build -f services/api/Dockerfile -t recap-api .   # build context = repo root
```
