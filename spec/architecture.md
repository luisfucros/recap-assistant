# Architecture — Recap

> System design for the personalized Agentic RAG reading companion. Ideas that thread through the design: (1) **reading position drives summaries and long-term memory**; (2) **providers are pluggable behind interfaces** (LLM, embeddings, storage, web search) so the same code runs fully hosted (Claude/OpenAI/S3/Brave) or fully local (Ollama/HuggingFace/MinIO); (3) it is a **microservice system** — the assistant/API and the ingestion pipeline are separate, independently scalable services; and (4) **observability is split into always-on metrics (Prometheus/Grafana) and optional tracing (Langfuse)**.

## 1. Architectural overview

A **microservice** system: two independently deployable, independently scalable backend services — the **Assistant/API service** (FastAPI + LangGraph agent, plus a **dedicated eval Celery worker** on the same image) and the **Ingestion service** (Celery workers for parse/chunk/embed) — sharing a common library and communicating **asynchronously** through Postgres (source of truth + transactional outbox), the Celery broker (Redis), object storage (S3/MinIO), and Qdrant (vectors). No synchronous calls between the two: the API commits an upload + outbox event; the ingestion service drains the queue and writes chunks/vectors back; the API reads status from Postgres. Evaluation scoring is an API-owned background job (FR-12.5): it needs the agent graph, so it must not run on ingestion workers. Every service exposes `/metrics` (Prometheus) and health; **Grafana** dashboards them. LLM/RAG tracing (Langfuse) is **optional** — absent credentials, it is a no-op.

```
                          ┌─────────────────────────────────────────────┐
   Browser (React SPA)    │      ASSISTANT / API service (FastAPI)        │
  ┌──────────────────┐    │  /api/v1/*  ·  /metrics  ·  stateless, scaled │
  │  Auth  Library    │   │  Routers ─► Services ─► Repositories          │
  │  Chat (SSE/WS)    │◄──┼─► AgentService (LangGraph: guardrails, tools, │
  │  Progress  Recs   │   │     HITL, streaming)                          │
  │  Admin (evals)    │   │  Eval Celery worker (queue `eval`, same image) │
  └──────────────────┘   │  IngestionService = validate+store+outbox ONLY│
          ▲              └───┬──────────┬───────────┬──────────┬──────────┘
          │ httpOnly         │          │           │          │
          │ cookies          ▼          ▼           ▼          ▼
          │        ┌──────────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
          │        │  PostgreSQL  │ │ Qdrant │ │ Redis  │ │ Storage  │
          │        │ users/docs/  │ │ vectors│ │ broker │ │ S3/MinIO │
          │        │ progress/mem/│ │ + meta │ │ +cache │ │  (files) │
          │        │ outbox       │ └────────┘ └───┬────┘ └────┬─────┘
          │        └──────┬───────┘                │ Celery    │
          │       outbox relay ─────────────────► broker       │
          │               ▼                        ▼           ▼
          │        ┌───────────────────────────────────────────────────┐
          │        │      INGESTION service (Celery workers)  /metrics   │
          │        │ parse → chunk → embed(batched) → upsert(Qdrant)     │
          │        │ → persist chunks → status=indexed (atomic, outbox)  │
          │        └───────────────────────────────────────────────────┘
          │
   ┌──────┴───────────────── shared providers (both services) ───────────────────┐
   │ LLM (LangChain, in api): Claude / OpenAI / Ollama   Embedder: OpenAI·Voyage/HF │
   │ StorageProvider: MinIO / AWS S3   WebSearch: Brave / Tavily (HITL-gated)      │
   └───────────────────────────────────────────────────────────────────────────┘

   Observability (scrapes/receives from both services):
     Prometheus ──► Grafana (latency, throughput, CPU/mem, retrieval/ingest timings)
     Langfuse (OPTIONAL — no creds ⇒ no-op): LLM/tool/embed/retrieval/guardrail traces
```

## 2. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| API framework | FastAPI (async) | Routes under `/api/v1/`; delegate to services |
| Language | **Python 3.13** (latest patch, e.g. 3.13.14) | One minor behind 3.14; LangGraph/Celery/SQLAlchemy don't yet support 3.14. Type hints mandatory. `requires-python = ">=3.13,<3.14"` |
| Agent orchestration | LangGraph | Tool-calling graph + checkpointer for session state |
| **LLM (pluggable)** | Claude · OpenAI · **Ollama (OpenAI-compatible client)** | **LangChain chat models** (`langchain-anthropic`/`langchain-openai`) in the API service; selected by `LLM_PROVIDER`; retries + `.with_fallbacks()` |
| **Embeddings (pluggable)** | OpenAI / Voyage (hosted) · **HuggingFace / sentence-transformers (local)** | Behind `Embedder`; selected by `EMBEDDINGS_PROVIDER`; dim read from provider |
| Relational DB | PostgreSQL 17 | Source of truth; outbox table |
| ORM/driver | SQLAlchemy 2.0 async + Alembic | Migrations |
| Vector DB | Qdrant | Named vectors + metadata payload filters |
| Cache / broker / sessions | Redis 7 | Celery broker + cache + short-term memory backing |
| Task queue | **Celery** (Redis broker) | Drives the separate ingestion service; beat for periodic outbox relay |
| **Object storage (pluggable)** | **MinIO (local) / AWS S3 (prod)** | Behind `StorageProvider` (S3 API) |
| **Web search (pluggable)** | **Brave / Tavily** | Behind `WebSearchProvider`; `web_search` tool (HITL-gated); `WEB_SEARCH_PROVIDER` |
| Auth | JWT (httpOnly cookies) + Google OAuth | 15-min access / 7-day refresh |
| Prompt management | Langfuse Prompts (versioned registry) | Named + versioned; no inline prompts |
| Metrics | **Prometheus + Grafana** | `/metrics` per service; latency/throughput/CPU/mem/retrieval timings; always on |
| Tracing / eval (optional) | **Langfuse** | Traces LLM/tools/embeddings/retrieval/guardrails; eval runs. **No creds ⇒ no-op** |
| Guardrails | Topical + safety classifiers (LLM/rules) | Input, topical-relevance, appropriateness, output-sanitize |
| HITL | LangGraph `interrupt` + checkpointer | Approve web search / recommend; confirm page ranges |
| Frontend | React + Vite + TypeScript | Vitest + RTL |
| Streaming | SSE (primary) + WebSocket (interactive) | Tool events + token stream + HITL prompts |
| Containerization | Docker + Compose (`watch` for dev) | Per-service named volumes |

## 3. Provider abstractions (pluggable, config-selected)

Embeddings, storage, and web search sit behind `Protocol` interfaces in `libs/shared/providers`, constructed once at startup from `Settings` and injected via DI. Swapping a provider is a config change, not a code change. **The LLM is deliberately *not* one of these shared protocols** — see below.

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    # slices `texts` into `batch_size` calls (hosted API or local `encode`);
    # sentence-transformers' own batch_size does not bound peak memory
    async def embed(self, texts: list[str], *, batch_size: int | None = None) -> list[list[float]]: ...

class StorageProvider(Protocol):        # S3 API — MinIO local, AWS S3 prod
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...

class WebSearchProvider(Protocol):      # Brave or Tavily
    async def search(self, query: str, *, count: int = 5) -> list[SearchResult]: ...

class Transcriber(Protocol):            # audio → text (hosted OpenAI Whisper API / local HuggingFace Whisper)
    async def transcribe(self, audio: bytes, *, mime_type: str) -> str: ...

class ImageDescriber(Protocol):         # image → text (hosted OpenAI vision / local Ollama vision model)
    async def describe(self, image: bytes, *, mime_type: str) -> str: ...
```

- **LLM (LangChain chat models, in the API service):** the agent runs on LangGraph, which binds tools to and streams from **LangChain chat models** — so rather than a bespoke `LLMProvider`, `services/api/api/llm.py` builds a configured `BaseChatModel` (`ChatAnthropic` / `ChatOpenAI`; Ollama = `ChatOpenAI` at `OLLAMA_BASE_URL`). `LLM_PROVIDER` selects the backend and `model_tier` (`default`/`cheap`) → model id (overridable via `LLM_MODEL`/`LLM_MODEL_CHEAP`). `build_chat_model` returns the raw tool-bindable model; `build_resilient_chat_model` adds `.with_retry()` + `.with_fallbacks()` (config: `LLM_MAX_RETRIES`, `LLM_FALLBACK_PROVIDERS`) for non-tool calls. LangChain lives only in the API service, keeping `shared`/ingestion free of it.
- **Embeddings:** `OpenAIEmbedder`, `VoyageEmbedder`, `HuggingFaceEmbedder` (sentence-transformers, runs locally). Must be **multilingual** across the supported languages (FR-16.5) so cross-lingual retrieval works (query and chunks in different languages share one space); the configured hosted/local model is chosen accordingly. The Qdrant collection is created with `Embedder.dim`; changing provider ⇒ re-embed (see §6).
- **Storage:** one S3 client library (`aioboto3`) with an endpoint/credentials switch — MinIO endpoint locally, AWS S3 in prod. Buckets/keys identical across environments.
- **Multimodal input (FR-19):** `Transcriber` (audio→text) and `ImageDescriber` (image→text) normalize non-text chat input to text before the agent sees it, selected by `TRANSCRIPTION_PROVIDER` / `VISION_PROVIDER`. Transcription is **hosted (OpenAI Whisper API)** or **local (`HuggingFaceTranscriber` — OpenAI's Whisper model via `transformers`, running offline)**, the same hosted/local split as the embedder; vision is **hosted (OpenAI vision API)** or **local (an Ollama vision model such as `llava` over Ollama's OpenAI-compatible endpoint at `OLLAMA_BASE_URL`)** — the same OpenAI-compatible client the local LLM uses, so the local vision path needs no `torch`/`transformers` install. **Embeddings stay text-only** (FR-19.3): audio/image is transcribed/described first, so the vector store is single-modality — no image/audio vectors, one embedding space. Original media is stored via `StorageProvider`; the derived text is what's reasoned over and optionally embedded.
- **Web search:** `BraveSearchProvider` and `TavilySearchProvider`, selected by `WEB_SEARCH_PROVIDER`; both normalize to a common `SearchResult` shape so the `web_search` tool is provider-agnostic.

## 4. Code layout & service boundaries (binding)

A **monorepo** with one shared library and two deployable services. Within each service the dependency direction is strict — **routers/tasks → services → repositories/providers**. No business logic in routers; authorization + user-scoping enforced in services and repositories.

```
libs/shared/                 # imported by BOTH services (no service imports the other)
├── core/                    # config (Pydantic Settings), security, logging, DI helpers
├── models/                  # SQLAlchemy ORM + dataclasses for internal structs
├── db/                      # SINGLE schema owner: alembic.ini, env.py, versions/  (migrations live with the models)
├── repositories/            # data access; every query scoped by user_id
├── providers/               # Embedder, StorageProvider, WebSearchProvider + factories (no LLM; see api/llm.py)
├── observability/           # Prometheus metrics + Langfuse tracing (OPTIONAL/no-op) helpers
├── prompt/                  # PromptService: versioned registry + rendering (no inline prompts)
└── ingestion_core/          # parsers/ (Strategy+Factory), chunker, batched embed step (pure logic)

services/api/                # ASSISTANT / API service (FastAPI)
├── main.py                  # app factory, lifespan (build providers + heavy singletons ONCE)
├── api/v1/                  # routers: auth · documents · chat · progress · recommendations · memory · evaluations
├── schemas/                 # Pydantic request/response models (external contract)
├── services/                # auth · ingestion (validate+store+outbox ONLY) · agent · retrieval
│                            #   · progress · memory · recommendation · guardrail · evaluation
├── llm.py                   # LangChain chat-model factory (build_chat_model + fallbacks/retry)
├── agent/                   # LangGraph graph, nodes, tools, guardrails, hitl, checkpointer
├── clients/                 # web search (Brave/Tavily), misc external boundaries
├── evaluation/              # datasets/ (versioned), scorers/ (retrieval, faithfulness, judge)
└── metrics.py               # /metrics exposition + HTTP instrumentation

services/ingestion/          # INGESTION pipeline service (Celery)
├── celery_app.py            # Celery app + config (Redis broker)
├── tasks.py                 # ingest_document task: parse→chunk→embed→upsert→persist→status (atomic)
├── outbox_relay.py          # beat-scheduled: poll outbox → enqueue Celery tasks
└── metrics.py               # /metrics exposition (via a small sidecar/exporter) + task metrics
```

- **Why the split:** the API is latency-sensitive and stateless; ingestion is CPU/memory-heavy (parsing, local embedding models) and bursty. Separating them lets ingestion scale on worker count and crash/restart without affecting chat/upload availability (FR-15). They never call each other — all coordination is via Postgres/outbox + the Celery broker + Qdrant + storage.
- **The API's `IngestionService` only** validates, stores the file, and writes the `documents` row + outbox event. It does **not** parse/embed — that is exclusively the ingestion service's job.

**DI / startup:** heavy objects (LLM client, embedder weights, Qdrant/S3 clients, DB engine, prompt registry, metrics registry) are constructed once — in the API via FastAPI `lifespan`, in the ingestion service via Celery worker-init signals — never module-level singletons (load heavy classes once at startup).

## 5. Data model (PostgreSQL)

Source of truth. **All user-owned tables carry `user_id` and every repository query filters by it** — cross-tenant access is impossible by construction (FR-6.4).

- **users** — `id (uuid)`, `email`, `hashed_password (nullable for OAuth-only)`, `google_sub (nullable)`, `display_name`, `preferred_language (Language enum, default en)`, `spoiler_safe (bool, default true)`, `created_at`.
- **documents** — `id`, `user_id`, `title`, `author`, `filename`, `object_key (S3/MinIO, content-addressed)`, `content_sha256`, `format`, `language (Language enum, detected at ingestion, user-overridable)`, `status (pending|processing|indexed|failed)`, `failure_reason`, `page_count`, `embed_model (which embedder produced current vectors)`, `created_at`, `indexed_at`. **Unique constraint `(user_id, content_sha256)`** enforces per-user duplicate rejection (FR-1.10).
- **chunks** — `id`, `document_id`, `user_id`, `ordinal`, `page_start`, `page_end`, `chapter`, `section`, `text`, `token_count`, `vector_id`. (Text is the source of truth here; vectors live in Qdrant keyed by `vector_id` and can be regenerated.)
- **reading_progress** — `id`, `user_id`, `document_id`, `current_page`, `last_summarized_page`, `status (not_started|reading|completed|cancelled)`, `spoiler_safe (nullable bool — per-document override of the user default)`, `last_accessed_at`. Unique `(user_id, document_id)`. `last_summarized_page` is the recap high-water mark (FR-3.1).
- **reading_events** — append-only activity trail powering analytics (FR-17): `id`, `user_id`, `document_id`, `type (position_advanced|status_changed|session|completed)`, `from_page (nullable)`, `to_page (nullable)`, `occurred_at`. Indexed by `(user_id, occurred_at)`; never updated, only inserted — so pace/streaks/history are derivable and auditable.
- **conversations** / **messages** — chat history; messages carry `role`, `content`, `tool_calls (jsonb)`.
- **long_term_memory** — `id`, `user_id`, `type (preference|summary|concept|fact|habit|faq)`, `content`, `document_id (nullable)`, `page_start (nullable)`, `page_end (nullable)`, `embedding_id (nullable, Qdrant)`, `created_at`. **Summary-type memories carry the document + page range they cover** (FR-4.3), so recaps retrieve by position.
- **outbox** — reliable-messaging table driving ingestion + memory-index events.

**Language enum (FR-16).** A shared `Language` enum — `en`, `es`, `de`, `fr`, `it` (ISO 639-1) — persisted as a native Postgres enum type and reused by `users.preferred_language` and `documents.language`. A user's chat language and a document's language are independent (a user may read a book in another language); the value is a small fixed set, so it is an enum, not a reference table.

**Schema migrations (Alembic) — single owner, applied as a one-shot job.** Because both services share `libs/shared/models`, the schema has exactly one owner: `libs/shared/db/` holds `alembic.ini` + `versions/`, co-located with the models so a model change and its migration land together (autogenerate diffs against the shared metadata). Migrations are **not run on service startup** — multiple services and replicas booting `alembic upgrade head` would race on the DB and `alembic_version`. Instead a dedicated short-lived **`migrate`** container runs `alembic upgrade head` to completion and exits; `api`, `ingestion`, and `ingestion-beat` gate on it via `depends_on: condition: service_completed_successfully`. In production the same migrate image runs as a pre-deploy/init job (e.g. a k8s Job) before rollout. Exactly one component ever writes the schema; the app/worker containers only read it.

## 6. Document ingestion pipeline

Decoupled from the request path — and into its **own service** — for responsiveness, reliability, and independent scaling.

1. **Upload** (`POST /api/v1/documents`, API service): `IngestionService` validates type/size and **computes `content_sha256` while streaming** the bytes to content-addressed object storage (`<user_id>/sha256/<hash>.<ext>`). It then inserts a `documents` row (`status=pending`, `embed_model=<active>`, hash) **`ON CONFLICT (user_id, content_sha256) DO NOTHING`** and writes a `DocumentUploaded` **outbox** event — in one DB transaction. If the insert conflicts (the user already has this content), no row/event is created and the API returns **`409 DUPLICATE_DOCUMENT`** with the existing document id. The unique constraint makes this **race-safe**: of two concurrent identical uploads, exactly one wins and the other 409s. The API's work ends here.
2. **Outbox relay** (ingestion service, Celery beat) polls pending outbox events and enqueues a Celery task per event. Guarantees a committed document gets exactly one task across crashes (at-least-once + idempotent task).
3. **Celery task** (`services/ingestion/tasks.py`): `status→processing` → **parse** (format Strategy via `ParserFactory`, PDF first: text + page/chapter/section/title/author) → **detect language** from the parsed text and store `documents.language` (FR-16.3; map to the supported `Language` enum, fall back to a default + flag when unknown/unsupported, user-overridable) → **chunk** structure-aware (record `page_start`/`page_end`) → **embed in configurable batches** via the active **multilingual** `Embedder` (batch size bounds memory; essential for local HF models — see below) → **upsert** to Qdrant with metadata payload (incl. `language`) → persist `chunks` → `status=indexed` (or `failed` + reason).
4. **Atomic status, outbox-protected (FR-1.7.1):** the terminal `status=indexed` transition and the `chunks` insert commit **together in one Postgres transaction**, and only *after* the Qdrant upsert succeeded. If Qdrant/storage/embedding-API connection fails mid-run, the transaction is not committed and the job retries — the document is never left showing `indexed` with missing vectors, and no `DocumentIndexed` outbox event (which downstream memory-indexing consumes) is emitted for a partial result. **Wrong info is never logged on a connection error.**
5. **Batched embeddings:** `EMBED_BATCH_SIZE` (setting) caps how many chunks are embedded per call. Batches are processed sequentially with per-batch retry; a failed batch retries without reprocessing succeeded batches. Prevents OOM on large PDFs and local sentence-transformers models.
6. **Retries**: transient failures retry with backoff; exhausted → dead-letter + `failed`. Worker is **idempotent** (re-run replaces the document's points/chunks).
7. **Re-embedding** (provider switch, FR-8.3): a maintenance job re-reads `chunks.text`, re-embeds with the new provider, replaces Qdrant points, and updates `documents.embed_model`. Text never has to be re-parsed.

> Why outbox: removes the dual-write race between the Postgres commit and the queue publish, and guarantees terminal status + downstream events reflect only durably-persisted results (NFR-6, FR-1.7.1).

## 7. Vector store (Qdrant)

Two collections, **both carrying `user_id` in the payload**, and **every search filters by `user_id`** (mandatory isolation — a query can never return another user's vectors). This is enforced in `RetrievalService`/`MemoryService`, not left to the caller: the `user_id` filter is injected from the authenticated context, never from tool arguments the LLM controls.

- **`document_chunks`** (doc-ingestion vectors), size = `Embedder.dim`. Payload: **`user_id`**, `document_id`, `page_start`, `page_end`, `chapter`, `section`, `ordinal`, `title`, `author`, `content_hash`. Powers page/chapter-specific extraction from the user's books/docs. **Read-range scoping** (FR-2.3): retrieval defaults to `page_end <= current_page` for the target document, so answers/summaries don't leak unread content unless the user opts in.
- **`long_term_memory`** (memory vectors), separate collection. Payload: **`user_id`**, `type (preference|summary|concept|fact|habit|faq)`, `document_id (nullable)`, `page_start`/`page_end (nullable)`. Powers semantic recall of user preferences, summaries of current readings (by page range), and important facts the user shared.
- **Retrieval de-dup (FR-1.12):** `document_chunks` results are collapsed by `content_hash` (and a near-duplicate score threshold) before being handed to the agent, so even if a duplicate slipped through, the same passage is never presented twice in one answer.

## 8. Agentic assistant (LangGraph)

The chat request builds/loads a per-conversation graph and runs a plan→act→observe loop. **Reading progress is fetched early** so downstream tools scope to the read range.

**Three knowledge sources the agent tool-calls into** (all user-scoped from the authenticated context — never from LLM-supplied `user_id`):

1. **Relational DB (Postgres) — reading state.** Structured, exact. The agent knows which books/docs the user is **currently reading**, which are **completed**, and which are **cancelled**, plus the current page and last-accessed times. Used for "what am I reading?", personalization, and read-range scoping.
2. **Vector DB — `long_term_memory`.** Semantic. The agent recalls the user's **general, lasting preferences/habits/facts** (not the current book or this sitting's recap range), **summaries of readings by page range**, and **important facts the user shared** across sessions.
3. **Vector DB — `document_chunks`.** Semantic + metadata-filtered. The agent extracts information from **specific pages/chapters** of the user's ingested docs/books.

- **Nodes:** `normalize_input → guardrail_in → load_progress → load_memories → plan → extract_memory → generate ⇄ tools → persist_memory → guardrail_out → compact`. `normalize_input` (FR-19) transcribes audio / describes images into text so every downstream node — guardrails included — sees text only. A cheap-tier model classifies query complexity to skip tools for trivial requests; like `guardrail_in`, it sees a **short prior user/assistant slice** so anaphoric follow-ups can still plan retrieval. `extract_memory` runs *before* generate so a personal fact the reader just shared is saved and injected as a system note for this turn's answer (the model must not ask whether to save it). Confirming a page-range summary (FR-4.6) is a UI interrupt after the recap; resume appends a short canned ack rather than re-running generate. Every node's prompt is pulled by name+version from the prompt registry, and every node/tool/LLM/embedding/retrieval call is wrapped in a Langfuse span.
- **Tools** (thin wrappers over services, user-scoped via injected context):
  - `get_reading_progress(document_id?)` → ProgressService. **Source 1 (DB).** Reading list + statuses (`reading | completed | cancelled | not_started`), current page, last-summarized page, recently accessed.
  - `query_long_term_memory(query, type?, document_id?, page_range?)` → MemoryService. **Source 2 (memory vectors).** Preferences, reading summaries by page range, remembered facts.
  - `retrieve_chunks(query, filters)` → RetrievalService. **Source 3 (doc vectors).** Page/chapter-filtered semantic search, read-range default; **spoiler-safe makes `page_end <= current_page` a hard filter** (FR-18.3), not just a default.
  - `summarize(document_id, range)` → cheap-tier summary over retrieved range (e.g. `last_summarized_page..current_page`).
  - `web_search(query)` → WebSearchProvider (Brave/Tavily). **`requires_approval=True`** → HITL-gated (FR-13).
  - `recommend()` → RecommendationService. **`requires_approval=True`** on its external-API path (a purely internal similarity recommendation is not gated).
- **Isolation invariant:** for tools hitting either vector collection, `RetrievalService`/`MemoryService` inject the `user_id` payload filter from the request context; the LLM cannot widen or spoof the scope. A missing/empty `user_id` filter is a hard error, not an unfiltered search.
- **Guardrails (`guardrail_in` / `guardrail_out`, FR-10):** input node runs prompt-injection detection + PII redaction, a **topical-relevance** check (is this about the user's reading?) and an **appropriateness/safety** check; off-topic or unsafe turns short-circuit to a polite refusal **in the reader's language**. The topical judge sees the current message plus a **short prior user/assistant slice** (not tool payloads or the full checkpoint) so anaphoric follow-ups stay on-topic. Voice notes and images about reading stay on-topic — they are how the reader asks, not a request to process media as a document. Output node sanitizes model text (XSS-safe) **and runs the spoiler check** (below). Full PII/secrets never sent to hosted providers (local Ollama/HF keep data on-prem). Each decision is traced with its reason. The answer model is also bound to what the tools can actually do: it must not suggest workarounds the product does not support (copy-pasting document text into chat, recapping an attached transcript/image as if it were a book), and it must not ask the reader to save a recap or a personal fact — those writes are owned by `persist_memory` (UI confirm) and `extract_memory` (automatic).
- **Spoiler-safe mode (FR-18):** effective setting = per-document `reading_progress.spoiler_safe` if set, else `users.spoiler_safe` (default on), overridable per-query. When on, protection is **layered across sources**: `retrieve_chunks` hard-filters to `page_end <= current_page`, `query_long_term_memory` bounds summaries by their `page_end`, and `guardrail_out` runs a **spoiler check** that catches ahead-of-position content leaking from `web_search` or the model's own knowledge — redacting/refusing it. If a request genuinely needs content past the current page, the graph **interrupts to warn and ask for opt-in** (HITL) rather than silently spoiling or returning nothing (FR-18.4).
- **Human-in-the-loop (FR-13):** implemented with LangGraph `interrupt()` + the checkpointer. The gate is driven by a **declared per-tool `requires_approval` property**, not a hard-coded list in the graph: the tool node checks the flag on the tool the planner/model chose and interrupts *before* executing it. A tool is **consequential** (and sets the flag) when it reaches beyond the user's own stored data — external egress, external cost/rate-limited API, or a side effect the user should authorize. The read tools (`get_reading_progress`, `retrieve_chunks`, `summarize`, `query_long_term_memory`) are read-only and user-scoped, so they never interrupt; a new outward-reaching tool inherits the gate just by setting its flag.
  - *Approval interrupts:* before executing any `requires_approval` tool — presently `web_search` and the external path of `recommend` — the graph interrupts with the proposed action; the run pauses (checkpointed) and **resumes** on approve/edit/deny.
  - *Ask-about-pages interrupts (FR-4.6/4.7):* a distinct interrupt class — before `persist_memory` saves a page-range summary, the agent interrupts to **confirm the page range** (proposing the read-range default); resume writes the summary and appends a short canned ack to the already-streamed recap (no second generate). When a query implies prior reading but no memory/progress exists for that document, it interrupts to **ask which pages the user has read** instead of guessing.
- **Progress → summary → memory loop:** when the user has advanced (`current_page > last_summarized_page`) or asks for a recap, the agent retrieves the newly-read span, summarizes it, and (after the page-range confirmation above) `persist_memory` writes a **summary memory tied to `(document_id, page_start, page_end)`**, then advances `last_summarized_page`. Future recaps hit the saved summary instead of re-reading (FR-3.4, FR-4.3).
- **Language handling (FR-16):** the turn resolves a **target answer language** = the user's `preferred_language`, passed to the `generate` and `guardrail_in` prompts as a rendered variable (one prompt, not one per language). The assistant **answers in that language even when the document is in another** (e.g. English question over a German book); verbatim quotes/citations stay in the document's original language, translated only on request. Guardrail refusals (the judge's `reason`, plus the canned injection/off-topic fallbacks that never reach the LLM) use the same language — they must not stay English when the chat does not. Retrieval is **cross-lingual** — the multilingual embedder places query and chunks in one space so a query in the user's language matches chunks in the document's language; `language` also rides in the chunk payload for optional filtering/labeling.
- **Short-term memory:** LangGraph **checkpointer** (Postgres/Redis) persists conversation state so sessions resume, follow-ups have context, and interrupted (HITL) runs can be resumed. Long histories are **auto-compacted** (see below).
- **Agent scratchpad (turn-scoped working memory, FR-7.8):** a `ScratchpadService` holds the agent's **plan, running findings, and open questions** for a turn **outside the model context window** — keyed by `(user_id, conversation_id, turn_id)`, **Redis-backed** and TTL'd (ephemeral). The `plan` node writes the initial plan; each tool-observation step appends findings/open questions; `generate` pulls back **only the relevant slices** (by recency/relevance), not the whole scratchpad. This keeps long multi-step research turns (e.g. "trace this arc across ch. 1–10") from bloating the context or tripping compaction prematurely. It is distinct from short-term conversation state (checkpointer) and long-term memory (vectors); salient conclusions can be promoted to `long_term_memory`. User-scoped like every store, and traced as its own span.
- **Structured node outputs (FR-7.9):** the agent's **internal** nodes emit **schema-validated JSON** via `.with_structured_output(<PydanticModel>)` (tool/function-calling under the hood) — one typed schema per task: planner (`{complexity, needs_tools, tool_plan[]}`), guardrail-in (`{on_topic, safe, reason}`), memory-classify (`{type, salient, page_range?}`), page-range interrupt (`{page_start, page_end, proposal_reason}`), evaluation (`{faithfulness, relevance, citation_ok, …}`). This makes routing deterministic and unit-testable. **The final user-facing answer is the deliberate exception** — natural-language streamed tokens with citations, never JSON-wrapped (JSON mode would break token streaming).
- **Auto-compaction (token-budget driven, FR-4.1.1–4.1.3):** short-term memory is bounded relative to the **active model's context window**, not a fixed turn count. A `CompactionService` maintains a running **token count** per session (provider-aware tokenizer — Anthropic/`tiktoken` for OpenAI-compatible, a documented heuristic for local models) updated after each turn. When the count crosses `COMPACTION_THRESHOLD_RATIO × context_window` (default `0.75`; both the per-model window and the ratio are config, leaving headroom for the next prompt+tools+completion), the graph runs a compaction step: a **cheap-tier model** summarizes the history via the `compaction@version` registry prompt into a **concise-but-sufficient** summary — enough to continue seamlessly (what the user is doing, decisions/answers already given, current document/page focus, any open HITL interrupt, unresolved threads) while dropping verbatim turns. The checkpoint is then **rewritten**: history is replaced by that summary as the seed context (optionally keeping the last few turns verbatim for continuity), and the token count resets. Compaction is **idempotent** and safe across the checkpointer; salient facts it surfaces can be promoted to `long_term_memory`. It is transparent to the user (no action required) and traced as its own span.

### Streaming

`POST /api/v1/chat/stream` returns **SSE**; a **WebSocket** variant supports bidirectional control (and is the natural fit for HITL, where the client must reply). Ordered event stream — structured tool/guardrail/interrupt events precede the token stream:

```
event: tool_call     data: {"tool":"get_reading_progress","args":{...}}
event: tool_result   data: {"tool":"get_reading_progress","summary":"page 84 / ch.5"}
event: tool_call     data: {"tool":"retrieve_chunks","args":{"page_end":84}}
event: interrupt     data: {"type":"confirm_pages","proposed":{"page_start":60,"page_end":84},"prompt":"Save a summary of pages 60–84?"}
                     # client replies (approve/edit/deny) → run resumes from checkpoint
event: token         data: {"delta":"Before this point, "}
...
event: done          data: {"message_id":"...","citations":[...],"trace_id":"lf_..."}
```

Guardrail blocks and HITL denials emit `event: blocked` / `event: interrupt` with a clear message; `done` carries the Langfuse `trace_id` for inspection.

## 9. API surface (`/api/v1`, all async)

Standard error shape `{ "detail": "...", "code": "SNAKE_CASE_CODE" }`; list endpoints paginate `{ items, total, page, page_size }` (default 10, max 100). Every endpoint has an OpenAPI docstring.

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register` · `/auth/login` · `/auth/refresh` · `/auth/logout` | Email/password + cookie sessions |
| GET | `/auth/google/login` · `/auth/google/callback` | Google OAuth |
| POST | `/documents` | Upload PDF → `{id, status:pending}` (async ingest); `409 DUPLICATE_DOCUMENT` if the user already has identical content |
| GET | `/documents` · `/documents/{id}` · `/documents/{id}/status` | List / detail / poll status |
| DELETE | `/documents/{id}` | Delete doc + chunks + vectors + object |
| POST | `/chat` · `/chat/stream` (SSE) | Agent turn (non-stream / streamed tool events + tokens); accepts **multimodal input** (text/audio/image) normalized to text (FR-19) |
| WS | `/chat/ws` | Interactive WebSocket session |
| GET | `/conversations` · `/conversations/{id}` | History |
| GET/PATCH | `/users/me` | Profile incl. `preferred_language` and `spoiler_safe` default (FR-16/FR-18) |
| GET/PUT | `/progress/{document_id}` | Read/update position & status (may trigger summary memory); per-document `spoiler_safe` override |
| GET | `/progress` | Reading list + recently accessed |
| GET | `/analytics` | Reading analytics: pace, streaks, pages-over-time, started/completed/cancelled (FR-17) |
| GET | `/recommendations` | Explainable recommendations |
| GET/DELETE | `/memory` | View / delete long-term memories (privacy) |
| POST | `/chat/{conversation_id}/resume` | Resume a HITL-interrupted run (approve / edit / deny) |
| POST | `/admin/users` | Create a regular or admin account (admin only; public self-registration cannot set `is_admin`) |
| POST | `/evaluations/run` | Enqueue an eval run against a named dataset (**202** + `pending` row; admin). Scoring is not on this request. |
| GET | `/evaluations` · `/evaluations/{id}` · `/evaluations/datasets` | List runs (newest first) / fetch one run / list shipped dataset names+versions (admin) |

## 10. Cross-cutting concerns

- **Config:** Pydantic `Settings` from env only; `.env.example` documents every var — DB/Qdrant/Redis URLs, `CELERY_BROKER_URL`, **storage** (`STORAGE_ENDPOINT`, S3 keys, bucket), **`LLM_PROVIDER`** + per-provider keys/base-urls (incl. `OLLAMA_BASE_URL`), **`EMBEDDINGS_PROVIDER`** + keys/model, `EMBED_BATCH_SIZE`, **`WEB_SEARCH_PROVIDER`** + `BRAVE_API_KEY`/`TAVILY_API_KEY`, **short-term-memory compaction** (`LLM_CONTEXT_WINDOW` per active model, `COMPACTION_THRESHOLD_RATIO` default `0.75`), **`SCRATCHPAD_TTL_SECONDS`** (agent turn scratchpad expiry), **`DEFAULT_LANGUAGE`** (default `en`; fallback for new users and undetected document language), **multimodal input** (`TRANSCRIPTION_PROVIDER`, `VISION_PROVIDER` + per-provider keys/base-urls), **`SPOILER_SAFE_DEFAULT`** (default `true`), **eval worker** (`EVAL_STUCK_THRESHOLD_SECONDS` — re-enqueue a `pending`/`running` evaluation run past this age), **Langfuse** (`LANGFUSE_HOST`, keys — *optional*), Prometheus/Grafana ports, JWT secret, Google OAuth client. Missing Langfuse keys ⇒ tracing no-op.
- **Auth dependency:** verifies the JWT on every protected route and yields the current user; services re-check ownership.
- **Isolation:** enforced at repository (SQL `WHERE user_id=`) and Qdrant (payload filter) layers — defense in depth, UI-independent.
- **Observability:** structured JSON logs (no tokens/PII), request-id tracing, metrics (ingestion throughput, query latency, per-user token spend, tool-call counts), **plus Langfuse LLM/RAG traces** (§13). Tracing is best-effort — a Langfuse outage never fails a user request.
- **Prompts:** all prompts resolved via `PromptService` (name+version) — no inline prompt strings (§13).
- **Security headers & CORS:** explicit origins; `X-Content-Type-Options`, `X-Frame-Options`, HSTS; HTTPS everywhere.

## 11. Prompt management

`PromptService` is the single source of prompts, backed by a **versioned registry** (Langfuse Prompts; an in-repo YAML/JSON store is the offline fallback). Prompts are addressed by `name` + `version`/label (e.g. `system@prod`, `summarizer@v3`) and rendered with typed variables. No service or agent node contains an inline prompt string. The resolved prompt `name@version` is attached to the Langfuse span for the LLM call so an eval regression or bad answer can be traced to an exact prompt version. Rolling a prompt forward/back is a registry change, not a deploy.

## 12. Guardrails

`GuardrailService` runs as the graph's `guardrail_in`/`guardrail_out` bookends (FR-10):

- **Input:** prompt-injection detection, PII detection/redaction before any hosted-provider call.
- **Topical relevance:** a cheap-tier classifier decides whether the request concerns the user's reading/documents; off-topic requests get a polite redirect **in the reader's language** and never reach the tools/LLM answer path. The judge is given a bounded prior-turn slice (user/assistant text only) so follow-ups like "what about him?" are classified in context. Voice notes and images about reading are on-topic (they are how the reader asks).
- **Appropriateness/safety:** unsafe/abusive content is blocked with a standard refusal in that same language.
- **Output:** sanitize model text (escape/scrub) to prevent prompt-injection-driven XSS before it streams to the client.

Every decision (`allow`/`redirect`/`block` + reason) is a traced span; blocks return a clear, non-leaky message and short-circuit the graph.

## 13. Observability — metrics (Prometheus/Grafana) + tracing (Langfuse, optional)

Two independent pillars. **Metrics are always on; tracing is optional.**

**Metrics (Prometheus + Grafana, FR-14).** Both services expose a `/metrics` endpoint. The API uses an ASGI instrumentator for HTTP **latency histograms**, **throughput**, in-flight requests, and error rates per route/status; the ingestion service exports Celery **task duration/throughput**, queue depth, and batch sizes. A shared `observability/metrics` module also records RAG timings as custom Prometheus metrics — **retrieval time**, embedding time, LLM-call latency, tool latency, tokens per request — plus process **CPU/memory** (process collector; host metrics via node-exporter). Labels are low-cardinality (`service`, `route`, `outcome`) — never raw user PII. Prometheus scrapes both services; **Grafana** dashboards cover API latency/throughput, resource usage, and RAG/ingestion timings, with alert rules in hardening (§16 / M8).

**Tracing (Langfuse, FR-11 — optional).** The `observability/tracing` module wraps a turn in a **Langfuse** trace: child spans for each LLM call (model, provider, prompt `name@version`, tokens, latency), tool call, embedding call, retrieval (query, filters, returned chunk ids + scores), and guardrail/HITL decision. Ingestion is traced too (parse/chunk/embed/upsert timings, batch sizes). Traces correlate to `user_id`/`conversation_id` without raw PII. **If no Langfuse credentials are configured the tracer is a no-op** (the factory returns a null tracer) — the app runs identically, no errors (FR-11.5). Even when enabled it is best-effort and non-blocking: a Langfuse outage is swallowed. When enabled, the trace id is returned to the client on `done`.

## 14. Evaluation service

`EvaluationService` runs the retrieval + agent pipeline against a **versioned, custom dataset** (`evaluation/datasets/`) of `question → {expected chunks / reference answer / expected behavior}` cases (FR-12). Scorers (`evaluation/scorers/`):

- **Retrieval:** hit-rate / recall / MRR of relevant chunks for a query.
- **Answer quality:** groundedness/faithfulness to retrieved context, relevance, citation correctness — with an **LLM-as-judge** scorer option.

**Trigger vs execute (FR-12.5).** Scoring is too slow and LLM-heavy to hold an HTTP replica. `enqueue_evaluation` loads the dataset (404 if unknown — **no row**), inserts an `EvaluationRun` as `pending` (prompt/model/embedding tags already filled so a waiting row is comparable), **commits**, then enqueues a Celery task with the run id on queue **`eval`**. `POST /evaluations/run` returns **202** with that row. `execute_evaluation` (worker) flips `pending` → `running`, seeds fixtures / scores as today, then writes `completed` or `failed` (+ `results` / `summary` / `error`). The client **polls** `GET /evaluations` / `GET /evaluations/{id}`; eval runs are not streamed. `GET /evaluations/datasets` lists shipped artifacts so the UI does not hard-code names.

**Why an API-owned worker, not ingestion.** The run calls `AgentService` (LangGraph, tools, judge LLM). LangChain stays out of `shared`/ingestion; the services never HTTP-call each other. Ingestion Celery is parse/chunk/embed on a different resource profile. Eval therefore gets a **second Celery app** in `services/api` (same `CELERY_BROKER_URL`, queue `eval`, `task_acks_late`, prefetch 1) and a compose service using the **API image** with `celery -A … worker --queues=eval`. Ingestion workers must not subscribe to `eval`. Commit-then-delay can lose the enqueue if Redis is down; a beat sweep on *this* app re-enqueues stuck `pending`/`running` rows (mirrors `sweep_stuck_documents`). Execute is a no-op on an already-terminal row.

CI / `python -m api.evaluation.run` enqueue then wait for a terminal status (or `--sync` for in-process); still exit non-zero only on `failed`, never on a low score. Runs stay tagged for Langfuse + the persisted row so two runs remain comparable across prompt/model/embedding versions (FR-12.3). Datasets stay versioned artifacts (FR-12.4).

## 15. Human-in-the-loop (HITL)

Built on LangGraph `interrupt()` + the checkpointer (FR-13).

**What triggers an approval — the rule, not a list.** Each tool declares a `requires_approval: bool` property (default `False`). The tool node checks it on the chosen tool and interrupts **before** executing any tool where it is `True`. A tool sets the flag when it is **consequential** — it acts beyond the user's own stored data: **external egress** (sends the reader's text/query to a third party), **external cost / rate-limited API usage**, or a **side effect** the user should authorize. This keeps the gate a property of *the tool*, so read-only user-scoped tools run freely and any future outward-reaching tool is gated the moment it's registered — no edit to the agent's routing. (This mirrors the isolation invariant's spirit: safety is enforced structurally, not by remembering to special-case each call site.)

Two interrupt classes:

1. **External-action approval** (`requires_approval` tools) — before executing e.g. `web_search` or the external-API path of `recommend`, the graph emits an `interrupt` with the proposed query/action and pauses (state checkpointed). The client replies approve / edit / deny via `POST /chat/{conversation_id}/resume` (or over WS); the run resumes from the checkpoint. On deny, the tool is skipped and the agent continues (or explains it couldn't complete the step) rather than calling it anyway.
2. **Page-range confirmation** — before persisting read-progress info to long-term memory the agent confirms the covered page range (proposing the read-range default, FR-4.6); when a query implies prior reading but no memory/progress exists it asks which pages were read (FR-4.7).

All interrupts and their resolutions are traced. Because state is checkpointed, an interrupted run survives disconnects and process restarts.

## 16. Deployment (Docker Compose)

Services:
- **`migrate`** — one-shot: runs `alembic upgrade head` (from `libs/shared/db`) and exits; the single component that writes the schema. `api`/`ingestion`/`ingestion-beat`/`eval` use `depends_on: condition: service_completed_successfully`.
- **`api`** — Assistant/API service (FastAPI); scales on replicas.
- **`eval`** — Celery worker for evaluation runs (FR-12.5); **same API image**, queue `eval`, concurrency 1. Scales independently of `api` and of `ingestion`.
- **`eval-beat`** — Celery beat for the stuck-eval-run sweep (single instance). Never share `ingestion-beat`.
- **`ingestion`** — Celery worker(s) for the ingestion pipeline; scales independently on worker count/replicas.
- **`ingestion-beat`** — Celery beat running the outbox relay (single instance).
- **`postgres`**, **`adminer`** (DB admin UI), **`qdrant`**, **`redis`** (Celery broker + cache), **`minio`**, **`frontend`**.
- **`prometheus`** + **`grafana`** (+ `node-exporter`) — always on; scrape `api`, `ingestion`, and `eval` `/metrics`; Grafana ships provisioned dashboards.
- **`langfuse`** (+ its own Postgres) — **optional**, behind a compose profile; if not run / no creds, tracing is a no-op.
- **`ollama`** — optional profile for the fully-local model configuration.

`api` and `ingestion` are **separate images** built from the same monorepo (shared `libs/`). In production, `minio` is replaced by AWS S3 (no container; `StorageProvider` points at the S3 endpoint), Langfuse can be managed cloud or self-hosted, and Prometheus/Grafana can be managed. Each stateful service gets a named volume; `depends_on` uses `condition: service_healthy`; only necessary ports exposed; secrets injected via env; `docker compose watch` for dev reload.

## 17. Production deployment (AWS Terraform)

**Milestone 10** implements a complete AWS infrastructure-as-code solution via **Terraform**, replacing Docker Compose with managed AWS services for production scalability, durability, and operational safety. The app code is **unchanged** — all deployment differences are infrastructure-level.

```
                          Internet (HTTPS / CloudFront)
                                    ▼
                          ┌─────────────────┐
                          │   CloudFront    │ (CDN, cache, WAF-ready)
                          └────────┬────────┘
                                   ▼
                          ┌─────────────────────┐
                          │   AWS ALB + TLS     │
                          │ (Application Load   │
                          │  Balancer, routing  │
                          │  to ECS task groups)│
                          └──────┬──────────────┘
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌─────────────────────────────────────────────────────┐
              │          ECS Fargate Cluster (Private Subnets)              │
              │  ┌────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐ ┌──────┐ │
              │  │  API   │ │  Eval  │ │ Eval     │ │ Ingestion │ │Ing.  │ │
              │  │ (HTTP) │ │ worker │ │ beat (1) │ │ workers   │ │beat  │ │
              │  └────────┘ └────────┘ └──────────┘ └───────────┘ └──────┘ │
              │  one-shot: migrate (recap-migrate) + checkpointer-setup    │
              │            (recap-api image, different command)            │
              └────┬─────────────────────┬────────────────────┬─────┘
                   │                     │                    │
         ┌─────────▼──────┐  ┌──────────▼────────┐  ┌────────▼────────┐
         │  RDS Aurora    │  │  ElastiCache      │  │  Single-Instance│
         │  PostgreSQL    │  │  (Redis 7)        │  │  Qdrant on      │
         │  Multi-AZ      │  │  Multi-AZ         │  │  Fargate + EBS  │
         │  automated     │  │  (broker + cache) │  │  + S3 snapshots │
         │  backups       │  └───────────────────┘  └─────────────────┘
         └────────────────┘
                    ▲
            ┌───────┴────────┐
            ▼                ▼
        ┌────────┐                       ┌─────────┐
        │   S3   │                       │   ECR   │
        │buckets │                       │ 3 repos │
        │(docs,  │                       │ (shared │
        │backups)│                       │ images) │
        └────────┘                       └─────────┘
```

### Services

**Managed databases / brokers (AWS):**
- **RDS Aurora PostgreSQL** (multi-AZ primary + read replica) replaces the Postgres container; automated backups retained 7 days, encrypted at rest, accessible only from the private VPC.
- **ElastiCache for Redis** (multi-AZ, automatic failover) replaces the Redis container; the Celery broker, cache, and session store. A small number of nodes covers both the low-throughput cache and the bursty broker load.
- **S3 buckets:** document storage (`<account-id>-recap-documents`), Qdrant EBS volume snapshots (`<account-id>-recap-qdrant-backups`).
- **ECR (Elastic Container Registry):** private repositories follow Dockerfiles (one repo per image), not Compose/ECS service names. `recap-api` (`services/api/Dockerfile`) is pulled by the HTTP API, eval worker, eval-beat, and the one-shot checkpointer-setup task; `recap-ingestion` (`services/ingestion/Dockerfile`) by the ingestion worker and ingestion-beat; `recap-migrate` (`docker/migrate.Dockerfile`) by the one-shot Alembic job only — checkpointer-setup cannot reuse it (the migrate image is shared-only and has no psycopg/LangGraph). Each of those is still a distinct ECS task definition (command, sizing, desired count). Scan-on-push, immutable tags (git SHA), lifecycle expiry of stale images. An ECR **pull-through cache** fronts public registries so Qdrant/Prometheus/Grafana tasks never pull Docker Hub at runtime. The frontend is not an image in production (S3 + CloudFront).
- **Config is env-only** (matches NFR-9 and the app's existing `Settings` path). Sensitive values are Terraform `sensitive` variables from a gitignored tfvars / `TF_VAR_*`, injected as ECS task `environment` entries — **not** AWS Secrets Manager or Parameter Store. Those services bill per secret and are the wrong cost profile for a learning deploy; RDS must not use `manage_master_user_password` (it auto-creates a billed SM secret). Trade-off: values appear in the task definition and Terraform state for anyone with those permissions; swapping to Secrets Manager later is an injection-site change, not an app change.

**Application services (ECS Fargate, private subnets):**
- **API service** — stateless FastAPI app, 2+ task replicas by default, scales on CloudWatch CPU/memory metrics. Each task definition specifies memory (512 MB min, e.g. 1024 MB for spare headroom), CPU (0.25–1 vCPU), log routing to CloudWatch, and a long-running HTTP container. Health checks via ALB target group every 30 s.
- **Eval workers** — same `recap-api` image as a Celery consumer on queue `eval` (FR-12.5); dedicated task definition + service, 1+ replicas, scaled independently of HTTP API tasks and of ingestion. Do not run eval on ingestion task definitions.
- **Eval beat** — same `recap-api` image; dedicated task definition, `desired_count=1` (stuck-run sweep). Do not share the ingestion beat.
- **Ingestion workers** — `recap-ingestion` image; stateless Celery worker; 1+ replicas, auto-scales from 1 to 5+ based on SQS `ApproximateNumberOfMessages` (via `aws-autoscaling` service). Each task similar to API (memory, CPU, logs).
- **Ingestion beat** — same `recap-ingestion` image; dedicated task definition, `desired_count=1` (outbox relay; no auto-scale). Do not share eval-beat.
- **One-shot Migrate job** — `recap-migrate` image; ECS task definition (not a service) that runs `alembic upgrade head` as a pre-deploy step; the main task-definition launch plan waits for its completion (via CloudWatch event / SNS / Lambda, or manual orchestration depending on deploy approach).
- **One-shot Checkpointer-setup job** — `recap-api` image with command `python -m api.checkpointer` (not `recap-migrate` — that image has no psycopg/LangGraph). Idempotent; runs after Migrate and before the first API/eval task starts.

**Qdrant (single instance):**
A **single-task Qdrant service on Fargate with EBS-backed persistent storage**. The task definition attaches an **EBS volume** (allocated to that task only; AWS now supports EBS on Fargate) and mounts it to `/qdrant/storage`. Qdrant runs behind an **internal Network Load Balancer** (NLB) — not exposed to the internet — for service discovery and zero-downtime restarts (the ALB routes chat requests to the API; the API's Qdrant client discovers the NLB endpoint via DNS). **No auto-scaling** (a single instance to preserve statefulness). **Daily snapshots** of the EBS volume are taken (via AWS Backup or a Lambda-scheduled EC2 CreateSnapshot call if the volume is attached to a Fargate task — AWS documentation for Fargate + EBS details the snapshot mechanism) and stored in S3 with a **7-day retention policy**. On catastrophic loss, a new Qdrant task can mount a restored snapshot and resume.

> *(Note: A production, high-availability Qdrant deployment would require Qdrant's **cluster mode** — multiple interconnected Qdrant instances with consensus and rebalancing — or a switch to a managed vector-DB service (e.g., Pinecone, Weaviate Cloud). Single-instance + backup is appropriate for M10 and the current product stage, with a documented upgrade path to cluster mode when scale demands HA.)*

### Networking & security

- **VPC** with public subnets (ALB, NAT) and private subnets (ECS, RDS, ElastiCache, Qdrant). No direct internet access from application services; outbound traffic via NAT gateway.
- **Security groups:** ALB allows inbound on ports 80/443; ECS services allow inbound from ALB only; RDS/ElastiCache/Qdrant allow inbound from ECS only. Egress is granular (outbound HTTPS to hosted API endpoints, no arbitrary internet access).
- **HTTPS everywhere:** ALB terminates TLS (certificate via ACM), routes to task HTTP; tasks communicate internally via HTTP (no need for per-task TLS overhead).
- **Config injection:** database credentials, API keys, and signing keys are Terraform-supplied env vars on the Fargate task (gitignored tfvars, never committed, never baked into images). They **are** present on the task definition — acceptable for this learning deploy; do not add Secrets Manager "just in case."

### Data durability & disaster recovery

- **RDS:** automated backups (7-day retention), point-in-time restore, cross-region snapshots (optional for HA). Multi-AZ deployment provides automatic failover.
- **Qdrant EBS snapshots:** daily snapshots stored in S3 with a 7-day retention policy. Recovery involves restoring a snapshot to a new EBS volume and attaching it to a new Qdrant task. Documented runbook for snapshot restore.
- **Transactional outbox invariants hold:** the Postgres→Qdrant dual-write race is still mediated by the outbox; RDS durability doesn't change this contract.
- **Ingestion idempotency:** re-running a task (on a crash or retry) re-upsets vectors and chunks; Qdrant's upsert semantics and the Postgres atomic status transition guarantee no wrong state.

### Observability at scale

- **CloudWatch logs:** all services log to CloudWatch (no need for a separate ELK/Loki stack locally); log groups per **task** (`/recap/api`, `/recap/eval`, `/recap/eval-beat`, `/recap/ingestion`, `/recap/ingestion-beat`, `/recap/migrate`, `/recap/checkpointer-setup`, `/recap/qdrant`), with a **retention policy** (default 30 days). Shared images still get separate log groups so a beat or eval failure is not buried in the HTTP API stream.
- **Prometheus + Grafana:** both services continue exporting `/metrics` to Prometheus. A managed Prometheus workspace (AWS Managed Service for Prometheus, optional) or a self-hosted Prometheus in EC2 scrapes the endpoints. Grafana (managed or self-hosted) displays the same dashboards as Docker Compose.
- **CloudWatch dashboards:** CPU, memory, network, errors, and request latency per service; RDS and ElastiCache metrics; Qdrant task state and EBS volume usage.
- **Alarms:** on API error rate > 5%, ingestion task failure, RDS CPU > 80%, Qdrant task state mismatch.
- **Langfuse tracing:** unchanged — no Langfuse credentials ⇒ no-op; with credentials, traces flow to managed Langfuse or self-hosted on EC2.

### Deployment & CI/CD

- **Terraform modules** organize infrastructure by concern: `vpc`, `ecr`, `rds`, `elasticache`, `ecs` (shared task definition factory), `ecs_api_service` (HTTP API + eval worker + eval-beat + checkpointer-setup — all pull `recap-api`; migrate is a separate one-shot on `recap-migrate`), `ecs_ingestion_service` (worker + beat, both `recap-ingestion`), `qdrant_ecs`, `alb`, `s3`, `cloudwatch`, `iam`. No `secrets` module — config is variables → task env. **No state-backend module** — remote state uses an already-created S3 (+ lock table) backend the operator supplies via `terraform init -backend-config`; this stack never creates the state bucket or DynamoDB lock table.
- **Docker images:** three images (`api`, `ingestion`, `migrate`) covering seven app containers (api, eval, eval-beat, checkpointer-setup, ingestion, ingestion-beat, migrate), built via CI, tagged with the git SHA, and pushed to the matching ECR repository; each ECS task definition references the URI of the image it shares. GitHub Actions authenticates to ECR with OIDC (no long-lived keys).
- **Deploy process:** `terraform plan` → review → `terraform apply`. Migrate then checkpointer-setup run as pre-deploy one-shot tasks before rolling out new API, eval, and ingestion revisions.
- **Blue-green deployments:** ALB target groups enable rolling updates: new task revisions are started with health checks; traffic drains from old tasks; old tasks are terminated. Zero-downtime deploys.
- **Terraform workspaces** or separate tfvar files support staging / production environments with config overrides (instance counts, CPU/memory, backup retention).

### Local development unchanged

Docker Compose (§16) remains the dev/test environment. No Terraform/AWS knowledge required to work on the codebase locally.

---

**Implementation:** See **Milestone 10** in `spec/tasks.md` for detailed tasks: Terraform module structure (against a pre-existing remote state backend), VPC/networking, ECR repositories + pull-through cache, RDS setup, Qdrant + EBS configuration, ElastiCache provisioning, S3 buckets (app data only, not Terraform state), ECS task definitions, ALB/CloudFront, env-based config (no Secrets Manager), IAM policies, CloudWatch monitoring, CI/CD integration, runbooks, and validation checklists.

## 18. Key design decisions (rationale)

1. **Reading position as first-class context** — retrieval, summaries, and memory are bounded/keyed by the read range; prevents spoilers and makes recaps cheap by reusing saved summaries.
2. **Provider interfaces for embeddings, storage, web search; LangChain for the LLM** — the same code runs fully hosted (Claude/OpenAI/S3/Brave) or fully local/private (Ollama/HuggingFace/MinIO). Embeddings/storage/web-search sit behind our own `Protocol`s in `shared`. The **LLM is a LangChain chat model in the API service** (not a bespoke protocol): LangGraph binds tools to and streams from it, and `.with_fallbacks()`/`.with_retry()` give cross-provider failover + transient-error retries for free. Ollama = the OpenAI chat model at a local `base_url`. Keeping LangChain in `api` only leaves `shared`/ingestion lightweight.
3. **Text in Postgres, vectors in Qdrant** — relational source of truth stays authoritative; embedding-provider swaps are a re-embed, not a re-ingest.
4. **Outbox + atomic terminal status** — eliminates the Postgres/queue dual-write race and guarantees no `indexed`/event is recorded on a partial or connection-failed run (no wrong info logged).
4b. **Duplicate rejection via `(user_id, content_sha256)` unique constraint** — dedup enforced at the DB, which also makes concurrent duplicate uploads race-safe (one wins, other 409s) without app-level locking. Content-addressed storage dedups bytes; cross-user content is kept separate for isolation (FR-1.11).
4c. **Migrations: single owner, one-shot job** — schema + Alembic versions live in `libs/shared/db` (with the models both services share); a dedicated `migrate` container applies them once and exits, and services gate on its completion. No service auto-migrates on startup, so replicas/services never race on `alembic_version`.
5. **Batched embeddings** — bounded memory per embed call; the difference between working and OOM on large PDFs / local sentence-transformers models.
6. **Microservices: API vs ingestion** — the two have opposite resource profiles (latency-sensitive/stateless vs CPU-memory-heavy/bursty). Separate services let ingestion scale and fail independently; they coordinate only through Postgres/outbox + broker + Qdrant + storage, never synchronous calls.
7. **Celery for ingestion** — mature task queue with beat scheduling (outbox relay), retries, and dead-letter; the ingestion service is not on the async request path, so Celery's model fits and its ecosystem (monitoring, routing) is a plus. **Evaluation uses Celery too, but a separate app/queue/worker on the API image** (FR-12.5): scoring needs LangGraph, so it must not share the ingest pool or live in `ingestion/`.
8. **Metrics always on, tracing optional** — Prometheus/Grafana give operational SLIs (latency, throughput, CPU/mem, retrieval time) with no external dependency; Langfuse (deep LLM traces) is opt-in and no-ops without credentials, so the app never hard-depends on it.
9. **Pluggable web search (Brave/Tavily)** — same `WebSearchProvider` interface; pick per cost/quality/availability via config.
10. **Prompt registry + eval dataset** — prompts are versioned and quality is measurable against a fixed dataset, so a prompt/model/embedding change is attributable and comparable.
11. **HITL via `interrupt()` + checkpointer, gated by a per-tool `requires_approval` flag** — consequential (outward-reaching / side-effecting) tools need user consent, and memory writes need the right page range. Making approval a declared property of each tool (rather than a hard-coded list in the agent) means new external tools are gated automatically and the read-only tools never are; checkpointing makes paused runs durable and resumable.
12. **Guardrails as graph bookends** — topical + safety checks keep the assistant on-task and safe without scattering checks through the code.
13. **LangGraph checkpointer for short-term memory** — resumable sessions, HITL resume, and a natural compaction point.
13.1 **Token-budget auto-compaction, not fixed-window** — bounding short-term memory by a fraction of the *active model's* context window (rather than a hard turn/message cap) keeps sessions long-lived across models with very different windows, and the concise summary-as-seed keeps continuity while collapsing cost. The threshold is a ratio so it self-scales when the model changes.
14. **SSE primary, WebSocket for interactivity** — SSE is simpler for one-way token/event streaming; WS is the fit for HITL's bidirectional exchange.
15. **Independent user/document languages + multilingual embeddings** — chat language and book language are decoupled so a user can read across languages; a multilingual embedding model (not per-language indexes or query translation) keeps cross-lingual retrieval in one vector space, and answer language is a prompt variable so prompts aren't duplicated five times. Fixed 5-language set ⇒ an enum, not a lookup table.
16. **Externalized agent scratchpad vs everything-in-context** — holding plan/findings/open-questions in a Redis-backed, turn-scoped store and pulling back only relevant slices keeps multi-step research turns cheap and defers compaction; it's a third memory tier, orthogonal to the checkpointer (conversation) and vectors (cross-session), and ephemeral by design (TTL) so it never becomes durable state to reason about.
17. **Structured JSON for internal nodes, natural language for the answer** — schema-validated outputs (`.with_structured_output`) make planner/guardrail/classifier routing deterministic and unit-testable, while the user-facing answer stays streamed tokens with citations (JSON mode would kill streaming). The split keeps machine-consumed steps typed without sacrificing UX.
18. **Event-sourced reading history for analytics** — an append-only `reading_events` trail (not just mutable `current_page`) is what makes pace/streaks/history computable and auditable; analytics are derived + cached, so the hot read/update path stays cheap.
19. **Spoiler-safe as a layered, cross-source constraint** — a single retrieval default can't hold the guarantee once web search and model knowledge enter, so spoiler-safety is enforced at retrieval (hard page filter), memory (page_end bound), *and* output (a generation-time spoiler check), with an explicit HITL opt-in rather than a silent reveal. Position-awareness is a product promise, so it's belt-and-suspenders.
20. **Multimodal in, text everywhere else** — normalizing audio/image to text at the front door (`normalize_input`) keeps one embedding space, one guardrail path, and one reasoning path; the alternative (native multimodal vectors/models throughout) multiplies the vector store, guardrails, and provider surface for marginal gain in a reading assistant.
21. **Eval scoring off the request path, on the API image** — `POST /evaluations/run` persists `pending` and returns 202; a dedicated `eval` Celery queue runs `AgentService` + judges. Putting that work on ingestion workers would import LangGraph into the CPU-heavy ingest process (or require a forbidden sync call to the API). An Admin shell section (FR-21) is the operator UI; it is not a second SPA.
