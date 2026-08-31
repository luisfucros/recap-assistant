# Requirements — Recap

> Personalized Agentic RAG reading companion. Users upload books/documents; the assistant helps them remember what they've read via contextual Q&A, summaries, progress tracking, and recommendations. **Reading position is the backbone**: what the user has read drives which summaries are produced and which long-term memories are saved for future recaps.

## 1. Goals

1. **Solve the "lost context" problem** — let users pick up a book after a break and instantly recall characters, events, concepts, and where the narrative/argument was heading.
2. **Ground every answer in the user's own documents** — retrieval-augmented, with page/chapter/section citations, never hallucinated recall.
3. **Use reading position as first-class context** — summaries and memories cover the pages the user has actually read, not the whole book.
4. **Personalize over time** — the assistant gets better as it learns the user's reading list, habits, preferences, and prior discussions.
5. **Be production-ready** — authenticated, multi-tenant, observable, secure, and horizontally scalable.

### Non-goals (v1)

- Real-time collaborative reading / shared annotations.
- In-app document reader/viewer (the app is a companion, not a reader) — position is user-reported or set via API.
- Formats beyond PDF at launch (architecture must not preclude EPUB/DOCX/TXT/HTML later).

## 2. Personas & user stories

### Reader (primary)

- As a reader, I want to **upload a PDF** and be notified when it's ready to query, without waiting on a blocked screen.
- As a reader, I want to ask **"What did I read last?"** and get a recap scoped to the pages I've actually read up to my current position.
- As a reader, I want to ask **"Who is this character?"** or **"What was the author arguing?"** and get an answer cited to specific pages/chapters **within what I've read** (no spoilers past my position unless I ask).
- As a reader, I want to request **"Summarize pages 5–15"** or **"What's the main idea of chapter 3?"**.
- As a reader, I want the assistant to **remember summaries of what I've read** so future recaps are instant and consistent.
- As a reader, I want **follow-up questions** to work without me repeating context ("and what happened after that?").
- As a reader, I want to **track my reading progress** and see what I'm currently reading, what I've finished, and my current page.
- As a reader, I want **recommendations** for what to read next based on what I've read.
- As a reader, I want to **watch the assistant think** — see which tools it's using and stream the answer as it's generated.
- As a reader, I want to **log in with Google** or email/password and trust that **only I can access my books, memories, and history**.

### Administrator / operator (secondary)

- As an operator, I want ingestion jobs to **retry on transient failure** and surface permanent failures clearly.
- As an operator, I want **per-user data isolation** enforced at the data layer, not just the UI.
- As an operator, I want to **swap LLM and embedding providers** (hosted or local) via configuration without code changes.
- As an operator, I want **metrics and logs** for ingestion throughput, query latency, token spend, and tool usage.

## 3. Functional requirements

### FR-1 Document ingestion

- FR-1.1 Accept PDF upload via the API; validate type and size at the boundary (see NFR-Security).
- FR-1.2 Store the original file in object storage via an **S3-compatible interface** (MinIO locally, AWS S3 in production); never trust client-supplied paths.
- FR-1.3 Ingestion runs as an **asynchronous background job**; the upload request returns immediately with a document id and `status: pending`.
- FR-1.4 Extract text and **rich metadata per chunk**: page number(s), chapter, section, document title, author, plus offsets. Missing metadata degrades gracefully (nullable fields).
- FR-1.5 Chunk text with a strategy that preserves structure (respect chapter/section/page boundaries; configurable size + overlap). Every chunk records its `page_start`/`page_end` so retrieval and summaries can be bounded by reading position.
- FR-1.6 Generate embeddings per chunk (via the configured embedding provider) and upsert into the vector store with full metadata payload for filtered retrieval.
- FR-1.6.1 Embeddings are generated in **configurable batches** to bound memory use and avoid out-of-memory failures — critical for local HuggingFace/sentence-transformers models. Batch size is a setting; a failed batch retries without reprocessing succeeded batches.
- FR-1.7 Expose document status lifecycle: `pending → processing → indexed | failed`. Failures record a reason and are retryable.
- FR-1.7.1 The pipeline uses a **transactional outbox** so a document is only marked `indexed` — and downstream events are only emitted — after chunks and vectors are durably persisted. A connection error to Qdrant, storage, or the LLM/embedding API must **never** leave a half-written or wrongly-logged state: either the step commits atomically or it is retried; no partial/incorrect status is recorded.
- FR-1.8 Metadata must support filtered queries: by page range, by chapter, by section, "before/after this position", and **"only pages I've read"**.
- FR-1.9 Architecture must allow adding new format parsers (EPUB, DOCX, TXT, HTML) without changing the ingestion pipeline contract (parser Strategy/Factory).
- FR-1.10 **Duplicate detection.** A content hash (`content_sha256`) is computed over the uploaded bytes. Documents are unique per `(user_id, content_sha256)`. Re-uploading identical content the user already has is **rejected with `409` + code `DUPLICATE_DOCUMENT`** (referencing the existing document id) — it is not re-ingested. Uniqueness is enforced by a DB constraint so concurrent duplicate uploads cannot both succeed (the loser gets the 409). Object storage is content-addressed by hash so identical bytes are stored once per user.
- FR-1.11 **Cross-user isolation over dedup.** Identical content uploaded by *different* users is **not** deduplicated — each user gets their own document, chunks, and vectors. Strict per-user isolation (FR-6.4) takes priority over storage/compute savings.
- FR-1.12 **Retrieval-level de-duplication (defense in depth).** Retrieval collapses near-identical chunks so a stray duplicate never surfaces the same passage twice in one answer.

### FR-2 Personalized reading assistant

- FR-2.1 Answer contextual questions grounded in the user's ingested documents with **citations** (document, page, chapter).
- FR-2.2 Recaps ("what did I read last", "remind me what happened before") are scoped to the user's **read range** — pages from the start (or last recap) up to the current position.
- FR-2.3 By default, retrieval and summaries are **bounded to pages the user has read** to avoid spoilers; the user can explicitly opt to include unread content.
- FR-2.4 Support both **concise reminders** and **detailed summaries**, chosen from the user's phrasing or an explicit parameter.
- FR-2.5 Answer character/concept/argument questions ("who is X", "what was the author arguing").
- FR-2.6 If retrieval finds nothing relevant, say so — do not fabricate.

### FR-3 Reading progress tracking (drives summaries & memory)

- FR-3.1 Track per-user, per-document: current position (page), status (`not_started | reading | completed | cancelled`), last-accessed timestamp, and the **last-summarized page** (high-water mark for recaps). The assistant can tell the user exactly which docs are in progress, completed, or cancelled.
- FR-3.2 Maintain reading history and a "recently accessed" list.
- FR-3.3 Allow the user to update their position; auto-advance is out of scope for v1 (no reader integration).
- FR-3.4 When the user advances their position, the assistant can **summarize the newly-read span** (from last-summarized page to current page) and persist it as a long-term memory keyed to that page range (see FR-4).
- FR-3.5 Progress is the scoping signal for position-bounded retrieval, recaps, and "before this section" queries.

### FR-4 Memory system

- FR-4.1 **Short-term memory**: session/conversation-scoped; enables follow-ups without repetition; bounded and **auto-compacted** when long (FR-4.1.1–4.1.3).
- FR-4.1.1 **Token accounting.** The running conversation's token count is tracked per session, measured with the active model's tokenizer (provider-aware; a documented estimator when no exact tokenizer is available). Every turn updates the count.
- FR-4.1.2 **Context-window threshold.** A session is considered "long" relative to the **active model's context window**, not a fixed number: compaction triggers when the running token count crosses a configurable fraction of that window (default 75%), leaving headroom for the next turn's prompt, tools, and completion. Both the per-model context window and the threshold fraction are configuration.
- FR-4.1.3 **Auto-compaction.** When the threshold is crossed the session is compacted automatically (no user action): the full history is summarized (cheap-tier model, registry prompt), the conversation is restarted from that summary as its seed context, and the token count is reset. The summary is **concise but complete enough to continue the session seamlessly** — it captures what the user is doing, decisions and answers already given, the current document/page focus, any open HITL interrupt, and unresolved threads, while dropping verbatim back-and-forth. Compaction is idempotent, and salient facts it surfaces may also be promoted to long-term memory (FR-4.2).
- FR-4.2 **Long-term memory**: persists across sessions — reading list, completed items, preferences, FAQs, **per-page-range summaries of what was read**, key concepts discussed, reading habits.
- FR-4.3 Long-term memories from summaries are **tied to a document + page range**, so a later "remind me what happened before page N" can retrieve the relevant saved summary instead of re-reading.
- FR-4.4 Long-term memory is **retrievable by the agent as a tool** and used to personalize answers and recommendations.
- FR-4.5 Memory writes are attributable and user-isolated; users can view and delete their long-term memories (privacy).
- FR-4.6 **Always confirm the page range before saving to long-term memory.** When the assistant is about to persist read-progress information (a summary, "what happened", a concept tied to the narrative), it must ask the user which pages the information covers (proposing the read-range default) rather than guessing.
- FR-4.7 **Ask when prior-read context is missing.** If the user asks about earlier reading of a document and the assistant finds no stored memory/progress for it, it must ask the user which pages they have read (or up to which page) instead of fabricating or silently returning nothing — this is a human-in-the-loop clarification (see FR-13).

### FR-5 Recommendation engine

- FR-5.1 Recommend documents/books similar to what the user is reading or has completed.
- FR-5.2 Signals: semantic similarity across documents, reading history, long-term memory, and **external sources** (web search / recommendation APIs). Web search is a **pluggable provider** (Brave or Tavily), selected by configuration.
- FR-5.3 Recommendations are explainable ("because you read X").

### FR-6 Authentication & authorization

- FR-6.1 Email/password auth with JWT (15-min access, 7-day refresh) and **Google OAuth** login.
- FR-6.2 Tokens delivered via httpOnly cookies; never localStorage.
- FR-6.3 Every protected endpoint verifies the token via a FastAPI dependency; **authorization enforced at the service layer**.
- FR-6.4 **Strict per-user isolation**: a user can access only their own documents, chunks, memories, progress, and conversations. Every document/chunk read is checked against the requesting user's id at the data layer.

### FR-7 Agentic assistant

- FR-7.1 The assistant is an **autonomous agent** that plans and calls tools across **three user-scoped knowledge sources**:
  - **Relational DB (reading state):** `get_reading_progress` — which docs are `reading | completed | cancelled | not_started`, current page, last-summarized page, recently accessed.
  - **Vector DB — long-term memory:** `query_long_term_memory` — user preferences, summaries of current readings (by page range), and important facts the user shared.
  - **Vector DB — document chunks:** `retrieve_chunks` — semantic + metadata-filtered extraction from specific pages/chapters (defaults to the user's read range).
  - Plus: `summarize` (page/chapter/section range), `web_search` (pluggable Brave/Tavily), `recommend`.
- FR-7.2 The agent decides which tools to call and may chain them (multi-step reasoning), e.g. read progress → retrieve read-range chunks → summarize → save memory.
- FR-7.7 **Vector isolation:** both vector collections (document chunks and long-term memory) store `user_id` in their payload, and **every vector search is filtered by the authenticated user's id**, injected server-side — never taken from LLM/tool arguments. No query can return another user's chunks or memories.
- FR-7.3 **Streaming**: stream intermediate tool-call/reasoning events AND the final token stream to the client (SSE and/or WebSocket).
- FR-7.4 Guardrails on input and output (see FR-10).
- FR-7.5 Conversation state is checkpointed so a session can resume.
- FR-7.6 **Consequential tools require human approval** before execution: any tool that reaches beyond the user's own stored data — sends data to a third party, spends external/rate-limited API budget, or has a side effect — is gated by HITL (see FR-13). Approval is driven by a per-tool `requires_approval` flag, not a hard-coded list, so read-only user-scoped tools run freely while new external tools inherit the gate. Currently gated: `web_search` and external `recommend`.
- FR-7.8 **Agent scratchpad (turn-scoped working memory).** For multi-step turns the agent keeps a scratchpad — **plan, running findings, open questions** — held **outside the model context window** (not re-sent verbatim each step). Only the **relevant** slices are pulled back in per step (by recency/relevance), so long research turns don't bloat the context or trigger premature compaction. The scratchpad is turn/conversation-scoped and ephemeral (distinct from short-term conversation state and long-term memory); its salient conclusions may be promoted to the answer or to long-term memory.
- FR-7.9 **Structured node outputs.** The agent's **internal** reasoning steps (planner, guardrail verdicts, memory-type classification, HITL interrupt payloads, evaluation scoring) emit **schema-validated JSON** (one typed schema per task) so they are deterministic to parse, route on, and test. The **final user-facing answer is exempt** — it remains natural-language streamed tokens with citations, never JSON-wrapped.

### FR-8 Configurable model providers

- FR-8.1 **LLM provider is pluggable** and selected by configuration: Anthropic Claude, OpenAI, or a **local model via Ollama using the OpenAI-compatible client**. No code changes to switch.
- FR-8.2 **Embedding provider is pluggable** and selected by configuration: hosted (OpenAI / Voyage) or **local HuggingFace / sentence-transformers**. Embedding dimension is read from the active provider so the vector store is provisioned correctly. The configured model must be **multilingual** across the supported languages to enable cross-lingual retrieval (FR-16.5).
- FR-8.3 Switching embedding providers triggers a documented **re-embedding** path (vectors are derived data; text remains the source of truth).

### FR-9 Prompt management

- FR-9.1 All LLM prompts (system, planner, summarizer, guardrail, memory-extraction, recommendation) are managed in a **central, versioned prompt registry** — not hard-coded inline strings scattered across services.
- FR-9.2 Prompts are addressable by name + version and rendered with typed variables; changing a prompt does not require a code change to callers.
- FR-9.3 Prompt versions are tied to traces and evaluations (FR-11, FR-12) so a regression can be attributed to a specific prompt version.

### FR-10 Guardrails

- FR-10.1 **Input guardrails**: reject or safely handle prompt-injection attempts and strip/redact PII before sending to hosted providers.
- FR-10.2 **Topical relevance guardrail**: the assistant is a reading companion — off-topic requests (unrelated to the user's documents/reading) are politely declined and redirected. The judge sees the current message plus a short prior user/assistant slice so follow-ups in the same conversation stay classifiable.
- FR-10.3 **Appropriateness/safety guardrail**: inappropriate, unsafe, or abusive requests are blocked with a standard refusal.
- FR-10.4 **Output guardrail**: sanitize model output before it reaches the frontend to prevent prompt-injection-driven XSS.
- FR-10.5 Guardrail decisions are traced (FR-11) with the reason, and a blocked request returns a clear, non-leaky message.

### FR-11 Observability & tracing (LLM/RAG)

- FR-11.1 End-to-end tracing of every agent turn via **Langfuse**: spans for LLM calls, each tool call, embedding calls, retrieval (query, filters, returned chunks + scores), and guardrail decisions.
- FR-11.2 Traces capture token usage, latency, model/provider, and prompt name+version per step.
- FR-11.3 Ingestion is traced too: parse/chunk/embed/upsert timings and batch sizes.
- FR-11.4 Traces are correlated to user and conversation ids (without logging raw PII) so a specific interaction can be inspected.
- FR-11.5 **Langfuse is optional.** If no Langfuse credentials are configured, tracing is a no-op and the application runs normally — no errors, no degraded functionality. Tracing must never be a hard dependency.

### FR-12 Evaluation service

- FR-12.1 An **evaluation service** runs the app (retrieval + agent answers) against a **custom, versioned dataset** of question → expected-behavior/reference cases.
- FR-12.2 Metrics cover retrieval quality (e.g. hit rate / recall of relevant chunks) and answer quality (groundedness/faithfulness, relevance, citation correctness), including an LLM-as-judge option.
- FR-12.3 Evaluations are runnable on demand and in CI against a fixed dataset; results are recorded/traced (Langfuse) and comparable across prompt/model/embedding versions.
- FR-12.4 Datasets are managed as artifacts (versioned) so runs are reproducible.
- FR-12.5 **An evaluation run is a background job, not an HTTP request.** Triggering a run (API or CLI) **persists a `pending` row and returns immediately**; scoring (seed fixtures, retrieval, agent turns, LLM-as-judge) runs in a **dedicated Celery worker that belongs to the Assistant/API service** — same image and `Resources` as the API, **separate queue and process** from ingestion. Ingestion workers never run eval tasks (they have no LangGraph/agent stack; the two services do not call each other). A run's status is `pending` → `running` → `completed` / `failed`; the client **polls** (list/get), it does not stream tokens. A stuck `pending`/`running` run is re-enqueued by a sweep on the eval worker, analogous to stuck-document ingest. The CLI / CI entrypoint still exits non-zero only if the run **failed to complete**, never on a low score.
- FR-12.6 Admins can **list shipped datasets**, **start a run**, and **inspect past runs** (status, prompt/model/embedding tags, summary metrics, per-case results, error) from the product UI (FR-21), not only via HTTP/CLI.

### FR-13 Human-in-the-loop (HITL)

- FR-13.1 **What gets gated — the principle.** HITL approval is required before any **consequential tool** runs: a tool that acts *beyond reading the user's own stored data* — one that **sends data to a third party**, **incurs external cost or rate-limited external-API usage**, or **has a side effect the user should authorize**. Gating is a **declared property of the tool** (each tool carries a `requires_approval` flag surfaced to the graph), **not** a hard-coded list in the agent — so every present and future consequential tool inherits the gate automatically. The user-scoped **read tools** (`get_reading_progress`, `retrieve_chunks`, `summarize`, `query_long_term_memory`) touch only the caller's own data with no external egress or side effect, so they run **without** approval; adding a new tool that reaches outside means setting its flag, not editing the agent.
- FR-13.2 **Current gated tools.** Under FR-13.1 this presently covers **`web_search`** and the **external recommendation API** path of `recommend` (a purely internal, similarity-based recommendation is not gated). Before such a call the agent pauses and requests approval, showing the **proposed query/action** so the user can approve, edit, or deny it; execution proceeds only on approval.
- FR-13.3 **Page-range confirmation (a distinct HITL class).** Before saving read-progress information to long-term memory, the agent confirms the **page range** with the user (FR-4.6); when prior-read context is missing it asks which pages were read (FR-4.7). This is a *confirmation* interrupt, not an external-action approval, but shares the same interrupt/resume machinery.
- FR-13.4 HITL interrupts are surfaced over the streaming channel and the paused run **resumes** from its checkpoint once the user responds (approve / edit / deny).
- FR-13.5 HITL prompts and decisions are traced (FR-11).

### FR-14 Server & application metrics (Prometheus / Grafana)

- FR-14.1 Every service exposes a Prometheus `/metrics` endpoint scraped by a Prometheus server; **Grafana** provides dashboards.
- FR-14.2 Metrics cover: HTTP request **latency** (histograms) and **throughput** per route/status, in-flight requests and error rates; process/host **CPU and memory**; and RAG-specific timings — **retrieval time**, embedding time, LLM call latency, tool latency, queue depth, and ingestion job duration/throughput.
- FR-14.3 Metrics are labeled by service, route, and outcome (not by raw user PII); high-cardinality labels are avoided.
- FR-14.4 Metrics collection is independent of Langfuse tracing (FR-11): metrics are always on; tracing is optional.

### FR-15 Service architecture (microservices)

- FR-15.1 The system is deployed as **independent microservices**; at minimum the **assistant/API backend** and the **ingestion pipeline** are separate, independently deployable and independently scalable services.
- FR-15.2 Services communicate **asynchronously** through shared infrastructure (Postgres + transactional outbox, the Celery broker, object storage, Qdrant) — not via synchronous in-process calls. The ingestion service can scale (worker count) without touching the API service.
- FR-15.3 Each service exposes its own health and `/metrics` endpoints and can start/stop independently; a failure or restart of the ingestion service must not take down the API (uploads queue and drain when it returns).

### FR-16 Multilingual support

- FR-16.1 **Supported languages**: English, Spanish, German, French, and Italian (ISO 639-1: `en`, `es`, `de`, `fr`, `it`). A single shared `Language` enum is the source of truth for both users and documents.
- FR-16.2 **User language.** Each user has a **preferred language** (persisted on their profile, default `en`, editable) — the language the assistant chats in and the UI defaults to.
- FR-16.3 **Document language.** Each document has a stored **language**, detected from its parsed text at ingestion and overridable by the user; unknown/unsupported detections fall back to a default and are flagged rather than rejected.
- FR-16.4 **User and document languages are independent.** The chat language may differ from a book's language (e.g. a Spanish-speaking user reading a German book). The assistant **answers in the user's preferred language regardless of the document's language**; verbatim quotes/citations are shown in the document's original language (translated only if the user asks). Guardrail refusals (including canned injection/off-topic fallbacks) use that same language.
- FR-16.5 **Cross-lingual retrieval.** Retrieval must match a query in the user's language against chunks in the document's language, so the active **embedding model must be multilingual** across the supported set (FR-8.2); the answer's target language is passed to the generation prompt as a variable rather than duplicating prompts per language (FR-9).

### FR-17 Reading analytics

- FR-17.1 The system records a **history of reading activity** (position advances, session starts/ends, status changes, completions) — an append-only event trail, not just the current position — so trends can be computed over time.
- FR-17.2 The user can see **analytics** derived from their own data: reading **pace** (pages/day), **streaks/habits** (active days, typical reading times), **pages read per period**, books **started / completed / cancelled**, per-document progress, and a simple **time-to-finish estimate** from recent pace.
- FR-17.3 Analytics are **strictly user-isolated** (computed only from the requesting user's events) and exposed via API + a UI dashboard.
- FR-17.4 Analytics may feed personalization and the **`habit`-type long-term memory** (FR-4.2); expensive aggregations are cached and refreshed, not recomputed per request.

### FR-18 Spoiler-safe mode

- FR-18.1 A **spoiler-safe mode** guarantees the assistant never reveals content **beyond the user's current reading position** for a document — building on read-range scoping (FR-2.3) but as a **hard constraint**, not just a retrieval default.
- FR-18.2 It is a **user setting (default on)**, with a **per-document override** and a **per-query opt-in** ("it's fine, spoil me").
- FR-18.3 Enforcement is **layered across every source**: `retrieve_chunks` hard-filters to `page_end <= current_page`; long-term-memory summaries are bounded by their `page_end`; and a **generation-time spoiler check** (guardrail) catches ahead-of-position content that could leak from web search or the model's own knowledge — redacting or refusing it.
- FR-18.4 When a request would require content past the current page, the assistant **warns and asks for explicit opt-in** (HITL, see FR-13) rather than silently revealing or silently returning nothing.

### FR-19 Multimodal input

- FR-19.1 The assistant accepts **multimodal input** — **text, audio, and images** — in a chat turn.
- FR-19.2 Non-text input is **normalized to text before reasoning**: audio is **transcribed** (speech-to-text) and images are **described/captioned** (vision model). Both run behind **pluggable providers** (hosted or local), config-selected like the other providers (FR-8). The **local transcription option is OpenAI's Whisper model run through Hugging Face** (`transformers`, offline), mirroring the HuggingFace-local embedder — so the whole multimodal path can run with no hosted API, exactly like the fully-local embeddings/storage options.
- FR-19.3 **Embeddings are always text.** The vector store stays **single-modality** — any audio/image is converted to text first, so there is one embedding space and no separate image/audio vectors.
- FR-19.4 The normalized text flows through the **same pipeline** (guardrails, retrieval, memory, spoiler-safe) as typed text. Original media is stored (object storage); the derived transcript/description is what the agent reasons over and what is optionally embedded.
- FR-19.5 Guardrails (injection/PII/topical/appropriateness, FR-10) run on the **normalized text**, so multimodal input cannot bypass them.

### FR-20 Frontend design system

- FR-20.1 The frontend uses a single, cohesive **design system** — shared color/spacing/typography tokens and a small set of reusable primitives (buttons, inputs, cards, badges, nav) — rather than per-component ad hoc markup, so every panel (library, reading, chat, memory, recommendations, analytics, and the admin console when shown) reads as one product.
- FR-20.2 The signed-in app is organized as a **persistent shell** (branding, navigation, account controls) with a focused content area per section, not one long undifferentiated scroll of every panel at once.
- FR-20.3 Interactive states — loading, empty, error, and success — are visually distinct and present for every panel that fetches data, so the user is never looking at a blank or ambiguous screen.
- FR-20.4 The layout is responsive down to a single-column mobile width; no horizontal scrolling of the page itself (wide content like tables/transcripts scrolls in its own container).
- FR-20.5 Visual design does not change any existing HTTP contract, component prop/behavior contract visible to tests, or accessibility semantics (labels, roles) already relied upon.

### FR-21 Admin console

- FR-21.1 Signed-in users with `is_admin` see an **Admin** section in the persistent shell (FR-20.2); non-admins never see it and receive 403 on admin-only routes (same `AdminUser` gate as today).
- FR-21.2 From that section an admin can **create a regular or admin account** (`POST /admin/users`) without self-registration — public sign-up still cannot set `is_admin`.
- FR-21.3 The same section is the operator UI for **evaluation runs** (FR-12.5 / FR-12.6): pick a dataset, enqueue a run, watch status, read scores. No separate admin SPA.

## 4. Non-functional requirements

### Performance & scale

- NFR-1 First streamed token (TTFT) < 2 s p50 for a typical query on an indexed doc.
- NFR-2 Ingestion of a 300-page PDF completes < 5 min p90; runs off the request path.
- NFR-3 Retrieval query (vector search) < 300 ms p95 excluding LLM.
- NFR-4 Stateless API workers scale horizontally; background workers scale independently of API.

### Reliability

- NFR-5 Ingestion jobs are idempotent and retry with backoff; poison jobs land in a dead-letter path.
- NFR-6 Document→vector consistency guaranteed via the **transactional outbox pattern** (no lost/duplicated indexing on crash).
- NFR-7 Target 99.5% API availability.

### Security

- NFR-8 Validate all external input via Pydantic; enforce upload type/size limits at the API boundary.
- NFR-9 No secrets in code; load from env; document in `.env.example`.
- NFR-10 CORS restricted to explicit origins; HTTPS everywhere; security headers set.
- NFR-11 Never log tokens/PII; sanitize LLM output to prevent prompt-injection-driven XSS.
- NFR-12 Never send full user PII or secrets to external LLM/search APIs (applies to hosted providers; local Ollama/HF keep data on-prem).

### Observability

- NFR-13 Structured logs, request tracing, and metrics: ingestion throughput, query latency, token spend per user, tool-call counts, error rates.
- NFR-13.1 LLM/RAG tracing via **Langfuse** covers agent turns, tools, embeddings, retrieval, and guardrails (FR-11); the app degrades gracefully (tracing failures never break a user request).

### Maintainability & quality

- NFR-14 Adhere to the project's code-style, API-design, testing, and Docker conventions. Type hints everywhere; layered architecture.
- NFR-15 80% line coverage on service/utility modules; three-tier test suite (unit/integration/functional).

### Usability

- NFR-16 The frontend follows a documented design system (FR-20): consistent visual language, responsive layout, and explicit loading/empty/error states across every panel.

## 5. Constraints & assumptions

- **LLM provider:** pluggable — Claude, OpenAI, or local Ollama (OpenAI-compatible client), chosen via `LLM_PROVIDER`. A cheap model tier is used for classification/summarization sub-steps when the provider offers one.
- **Embeddings:** pluggable — OpenAI/Voyage (hosted) or HuggingFace/sentence-transformers (local), chosen via `EMBEDDINGS_PROVIDER`.
- **Object storage:** S3-compatible — MinIO in local/dev, AWS S3 in production.
- **Web search:** pluggable — **Brave or Tavily**, selected via `WEB_SEARCH_PROVIDER`; gated behind HITL approval (FR-13).
- **Task queue:** **Celery** (Redis broker) drives the ingestion pipeline service.
- **Architecture:** microservices — assistant/API backend and ingestion pipeline are separate, independently scalable services (FR-15).
- **Metrics:** Prometheus scrape + Grafana dashboards, always on (FR-14).
- **Tracing/eval:** Langfuse for tracing, prompt-version linkage, and evaluation-run recording (self-hosted in dev via compose). **Optional** — absent credentials disable tracing without breaking the app (FR-11.5).
- **Prompts:** managed in a versioned registry (Langfuse prompts or an equivalent store), not inline.
- **Formats:** PDF only at launch; parser abstraction ready for more.
- Users self-report reading position (no e-reader integration in v1).
- **Frontend styling:** Tailwind CSS utility classes plus a small shared token/primitive layer (FR-20) — no separate component-library runtime dependency.
- **Deployment:** development runs on Docker Compose (§16 in `spec/architecture.md`); **production runs on AWS via Terraform** (§17 in `spec/architecture.md`, Milestone 10) with RDS Aurora PostgreSQL, Qdrant on Fargate (EBS-backed, S3 snapshots), ElastiCache Redis, ECR (private image repos + pull-through cache), ALB, and CloudFront. The application code is **identical across environments**; all differences are infrastructure-level.

## 6. Acceptance criteria (v1 "done")

- A user can register/login (email+password and Google), upload a PDF, and see it transition to `indexed`.
- Asking "summarize pages 5–15" and "who was introduced in chapter 3" returns cited, grounded answers **bounded to what the user has read** by default.
- "What did I read last?" produces a recap scoped to the user's read range; advancing position lets the assistant summarize the newly-read span and save it as a long-term memory keyed to those pages.
- A later recap retrieves the saved summary for the relevant page range instead of re-reading.
- Follow-up questions work within a session without restating context.
- The agent visibly streams tool-call events then the final answer.
- Recommendations return explainable suggestions.
- Before saving a summary to long-term memory, the assistant asks the user to confirm the page range; when no prior-read context exists it asks which pages were read.
- Consequential tools (those reaching outside the user's own data — currently web search / external recommendation) prompt for approval and only run once approved; read-only user-scoped tools run without a prompt.
- Off-topic and inappropriate requests are declined by the guardrails.
- Every agent turn (LLM, tools, embeddings, retrieval, guardrails) appears as a Langfuse trace with token/latency/prompt-version data **when Langfuse is configured**; with no credentials the app runs identically without tracing.
- The evaluation service runs against a custom dataset and reports retrieval + answer-quality metrics; an admin can enqueue a run from the UI and the HTTP request does not wait for scoring to finish.
- An admin can create another user (regular or admin) from the Admin section; a non-admin never sees that section.
- The assistant/API backend and ingestion pipeline run as **separate services**; stopping the ingestion service queues uploads without taking down the API.
- Prometheus scrapes each service's `/metrics`; Grafana shows request latency/throughput, CPU/memory, and retrieval/ingestion timings.
- Web search works with either Brave or Tavily by changing `WEB_SEARCH_PROVIDER`.
- Re-uploading a document the user already has returns `409 DUPLICATE_DOCUMENT` (no duplicate chunks/vectors); the same file from a different user ingests independently; concurrent duplicate uploads never both succeed.
- **Isolation:** a user cannot read another user's documents, memories, or progress (verified by test).
- **Multilingual:** documents are language-detected on ingestion; a user asking in their preferred language (one of en/es/de/fr/it) about a document in another supported language gets a cross-lingual, grounded answer **in their language**, with original-language quotes.
- **Reading analytics:** a dashboard shows the user's pace, streaks, pages-read-over-time, and started/completed/cancelled counts, computed only from their own activity.
- **Spoiler-safe:** with spoiler-safe on, asking about events past the current page yields a warning + opt-in rather than a spoiler; opting in (or turning the mode off) reveals the content; no source (chunks, memory, web) leaks ahead-of-position content while on.
- **Multimodal input:** a chat turn with an audio clip or an image is transcribed/described to text and answered through the same grounded, guardrailed pipeline; nothing but text is ever embedded.
- LLM and embedding providers can be switched to a local (Ollama/HuggingFace) configuration without code changes.
- CI runs the three-tier test suite; service-module coverage ≥ 80%.
- **Frontend:** the signed-in app renders as a consistent, navigable shell (not one long unstyled scroll); every data-fetching panel shows a distinct loading/empty/error/success state; the layout works down to a single-column mobile width with no page-level horizontal scroll.
