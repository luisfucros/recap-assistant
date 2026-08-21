# ingestion — Ingestion pipeline service

Celery-based service that turns uploaded documents into searchable, page-tagged
chunks off the API's request path. CPU/memory-heavy (parsing, embedding); scales
on worker count. It never calls the API — it is driven by the transactional
outbox: the API commits an upload + outbox event, this service drains the queue
and writes chunks/vectors back to Postgres/Qdrant.

## Responsibilities

- Consume ingestion jobs (`ingest_document`: fetch original → parse → detect language → chunk → embed in batches → upsert to Qdrant → persist chunks → mark `indexed`). The terminal status + chunk insert + `document.indexed` event commit in one transaction, only after the vector upsert succeeds; a re-run replaces prior vectors/chunks (idempotent). Transient failures retry with backoff; a bad-bytes parse fails permanently.
- Run the beat-scheduled outbox relay (single instance): dispatch a task per pending outbox event, then mark it processed.
- Expose the worker's Prometheus `/metrics`.

## Key entry points

- **`ingestion.celery_app:app`** — the Celery application (`-A ingestion.celery_app:app`); registers the tasks and the beat schedule.
- **`ingestion.tasks`** — `ingest_document` (the pipeline task; retry/fail policy).
- **`ingestion.pipeline`** — `run_ingestion` / `fail_document`: the async pipeline the task runs.
- **`ingestion.outbox_relay`** — `relay_outbox` (beat task) + `drain_outbox` (dispatch logic).
- **`ingestion.resources`** — `IngestionResources` (per-worker singletons: DB, Qdrant; lazy storage/embedder) via `get_ingestion_resources()`.
- **`ingestion.bootstrap`** — Celery signal hooks: start the `/metrics` server on `worker_ready`, dispose resources on `worker_shutdown`.

The parse/chunk/embed/language building blocks are pure and live in the shared
library (`shared.ingestion_core`); vector access is `shared.vectorstore`.

## Run in isolation

```bash
uv run celery -A ingestion.celery_app:app worker -l INFO   # worker (needs Redis for the broker)
uv run celery -A ingestion.celery_app:app beat -l INFO     # beat (outbox relay; single instance)
```

Inspect config without a broker: `uv run celery -A ingestion.celery_app:app report`.

## Test in isolation

```bash
uv run pytest services/ingestion
```

## Container

```bash
docker build -f services/ingestion/Dockerfile -t recap-ingestion .   # build context = repo root
```

The same image runs the worker (default `CMD`) and, with an overridden command, beat.
