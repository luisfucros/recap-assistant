# Recap

**Your reading assistant.** A personalized **Agentic RAG reading companion**. Users upload books and documents; the assistant helps them remember what they've read — contextual Q&A, summaries scoped to their reading position, progress tracking, and recommendations — grounded in their own library with per-user isolation.

> **Status:** early development. The infrastructure, service scaffolding, and provider abstractions are in place and tested; user-facing features (auth, upload/ingestion, retrieval, the agent) are being built milestone by milestone. See `spec/tasks.md` for the roadmap.

## Architecture

A microservice system in a monorepo. Two independently deployable services share one library and coordinate **asynchronously** — they never call each other directly.

- **`services/api`** — Assistant/API service: FastAPI + a LangGraph agent. Latency-sensitive, stateless, scales on replicas.
- **`services/ingestion`** — Ingestion pipeline: Celery workers + beat. CPU/memory-heavy (parse → chunk → embed → upsert), scales on worker count.
- **`libs/shared`** — imported by both: config, models, DB/migrations, repositories, provider abstractions, observability, prompt registry.
- **`services/frontend`** — React SPA (Vite + TypeScript), served by nginx.

**Infrastructure:** PostgreSQL (source of truth + transactional outbox), Qdrant (vector search), Redis (Celery broker + cache), S3-compatible object storage (MinIO locally, AWS S3 in production), Prometheus + Grafana (metrics), and optional Langfuse (LLM tracing). Model providers (LLM, embeddings) and web search are pluggable by configuration, so the app runs fully hosted or fully local.

## Repository layout

```
libs/shared/        Shared library (core, models, db, repositories, providers, observability, prompt)
services/api/       FastAPI + LangGraph agent service
services/ingestion/ Celery ingestion pipeline (worker + beat)
services/frontend/  React + Vite SPA
docker/             Dockerfiles and provisioning (migrate image, prometheus, grafana)
infra/terraform/    AWS Terraform (Milestone 10) — VPC + ECR today; more modules later
spec/               Authoritative design: requirements, architecture, tasks (roadmap)
docker-compose.yml  Local full-stack orchestration
Makefile            Local Compose + Terraform wrappers (`make help`)
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.13 is fetched automatically)
- Docker + Docker Compose (for the full stack)
- Terraform **1.15.x** and AWS credentials (only if applying Milestone 10; see `infra/terraform/README.md`)

## Setup

```bash
cp .env.example .env          # local defaults are wired for docker compose
uv sync --all-packages        # create the workspace venv with every member installed
```

> Use `uv sync --all-packages`, not plain `uv sync` — the workspace root is virtual, so a plain sync installs only tooling and omits the service packages.

## Run the full stack

```bash
make up-d                          # docker compose up --build -d
make watch                         # docker compose watch (sync + reload)
make down                          # stop containers, keep volumes
```

Equivalent Compose commands (`docker compose up --build`, `docker compose watch`) still work.

AWS deploy is sequenced so ECS never starts before images exist: copy `infra/terraform/backend.hcl.example` and `environments/dev.tfvars.example`, then `make tf-init` and `make aws-bootstrap` (ECR → push images → remaining Terraform). See [`infra/terraform/README.md`](infra/terraform/README.md).

A one-shot `migrate` container runs database migrations to completion, then the services start. Once up:

| URL | Service |
|---|---|
| http://localhost:8000/api/v1/docs | API (OpenAPI UI) |
| http://localhost:8000/metrics | API Prometheus metrics |
| http://localhost:5173 | Frontend |
| http://localhost:8080 | Adminer (Postgres UI) |
| http://localhost:3000 | Grafana (admin / admin) |
| http://localhost:9090 | Prometheus |
| http://localhost:9001 | MinIO console |

Optional local LLM: `docker compose --profile ollama up`.

## Run a service without Docker

```bash
uv run uvicorn api.app:create_app --factory --port 8000    # API (serves /api/v1/health, /metrics)
uv run celery -A ingestion.celery_app:app worker -l INFO   # ingestion worker
uv run celery -A api.eval_worker.celery_app:app worker --queues=eval -l INFO  # eval worker
```

## Tests

```bash
make test-unit                 # uv run pytest -m unit (fast, no I/O)

# The integration tier runs against throwaway infra (Postgres/Qdrant/MinIO) on
# offset ports, isolated from the dev stack. Start it first; the tier skips
# cleanly (with a hint) if it isn't running.
make test-infra
uv run pytest -m integration   # repositories/services/pipeline vs real infra
uv run pytest -m functional    # HTTP-level against the app (DB mocked at the boundary)
make test-infra-down           # tear down when done

cd services/frontend && npm install && npm test
```

## Lint & format

```bash
uv run ruff check .            # lint
uv run ruff format .           # format
```

## Environment variables

All configuration is read from the environment (or a local `.env`). Every variable — database/Qdrant/Redis URLs, object storage, LLM/embeddings/web-search provider selection and keys, JWT, Google OAuth, optional Langfuse — is documented in [`.env.example`](.env.example). Secrets are never committed; `.env` is git-ignored.
