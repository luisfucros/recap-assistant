# One-shot migration image. Build context is the REPO ROOT.
#
#   docker build -f docker/migrate.Dockerfile -t recap-migrate .
#
# Runs `alembic upgrade head` to completion and exits. This is the SINGLE
# component that writes the schema — services never migrate on startup; they
# gate on this container finishing (compose `service_completed_successfully`).
# Installs only the `shared` package (which owns the models + Alembic config).

# ---- builder ---------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependency layer (all member manifests needed for workspace resolution).
COPY pyproject.toml uv.lock ./
COPY libs/shared/pyproject.toml libs/shared/pyproject.toml
COPY services/api/pyproject.toml services/api/pyproject.toml
COPY services/ingestion/pyproject.toml services/ingestion/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package shared

# 2) Project layer — only the shared library is needed to migrate.
COPY libs/shared/ libs/shared/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package shared

# ---- runtime ---------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

# env.py sources DATABASE_URL from the environment; the ini path is stable.
CMD ["alembic", "-c", "libs/shared/alembic.ini", "upgrade", "head"]
