# Ingestion pipeline service image (Celery worker / beat). Build context is the
# REPO ROOT.
#
#   docker build -f services/ingestion/Dockerfile -t recap-ingestion .
#
# The default CMD runs a Celery worker; the compose `ingestion-beat` service
# overrides the command to run beat. Built with the `local` extra
# (torch/sentence-transformers) so the HuggingFace embedder can run on-box —
# EMBEDDINGS_PROVIDER=huggingface. This makes the image large (torch pulls the
# CUDA stack on linux); for a hosted-only footprint, drop `--extra local`.

# ---- builder ---------------------------------------------------------------
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependency layer (cached until lock / manifests change).
COPY pyproject.toml uv.lock ./
COPY libs/shared/pyproject.toml libs/shared/pyproject.toml
COPY services/api/pyproject.toml services/api/pyproject.toml
COPY services/ingestion/pyproject.toml services/ingestion/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package ingestion --extra local

# 2) Project layer.
COPY libs/shared/ libs/shared/
COPY services/ingestion/ services/ingestion/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package ingestion --extra local

# ---- runtime ---------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# Pre-create the HuggingFace cache dir owned by the non-root runtime user. The
# local embedder downloads model files here; when the compose `hf_cache` named
# volume is first mounted, Docker populates it from this image path and so the
# empty volume inherits `app` ownership — otherwise a fresh volume is root-owned
# and the `app` process gets PermissionError creating the model blobs.
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

# Default: Celery worker. compose overrides `command:` for the beat service.
CMD ["celery", "-A", "ingestion.celery_app:app", "worker", "--loglevel=INFO"]
