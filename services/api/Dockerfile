# Assistant/API service image. Build context is the REPO ROOT (needs the root
# workspace manifest + uv.lock + libs/shared + this service).
#
#   docker build -f services/api/Dockerfile -t recap-api .
#
# Multi-stage: a builder installs deps into /app/.venv from the committed lock,
# then a slim runtime carries only the venv + source and runs as non-root. The
# API depends on `shared[local]` unconditionally (FR-19's local transcription/
# vision providers need transformers/torch even in an otherwise-hosted
# deployment, since TRANSCRIPTION_PROVIDER/VISION_PROVIDER are runtime config,
# not a build-time choice) — the runtime image also needs the `ffmpeg` binary,
# since `transformers`' audio pipeline shells out to it to decode arbitrary
# input formats (webm/mp4/ogg) rather than accepting only raw PCM.

# ---- builder ---------------------------------------------------------------
FROM python:3.13-slim AS builder

# uv provides fast, reproducible installs from uv.lock (pinned uv version).
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependency layer — cached until the lock or any member manifest changes.
#    All workspace member manifests are needed for uv to resolve the workspace.
COPY pyproject.toml uv.lock ./
COPY libs/shared/pyproject.toml libs/shared/pyproject.toml
COPY services/api/pyproject.toml services/api/pyproject.toml
COPY services/ingestion/pyproject.toml services/ingestion/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package api

# 2) Project layer — copy source and install the api + shared packages (editable).
COPY libs/shared/ libs/shared/
COPY services/api/ services/api/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

# ---- runtime ---------------------------------------------------------------
FROM python:3.13-slim AS runtime

# ffmpeg: required by transformers' local ASR pipeline (HuggingFaceTranscriber)
# to decode a voice note's raw bytes; not needed by the hosted OpenAI Whisper
# path, but that's a runtime config choice, not a build-time one, so it's
# always installed here.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# Put the venv on PATH; no secrets baked in (injected at runtime).
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# create_app is a factory, so uvicorn needs --factory.
CMD ["uvicorn", "api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
