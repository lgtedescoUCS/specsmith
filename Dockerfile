# Multi-stage build: uv resolves the locked environment in the builder, and
# the runtime image gets only that environment plus the files the app reads
# at run time. Pattern from https://docs.astral.sh/uv/guides/integration/docker/

# --- builder ---------------------------------------------------------------
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Use the image's Python (the runtime stage has the same one at the same
# path, which the venv points to), precompile bytecode for faster starts,
# and skip dev dependencies and uv's cache.
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Dependencies first, in their own layer: rebuilt only when the lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Then the project itself. It stays an editable install on purpose: the
# prompt loader finds prompts/ relative to the source tree, so the runtime
# image keeps the /app layout instead of a copy inside site-packages.
COPY src ./src
RUN uv sync --locked

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim-trixie

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY prompts ./prompts
COPY migrations ./migrations
COPY alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app

# Render sends traffic to $PORT (10000 unless set). Migrations run before
# the server on every start: the free plan has no pre-deploy step, and with
# a single instance there is no race between two upgrades. A failed
# migration stops the start, so the old deploy keeps serving.
EXPOSE 10000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn specsmith.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
