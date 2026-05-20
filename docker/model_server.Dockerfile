FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY pyproject.toml uv.lock .python-version ./
# If pyproject.toml and uv.lock are out of sync (e.g., torch removed from project),
# regenerate the lock to match. After first build, this is a no-op.
RUN uv sync --frozen --no-dev || (uv lock && uv sync --frozen --no-dev)

FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Model weights are downloaded from MinIO at startup by the lifespan handler.
EXPOSE 8001
