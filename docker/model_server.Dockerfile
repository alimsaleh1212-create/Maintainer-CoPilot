FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ONLY the model-server group (torch + transformers, CPU-only).
# Main project deps (api, redis, sqlalchemy, etc.) are NOT installed here.
FROM base AS deps
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --group model-server --no-cache

FROM deps AS runtime
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Stay in /app so "model_server.main" is importable as a package
EXPOSE 8001
