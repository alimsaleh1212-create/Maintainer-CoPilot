FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install system deps needed by psycopg2-binary, spaCy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- dependency layer (cached unless lockfile changes) ----
FROM base AS deps
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# ---- runtime image ----
FROM base AS runtime
COPY --from=deps /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

EXPOSE 8000
