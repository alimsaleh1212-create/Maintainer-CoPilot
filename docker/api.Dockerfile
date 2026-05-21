FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# System deps required by psycopg2-binary, spaCy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- dependency layer (rebuild only when lockfile changes) ----
FROM base AS deps
COPY pyproject.toml uv.lock .python-version ./
# uv sync creates /app/.venv with all packages; the runtime stage extends deps
# so the venv is already present — no cross-stage copy needed.
RUN uv sync --frozen --no-dev --no-cache

# ---- runtime image (extends deps — .venv already in this layer) ----
FROM deps AS runtime
COPY . .
ENV PYTHONUNBUFFERED=1
# Add venv to PATH so python/uvicorn/alembic scripts are found without uv run.
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
