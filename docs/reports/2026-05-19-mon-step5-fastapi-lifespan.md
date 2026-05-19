# MON Step 5 — FastAPI Lifespan, Domain Exceptions, Boundary Handler

**Date:** 2026-05-19
**Branch:** feature/foundations-skeleton

## What was built

- **`app/api/main.py`** — FastAPI app factory with `lifespan()` context manager. Builds DB engine (SQLAlchemy async) + Redis pool at startup. Refuses to boot (sys.exit 1) if Vault unreachable, secrets missing, DB unreachable, or Redis unreachable. Stores all singletons on `app.state`; classifier/Langfuse stubs noted with [wired TUE/WED] comments.
- **`app/domain/errors.py`** — Domain exception hierarchy: `AppError` → `NotFoundError`, `PermissionDenied`, `ToolFailure` (with `retryable` flag), `ValidationError`. Never `HTTPException` in services.
- **`app/api/exceptions.py`** — Single `add_exception_handlers(app)` mapping domain → HTTP. Users see `{code, message, request_id}`; no stack traces ever.
- **`app/api/dependencies.py`** — `Depends()` providers for settings, DB session (with rollback on error), and Redis. `SettingsDep`, `DbSessionDep`, `RedisDep` type aliases for routes.
- **`app/api/routes/health.py`** — `/healthz` endpoint returning `{status, environment}`.
- Runtime deps added: `fastapi`, `uvicorn[standard]`, `sqlalchemy[asyncio]`, `asyncpg`, `redis[hiredis]`, `structlog`, `httpx`.

## Tests written

No new unit tests added for this step (the lifespan refuse-to-boot integration tests require running containers; tagged `@pytest.mark.integration` — written WED alongside compose smoke test). Existing 8 tests still pass.

## Checks run

- `uv run ruff check .` → all checks passed
- `uv run mypy app/` → no issues (18 source files)
- `uv run pytest -q` → 8 passed

## What's next

MON Step 6 — Alembic baseline migration: `users`, `widgets`, `conversations`, `messages`, `memories` (pgvector 384-dim), `audit_log` tables with pgvector extension enabled.
