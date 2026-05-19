# MON Step 6 — Alembic Baseline Migration

**Date:** 2026-05-19
**Branch:** feature/foundations-skeleton

## What was built

- **`backend/alembic.ini`** — Alembic config; no DB URL hardcoded — URL resolved from Vault at migration time.
- **`backend/alembic/env.py`** — Async migration runner (`create_async_engine` + `run_sync`). Falls back to `DATABASE_URL` env var in offline/CI-no-vault mode.
- **`backend/app/repositories/base.py`** — `DeclarativeBase` shared by all ORM models.
- **`backend/app/repositories/models.py`** — Six SQLAlchemy 2.x ORM models (`User`, `Widget`, `Conversation`, `Message`, `Memory`, `AuditLog`). `Memory.embedding` uses `Vector(384)` from pgvector. `AuditLog.extra_metadata` stores redacted JSON context.
- **`backend/alembic/versions/001_c0ffee_baseline.py`** — Initial migration: enables `pgvector` extension, creates all six tables with indexes. `memories` table gets an IVFFlat cosine-similarity index (`lists=100`, suitable to ~1M rows).

## Tests written

- **`tests/integration/test_migrations.py`** — Placeholder stub tagged `@pytest.mark.integration`; skipped in pre-commit. Full testcontainers-backed test (upgrade head → assert tables → downgrade base) wired on WED when testcontainers is added.

## Checks run

- `uv run ruff check .` → all checks passed
- `uv run mypy app/` → no issues (20 source files)
- `uv run pytest -q` → 8 passed, 1 deselected (integration placeholder)

## What's next

MON Step 7+8 — Redaction layer (`app/infra/redaction.py`) with the mandatory redaction test, then the label mapping and dataset fetch scripts.
