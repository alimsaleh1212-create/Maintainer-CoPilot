# MON Step 8 — Redaction Layer

**Date:** 2026-05-19
**Branch:** feature/foundations-skeleton

## What was built

- **`backend/app/infra/redaction.py`** — `redact_text(str) → str` scrubs 9 secret patterns (Anthropic, OpenAI/generic sk-, Gemini, GitHub PAT classic+new, Langfuse, AWS IAM, JWT, PEM headers). `redact_dict(dict) → dict` for Langfuse span attributes (non-strings pass through unchanged). `structlog_redaction_processor` typed to `MutableMapping` so it slots cleanly into structlog's processor chain.
- **`backend/app/api/main.py`** — Redaction processor wired into `_configure_logging()` as the fourth step in the structlog pipeline, before `JSONRenderer`. Every log emission now scrubs secrets automatically.
- **`docs/SECURITY.md`** — Pattern table with regex and replacement labels, secrets list (names only), CSP/CORS posture, audit log fields, refuse-to-boot policy, response shape policy.

## Tests written

- **`tests/unit/test_redaction.py`** — 18 tests across four classes:
  - `TestRedactTextPatterns` — one test per secret type + clean text + multi-key string.
  - `TestLoggerPath` — structlog processor scrubs `event` field and arbitrary fields; non-strings pass through.
  - `TestTracesPath` — `redact_dict` scrubs span attribute values; non-string metric values preserved.
  - `TestMemoryPath` — `redact_text` applied to episodic memory summaries before pgvector write.

## Checks run

- `uv run ruff check .` → all checks passed
- `uv run mypy app/` → no issues (21 source files)
- `uv run pytest -q` → 26 passed, 1 deselected

## What's next

MON Step 9 — Dataset fetch script (`scripts/fetch_issues.py`) to pull closed MONAI issues from GitHub API and `scripts/build_splits.py` for stratified time-aware train/val/test split with the 3-class label merge.
