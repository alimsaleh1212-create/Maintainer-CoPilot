# Step Report: Missing Test Files

**Date:** 2026-05-21
**Branch:** `feature/foundations-skeleton`

## What was built

- **`tests/unit/test_short_term_memory_ttl.py`** — Redis short-term memory TTL contract: `get_history`, `append_message` (key format, TTL reset on every write, history preservation), `clear_history`, and `_HISTORY_TTL_SECONDS == 86400` constant assertion.
- **`tests/unit/test_long_term_memory_retrieval.py`** — pgvector long-term memory: save/embed/redact pipeline, multi-tenant isolation (`search_memories` user-scoped), delete ownership enforcement via `NotFoundError` on wrong owner.
- **`tests/unit/test_write_memory_audit_log.py`** — Audit log contract for every `save_memory` and `delete_memory` call: exactly 1 `AuditLog` row, correct `action`/`actor_id`/`target_type`/`target_id`, both `Memory` + `AuditLog` in same atomic transaction, redacted summary length in metadata.
- **`tests/unit/test_csp_frame_ancestors.py`** — `_build_frame_ancestors()` pure function + `/embed` route via `TestClient` with `dependency_overrides`: CSP header derives from `widget.allowed_origins`, 404 on missing/disabled widget, no stack trace in error body, HTML content-type, CORS header from allowlist.
- **`tests/unit/test_redaction_in_traces.py`** — `TracingClient` redaction pipeline: `start_span()` input, `end_span()` output, `span()` CM auto-end on exception, no-op mode (None client, placeholder keys → `_is_noop=True`), end-to-end guarantee that fake API keys never reach Langfuse.
- **`tests/unit/test_widget_bundle_size.py`** — Bundle size gate: skips if `frontend/widget/dist/widget.js` not built; asserts gzip-compressed size ≤ 200 KB; asserts file exists at expected path.

## Tests written

| File | Count | Key assertions |
|------|-------|----------------|
| `test_short_term_memory_ttl.py` | 12 | Redis key format, TTL=86400, empty on None, corrupt JSON → [] |
| `test_long_term_memory_retrieval.py` | 10 | Redaction before embed/store, multi-tenant isolation, ownership-enforced delete |
| `test_write_memory_audit_log.py` | 10 | 1 audit row per write, action/actor/target correct, atomic commit |
| `test_csp_frame_ancestors.py` | 13 | CSP header, CORS, 404 paths, no stack trace, HTML content-type |
| `test_redaction_in_traces.py` | 14 | Key scrubbed from span input/output/metadata, no-op mode, CM lifecycle |
| `test_widget_bundle_size.py` | 3 | ≤200KB gzip, file exists, skips gracefully if not built |

**Total: 62 new tests. Suite: 135 passing (up from 73 before this session).**

## Checks run

```
uv run ruff check . → All checks passed!
uv run pytest -q -m "not eval" → 135 passed in 2.58s
```

Key fix during writing: `Memory.__new__(Memory)` raises `AttributeError: '_sa_instance_state'` — replaced with `MagicMock()` in all test fabrication helpers to avoid SQLAlchemy 2.x instrumentation.

## What's next

Wire the eval test suites (`tests/eval/`) for classification and RAG golden-set thresholds, then run the full acceptance checklist (item 4–6) before Friday tagging.
