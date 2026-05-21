# RAG Ingest Completion + Ablation Study

**Date:** 2026-05-21  
**Branch:** `feature/foundations-skeleton`

## What was built

- **`backend/app/rag/ingest.py`** — added `_existing_chunk_ids()` dedup: pre-queries DB before embedding, skips already-present chunks; restart now costs O(new) not O(all)
- **`backend/app/rag/embeddings.py`** — added `typing.Any` import, explicit type annotations on `embed_batch()` return; fixes mypy `no-any-return` error
- **`backend/app/rag/citations.py`** — fixed mypy `arg-type` error in `_get_score()` via intermediate `raw: Any` variable
- **`backend/scripts/ablate_rag.py`** — added p50/p95 latency columns and Δ score vs baseline A to summary table
- **`docs/RAG_ABLATION.md`** — full ablation report: 4 configs × 10 probes, accuracy + latency, findings, config-D selection rationale

## Corpus ingested

- **Issues:** 2878 chunks (1632 already present + 1246 new, 55 skipped as < 50 chars, 0 errors)
- **Wiki:** 162 chunks from 31 pages (all new, 0 errors)
- **Total:** 3040 chunks in `rag_chunks` table (`source: docs | issue`)

## Tests written / fixed

- `backend/tests/unit/test_rag_service.py` — fixed `mock_retriever` fixture: added explicit `metadata`, `parent_id`, `parent_text` fields to `MagicMock` to avoid `MagicMock` objects leaking into assertion
- All 135 unit tests passing after fix

## Checks run

- `uv run ruff check app/rag/` — clean
- `uv run mypy app/rag/ --ignore-missing-imports` — clean (was 2 errors, now 0)
- `uv run pytest tests/unit/ -q` — 135 passed

## What's next

Run the classification eval (`eval/run_classification_eval.py`) and wire the
`/chat` endpoint so Gemini can call the `rag_search` tool with source filtering.
