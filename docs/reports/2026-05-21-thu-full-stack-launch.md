# Full Stack Launch — 2026-05-21

## What Was Built

- **Full Docker stack** running: api, model-server, chatbot (Streamlit), widget, demo hosts, Postgres, Redis, MinIO, Vault, Ollama, Langfuse, pgAdmin.
- **Model-server** now pulls DistilBERT weights from MinIO on container start (no bind mount). Upload script: `scripts/upload_model_to_minio.py`.
- **Auth** fixed: replaced passlib (incompatible with bcrypt 5.0) with direct `bcrypt` calls in `app/api/routes/auth.py`.
- **All routes registered** in `app/api/main.py`: classify, ner, summarize added alongside existing auth, chat, rag, widgets, embed, memory.
- **Chunker fixed** (`app/rag/chunking.py`): added `_split_large_paragraph()` to handle documents with no sub-headers (previously produced 34k-char chunks crashing Ollama).
- **Ingest** (`app/rag/ingest.py`): fixed SQL parameter shadowing (`text` param shadowed SQLAlchemy `text()`), switched to `CAST(:embedding AS vector)` syntax, JSON-serialized metadata.
- **Corpus ingest** script: `scripts/ingest_corpus.py` — runs inside the api container to embed and store MONAI issues in pgvector.
- **Widget created**: `wgt_49595af3` with `http://localhost:8090` as allowed origin. Demo host HTML updated with real widget ID.
- **CSP verified**: `GET /embed?widget_id=wgt_49595af3` returns `frame-ancestors http://localhost:8090`.
- **docs/RUNBOOK.md** written (was missing).

## Tests Written

- All 78 existing unit tests pass (`uv run pytest -q` → 78 passed).
- Manual endpoint tests: `/auth/register`, `/auth/login`, `/classify` (→ bug label), `/ner` (→ entities), `/embed` (→ CSP headers).

## Checks Run

- `uv run pytest -q` → **78 passed**, 10 deselected
- All Docker services healthy: api, model-server, chatbot, widget, host-allowed, host-disallowed, db, redis, minio, vault, ollama
- Corpus ingestion: ~700+ chunks in pgvector and growing

## What's Next

- Wait for `llama3.2:1b` to finish downloading (Ollama chat fallback)
- Re-run full corpus ingestion to completion
- Test chat endpoint end-to-end
- Commit all changes and run final `git tag v0.1.0-week7`
