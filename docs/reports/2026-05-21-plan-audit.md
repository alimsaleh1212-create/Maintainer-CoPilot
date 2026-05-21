# PLAN.md Audit — 2026-05-21

Audit of completed features against `docs/PLAN.md`, executed after the
LLM-abstraction + Vault-init refactor commits (213da6c, 356a19c).

## ✅ Done — production-ready

### MON (foundations)
- Repo layout matches plan (`backend/ app/{api,services,repositories,domain,infra,ml,rag,tools}`)
- `docker-compose.yml` brings up 13 services; vault-init seeds from `.env` via [docker/vault-init.sh](docker/vault-init.sh)
- Vault adapter; refuse-to-boot on missing keys
- `pydantic-settings` Settings(extra="forbid"); `@lru_cache(maxsize=1) get_settings()`
- FastAPI lifespan builds DB engine, Redis, embedder, classifier, **Gemini + Ollama clients**
- Alembic baseline migration; pgvector + `users/widgets/conversations/messages/memories/audit_log`
- Domain exceptions (`NotFoundError, PermissionDenied, ToolFailure, ValidationError`) + boundary handler
- Redaction layer with pattern set; 71/71 unit tests passing
- MONAI dataset fetched: `bug=337, feature_request=535, documentation=28, questions=250`; 3-class merge to `bug/feature/support`

### TUE (DL Track)
- DistilBERT fine-tuned. **eval_report.json**:
  - DistilBERT: 76.4% macro-F1 (winner)
  - TF-IDF+LogReg: 72.3%
  - Gemini-2.5-flash: 64.4%
- DistilBERT artifact in MinIO; `model-server` container serves `/predict`
- `/classify` endpoint working (label, confidence, model_version, latency_ms)
- `/ner` endpoint: spaCy `en_core_web_sm` + regex (FunctionName, ErrorType, FilePath, PackageName, ORG, PERSON, …)
- `/summarize` endpoint refactored to use injected `PrimaryLLMDep` (no more `os.getenv`)

### WED (RAG)
- Corpus ingested (2,933 issues → pgvector via `nomic-embed-text` 768-dim)
- Markdown-aware chunking; hybrid retrieval (dense pgvector + sparse BM25)
- Multi-query rewrite via Gemini Flash
- `/rag/search` returns ranked chunks; **fixed `query_embedding` SQL binding** (CAST not `::`)
- Exception handler maps every domain error → `{code, request_id, message}` (no stack traces)
- Redaction integrated into chatbot persists-to-Redis path

### THU (Chat + Auth + Widget)
- `fastapi-users` + JWT auth working (`/auth/register`, `/auth/login`)
- Tool-calling chatbot: Gemini → Ollama fallback; LLM picks tools (verified: `classify_issue` invoked on test prompt)
- 5 tools registered: `classify_issue`, `rag_search`, `write_memory`, `extract_entities`, `summarize_text`
- Short-term memory: Redis (24h TTL); Long-term: pgvector episodic
- `write_memory` produces `audit_log` row per call ([app/services/memory.py:148](backend/app/services/memory.py#L148))
- `/embed` route sets `Content-Security-Policy: frame-ancestors` from widget allowlist
- Streamlit pages: `1_chat.py`, `2_admin_widgets.py`, `3_memory_inspector.py`
- React widget structure: `src/{App,Chat,api,types}.tsx`; `dist/widget.js` builds in CI
- Demo hosts: `frontend/demo_host/{allowed,disallowed}/`
- CI workflow runs ruff, mypy, secret hygiene, tests, bundle-size check

---

## ⚠️ Gaps — work needed before Friday

### Critical (graded)

| # | Gap | Impact | Where |
|---|---|---|---|
| 1 | **Langfuse tracing built but never instantiated** in lifespan or used in chatbot | Demo step 8 fails ("Trace tree demo: open Langfuse, walk a real /chat conversation") | [app/infra/tracing.py](backend/app/infra/tracing.py) exists; zero imports outside its own file |
| 2 | **2 of 5 chatbot tools are stubs** — `extract_entities` returns empty list with note; `summarize_text` truncates instead of calling LLM | Demo prompts that use NER or summarize via chat return obviously fake results | [app/services/chatbot.py:349-358](backend/app/services/chatbot.py#L349-L358) |
| 3 | **`rag/ingest.py` has F821 undefined `RagChunkCreate`** | mypy/ruff fail in CI on main; future ingest runs may crash | [app/rag/ingest.py:64](backend/app/rag/ingest.py#L64) |
| 4 | **65 ruff errors** (mostly B008 false positives on FastAPI Depends; some I001/F401) | CI red on lint step | Various — see `uv run ruff check .` |

### High (test coverage gaps)

| # | Gap | Plan reference |
|---|---|---|
| 5 | `tests/eval/` directory is empty (`__init__.py` only) | THU step 14 "both eval suites in CI" |
| 6 | No `test_short_term_memory_ttl.py` | THU tests |
| 7 | No `test_long_term_memory_retrieval.py` (multi-tenant isolation) | THU tests |
| 8 | No `test_write_memory_audit_log.py` | THU tests |
| 9 | No `test_csp_frame_ancestors.py` / `test_cors_from_db.py` | THU tests |
| 10 | No `test_widget_bundle_size.py` (CI step exists, no pytest version) | THU tests |
| 11 | `tests/e2e/` directory does not exist (Playwright widget-in-host test) | THU tests, FRI demo |
| 12 | No `test_redaction_in_traces.py` | WED tests |

### Medium (docs/polish — FRI work)

- `docs/ARCH.md`, `DECISIONS.md`, `RUNBOOK.md`, `EVALS.md`, `SECURITY.md` all exist but should be re-read end-to-end before tagging
- Widget styling polish deferred to FRI AM
- `README.md` setup section should reference new vault-init flow

---

## 🔧 Code-quality / CLAUDE.md compliance

| Rule | Status |
|---|---|
| No `os.getenv` outside config/vault | ✅ Clean |
| No `requests` (httpx everywhere) | ✅ Clean |
| No `time.sleep` (await asyncio.sleep) | ✅ Clean |
| No bare `except:` | ✅ Clean |
| Domain exceptions, not HTTPException in services | ✅ Clean |
| Vault required keys + refuse-to-boot | ✅ Working |
| Redaction wired to logger + memory writer | ✅ Working |
| Redaction wired to Langfuse spans | ⚠️ Spans never created — see gap #1 |

---

## 🎬 Demo readiness

| Demo step (FRI rehearsal) | Ready? |
|---|---|
| 1. Fresh-clone `docker compose up` → green | ✅ |
| 2. Widget loads in allowed host, blocked in disallowed | ⚠️ Needs e2e verification in browser |
| 3. Tool-calling chat triages a real issue | ✅ (Gemini calls `classify_issue`) |
| 4. RAG search returns relevant chunks | ✅ |
| 5. Trace tree in Langfuse | ❌ Tracing not wired up |
| 6. Cross-conversation memory recall | ⚠️ Code path exists, untested end-to-end |
| 7. Eval suites pass in CI | ⚠️ Eval scripts exist; no pytest-tagged tests |

---

## 📋 Recommended next steps (priority order)

1. **Fix F821 in `rag/ingest.py`** — quick win, unblocks ruff
2. **Wire up Langfuse tracing**: instantiate `TracingClient.from_keys(...)` in lifespan; wrap `chatbot.chat()`, `_call_llm()`, `_execute_tool()`, RAG retrieval in spans
3. **Implement the 2 stub tools** — call NER and summarize services from chatbot
4. **Add `tools/extend the per-file ruff ignore for B008** in `[tool.ruff.lint.per-file-ignores]` for `app/api/routes/**` (FastAPI standard pattern)
5. **Write missing tests** (THU test list above) — particularly the redaction-in-traces one (graded)
6. **Manual demo dry-run** — e2e widget on both hosts, memory survival across conversations, Langfuse trace walk

Total estimated work: **half a day** to close all critical+high gaps. The foundation is solid; the gaps are surfacing wiring + tests, not architecture changes.
