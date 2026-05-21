# Critical Gaps Fix — 2026-05-21

## What was built

- **Langfuse TracingClient wired end-to-end**
  - `app/api/main.py` lifespan: `TracingClient.from_keys(...)` constructed and stored on `app.state.tracer`; `tracer.flush()` called on shutdown
  - `app/api/dependencies.py`: `get_tracer()` dependency added; `TracerDep` alias exported; `get_chatbot_service()` now passes `tracer=` to constructor
  - `app/services/chatbot.py`: `ChatbotService.__init__` accepts `tracer: TracingClient`; `_call_llm()` wraps primary+fallback in a `"llm_call"` span; `_execute_tool()` wraps each tool dispatch in a `"tool_<name>"` span; `chat()` starts a `"chat_turn"` trace at the top of every turn

- **Stub tools replaced with real implementations**
  - `extract_entities`: calls `app.ml.ner.extract_entities(text)` (spaCy + regex, dispatched via `asyncio.to_thread`)
  - `summarize_text`: calls `self._primary.chat()` with a 150-word summarize prompt (uses the same `PrimaryLLMDep` LLM abstraction)

- **F821 fixed** in `app/rag/ingest.py:64` — `RagChunkCreate` (undefined) replaced with a plain `dict[str, Any]` built inline

- **Ruff: 0 errors** — added per-file-ignores for B008 (FastAPI Depends in routes), E402 (alembic env.py), ASYNC240 (scripts/model_server), S105/S106/S603/S607 (tests); ran `--fix` auto-correcting 22 additional issues (UP017, I001, F401, F541, F841)

## Tests written

No new tests added in this commit — fixing pre-existing gaps. 71/71 existing unit tests still pass.

## Checks run

- `uv run ruff check .` → **All checks passed** (0 errors, down from 65)
- `uv run pytest tests/unit/ -q` → **71 passed**

## What's next

Write the missing test suite: `test_short_term_memory_ttl.py`, `test_long_term_memory_retrieval.py`, `test_write_memory_audit_log.py`, `test_csp_frame_ancestors.py`, `test_redaction_in_traces.py`
