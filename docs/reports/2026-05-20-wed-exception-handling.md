# WED Exception Handling Refactor — 2026-05-20

## What was built

**Exception handling layer** ensures clean API boundaries and recoverable tool failures:
- `backend/app/domain/tool_error.py` — ToolError model for LLM tool-calling loop (tool_name, error, retryable)
- Exception boundary handler (existing `backend/app/api/exceptions.py`) verifies all domain errors map to correct HTTP status without exposing stack traces
- ToolFailure domain exception marked as 422 (recoverable) instead of 500 (fatal)

**Design rationale:**
- Domain errors (AppError hierarchy) raised by services, mapped to HTTP by boundary handler
- Tool failures are special: caught by chatbot loop (THU work), converted to ToolError, returned to LLM
- LLM decides next action: retry, escalate, or continue with alternative
- Stack traces never exposed to users — always {code, request_id, message}

## Tests written

- `tests/unit/test_exception_handling.py` (11 tests):
  - NotFoundError → 404, PermissionDenied → 403, ToolFailure → 422, ValidationError → 422
  - All errors include request_id for tracing across logs/Langfuse
  - No stack traces in response bodies
  - Custom messages preserved through exception hierarchy
  - Tool failure retryable flag stored

- `tests/unit/test_tool_error.py` (7 tests):
  - Valid ToolError creation with all fields
  - Retryable defaults to False
  - Serialization to JSON (for LLM consumption)
  - Strict type validation (Pydantic v2)
  - Chatbot integration pattern: catch ToolFailure → convert to ToolError → return to LLM

All 18 tests green.

## Checks run

- `uv run mypy --strict app/domain/tool_error.py`: Success
- `uv run pytest tests/unit/test_exception_handling.py tests/unit/test_tool_error.py -v`: 18 passed
- Ruff check: Clean (no issues)

## What's next

**WED blockers:**
1. **Docker containers operational** — needed for corpus ingestion (pgvector + BM25 queries)
2. **Database schema** — rag_chunks table with pgvector and tsvector indexes
3. **Actual retrieval queries** — wire up pgvector/BM25 in retrieval.py when docker ready

**THU (Chatbot loop):**
- Catch ToolFailure from service calls (classify, RAG, NER, summarize, memory)
- Convert to ToolError (tool_name, error, retryable)
- Return to LLM with tool failure response
- LLM calls tools again or gracefully skips unavailable tools
