# WED RAG Service + Endpoint — 2026-05-20

## What was built

**RAG service orchestration layer** bridges multi-query expansion, hybrid retrieval, and ranked results:
- `backend/app/services/rag.py::RAGService` — orchestrates expander → retriever → format conversion
- `backend/app/api/routes/rag.py::POST /rag/search` — HTTP endpoint (query, top_k: 1-20)
- Embedding model loading in lifespan startup → refuse-to-boot on failure

**Key improvements to existing code:**
- `backend/app/rag/retrieval.py` — added `top_k` parameter to `retrieve()` for caller control
- Type annotations on all retrieval methods (embedding_fn, db_session, reranker)

## Tests written

- `tests/unit/test_rag_service.py` (8 tests):
  - Happy path, empty results, top_k parameterization
  - Query expansion, expansion failure → ToolFailure, retrieval failure
  - Result conversion, rerank score precedence
- `tests/integration/test_rag_endpoint.py` (7 tests):
  - Valid request → 200 + schema, missing query → 422, invalid top_k → 422
  - Default top_k=5, empty chunk list handling, response schema validation

All 15 tests green.

## Checks run

- `uv run ruff check`: 1 B008 flag (FastAPI idiom `Depends()` in defaults — acceptable, used consistently)
- `uv run mypy --strict`: Success (added type annotations to retrieval methods)
- `uv run pytest tests/unit/test_rag_service.py tests/integration/test_rag_endpoint.py -q`: 15 passed

## What's next

Exception handling refactor (WED step 11) — boundary handler for tool failures + graceful degradation when classifier/RAG services are down.

## Dependencies added

- `sentence-transformers==5.5.1` (BAAI/bge-small-en-v1.5 embedding model loading)
