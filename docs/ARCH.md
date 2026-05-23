# Architecture — Maintainer's Copilot

## Top-level layout

```
project7/
├── backend/        Python services (FastAPI API + model-server inference)
├── frontend/       All user-facing surfaces (React widget, Streamlit UI, demo host)
├── docker/         All container infra (Dockerfiles + docker-compose.yml)
├── docs/           Architecture, decisions, runbook, evals, security
├── tests/e2e/      Cross-cutting Playwright browser tests
└── .github/        CI workflow
```

## Backend layers (strict boundaries)

```
backend/app/
├── api/            HTTP routers ONLY. No SQLAlchemy, no Redis, no external calls.
│   ├── routes/     One file per resource (auth, chat, classify, rag, widgets, embed, memory)
│   ├── dependencies.py   Depends() — current_user, db_session, llm, embedder
│   └── exceptions.py     Boundary handler: domain errors → {code, request_id, message}
├── services/       Business logic, tx boundaries, cache/memory invalidation
│   ├── chatbot.py  Tool-calling LLM loop (5-round max)
│   ├── classification.py  Routes to model-server, graceful degradation
│   ├── rag.py      Hybrid retrieval + rerank + query rewrite
│   ├── memory.py   Redis short-term (24h TTL) + pgvector long-term (episodic)
│   └── widget.py   CRUD + ownership checks + audit log
├── repositories/   SQL ONLY. No HTTP errors. No cache invalidation.
├── domain/         Pydantic models (distinct from ORM) + domain exceptions
├── infra/          External adapters
│   ├── vault.py           Boot-time secret resolution (refuse-to-boot on failure)
│   ├── redaction.py       Regex redaction for logs, traces, memory writes
│   ├── tracing.py         Langfuse span helpers
│   ├── llm/               GeminiClient (primary) + OllamaClient (fallback)
│   ├── redis_client.py    Redis wrapper
│   ├── minio.py           MinIO blob store
│   └── model_server_client.py  HTTP client with retry + ToolFailure on error
├── ml/             Runtime-only ML loaders (NER, summarizer, classifier loader)
├── rag/            Corpus build, chunking, embeddings (Ollama), retrieval, reranker
└── tools/          LLM-callable tool wrappers (classify, ner, summarize, rag_search, write_memory)
```

**Layer rule**: a new endpoint = new route file + Pydantic domain model + service method + repo method (if SQL). Zero churn elsewhere.

## Data flow

```
Browser/Widget
    │  POST /chat
    ▼
api/routes/chat.py          (auth check, request_id, trace start)
    │
    ▼
services/chatbot.py          (build messages, inject memory, LLM tool-calling loop)
    ├── tools/classify_tool  → infra/model_server_client → model-server:8001/predict
    ├── tools/rag_tool        → services/rag → rag/retrieval (pgvector + BM25)
    ├── tools/ner_tool        → ml/ner (spaCy + regex)
    ├── tools/summarize_tool  → ml/summarizer (LLM-driven via Gemini)
    └── tools/write_memory    → services/memory → repositories/memory_repo → pgvector
    │
    ▼
infra/llm/gemini.py          (Gemini REST API, tool_call())
    │
    ▼
services/memory.py           (post-turn: summarize + embed conversation, store episodic)
    │
    ▼
HTTP 200 {response, conversation_id, tool_calls_made}
```

## Memory architecture

```
Short-term (Redis):
  Key:   conv:{conversation_id}:history
  Value: JSON array of {role, content, tool_calls?}
  TTL:   24h (covers a full workday; after expiry chatbot asks for context)

Long-term (pgvector — episodic):
  Table: memories (user_id, summary, embedding[768], created_at)
  Write: only via write_memory LLM tool (explicit) — never auto-written
  Read:  top-3 semantic similarity on new conversation start → injected in system prompt
  Audit: every write produces an audit_log row
```

## RAG pipeline

```
Ingest time:
  MONAI docs (markdown) + resolved issues
    → MarkdownChunker (recursive, respects headers)
    → Ollama nomic-embed-text (768-dim)
    → pgvector (rag_chunks table, tsvector column for BM25)

Query time:
  user query
    → query rewrite (multi-query template + Gemini Flash fallback)
    → dense retrieval (pgvector cosine similarity)
    + sparse retrieval (PostgreSQL tsvector + ts_rank_cd)
    → score fusion (0.6 dense + 0.4 sparse, top-20 candidates)
    → cross-encoder rerank — calls model-server:8001/rerank (BAAI/bge-reranker-base)
    → context assembly → LLM
```

The reranker runs in the **model-server** container (not the API container) because:
1. model-server already has torch loaded for the DistilBERT classifier — adding the reranker there is free.
2. Keeping torch + sentence-transformers out of the API container dropped its image from 1.56 GB → 727 MB.
3. If model-server returns 503 (model failed to load) or is unreachable, the retriever falls back to the dense+sparse hybrid score — graceful degrade, not failure.

## Security posture

- **Secrets**: all in Vault (KV-v2). API reads at boot, refuses if unreachable.
- **Redaction**: `redact_text()` runs on every log line, trace span attribute, and memory write.
- **Auth**: fastapi-users + JWT (HS256, key from Vault).
- **CSP**: `/embed` sets `frame-ancestors <allowed_origins>` per widget config.
- **CORS**: `allowed_origins` list from DB, not env var.
- **Audit log**: every `write_memory` produces a row with redacted content hash.

## Container topology

```
┌───────────────────────────────────────────────────────────┐
│  Docker bridge network: backend                           │
│                                                           │
│  db (pgvector) ◄─── migrate ─── api ──► model-server    │
│  redis ──────────────────► api                           │
│  minio ──────────────────► api                           │
│  vault ──────────────────► api, model-server             │
│  ollama ─────────────────► api (embeddings)              │
│  langfuse ───────────────► api (traces)                  │
│                                                           │
│  chatbot (Streamlit) ──► api                             │
│  widget (nginx) ────────► api                            │
│  host-allowed (nginx) ──► widget                         │
│  host-disallowed (nginx) ► widget (blocked by CSP)       │
└───────────────────────────────────────────────────────────┘
```
