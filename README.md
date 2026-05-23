# Maintainer's Copilot

> Authenticated chatbot for open-source maintainers — classifies issues with a fine-tuned DistilBERT, retrieves docs with hybrid RAG + cross-encoder rerank, calls tools through Gemini, remembers across conversations, and embeds anywhere as a React widget.

---

## What this is

A graded, end-of-week AIE Bootcamp project. One repo, one `docker compose up`, all the moving parts you'd expect of a production-shaped LLM application:

- **Fine-tuned classifier** — DistilBERT (3-class: `bug` / `feature` / `support`) on `Project-MONAI/MONAI` issues.
- **Advanced RAG** — pgvector dense + Postgres BM25 hybrid → BAAI/bge-reranker-base cross-encoder rerank → markdown-aware chunks across MONAI wiki + resolved issues.
- **Tool-calling LLM** — Gemini 2.5 Flash (primary) with Ollama fallback. Five tools: `classify_issue`, `rag_search`, `extract_entities`, `summarize_text`, `write_memory`.
- **Memory** — short-term in Redis (24h TTL, per-conversation), long-term episodic in pgvector (per-user, semantic recall).
- **Auth & multi-tenant** — fastapi-users + JWT. Per-user conversation isolation; server enforces ownership.
- **Embeddable widget** — React + Vite. Single bundle. CSP `frame-ancestors` restricts embedding to whitelisted origins per widget.
- **Streamlit admin** — login, chat, widget CRUD (create / edit / delete with tool selection), memory inspector.
- **Observability** — Langfuse traces every LLM call, tool call, and retrieval; structlog JSON with redaction.
- **Two eval gates** — RAGAS (faithfulness, answer relevancy, Hit@5) on wiki + issue golden sets, classification macro-F1 with per-class breakdown. Thresholds gate CI.

The full layered architecture, decision log, security posture, and ops runbook live in [docs/](docs/).

---

## 30-second setup

```bash
git clone <repo> && cd project7
cp .env.example .env
# Edit .env: set VAULT_ROOT_TOKEN (any string for dev) and paste your GEMINI_API_KEY
docker compose -f docker/docker-compose.yml up --build
```

Every service is healthy in ~60s on a warm machine. Then open:

| Surface | URL |
|---|---|
| Streamlit admin | http://localhost:8501 |
| API (Swagger) | http://localhost:8000/docs |
| Allowed-host demo (widget loads) | http://localhost:8090 |
| Disallowed-host demo (CSP block) | http://localhost:8091 |
| Langfuse traces | http://localhost:3000 |
| Vault UI | http://localhost:8200 |
| MinIO console | http://localhost:9001 |

Sign in as **`user1@mail.com` / `12345678`** (seeded admin) or register a new user.

---

## What's where

```
project7/
├── backend/                # FastAPI service + model-server + alembic + eval + scripts
│   ├── app/{api,services,repositories,domain,infra,ml,rag,tools}/
│   ├── eval/{classification,rag}/   # golden sets + runners + thresholds
│   └── pyproject.toml      # uv-managed; lockfile committed
├── frontend/
│   ├── widget/             # React + Vite embeddable bundle
│   ├── streamlit_ui/       # Internal admin tool (HTTP client of backend)
│   └── demo_host/{allowed,disallowed}/   # nginx static hosts for CSP demo
├── docker/                 # Dockerfiles + compose
├── docs/
│   ├── ARCH.md             # layered diagram + data flow
│   ├── DECISIONS.md        # every design choice backed by a number
│   ├── RUNBOOK.md          # start/stop/debug/incident response
│   ├── EVALS.md            # golden sets + thresholds + judge agreement
│   ├── SECURITY.md         # redaction patterns, CSP/CORS, refuse-to-boot
│   ├── PLAN.md             # day-by-day build plan
│   └── reports/            # step completion reports
└── tests/e2e/              # Playwright browser tests
```

---

## How the pieces talk

```
                  +-------------------+
                  |  React widget     |        +--------------+
                  |  (host page CSP-  | <----- | demo_host    |
                  |   restricted)     |        +--------------+
                  +---------+---------+
                            |  POST /chat {message, widget_id, conv_id}
                            v
+-----------------+   +-----+-------+   +----------------+   +-----------+
| Streamlit admin |-->| FastAPI API |-->| Chatbot service|-->| Gemini    |
| (HTTP client)   |   | /chat /auth |   | (tool loop)    |   | (primary) |
+-----------------+   +-+--+--+--+--+   +--+-----+----+--+   +-----+-----+
                        |  |  |  |        |     |    |            |
                        |  |  |  |        |     |    +-> Ollama (fallback)
                        |  |  |  |        |     |
                        |  |  |  |        |     +-> RAGService -> pgvector + BM25 -> bge-reranker
                        |  |  |  |        +-> Memory (Redis + pgvector)
                        |  |  |  +-> Classifier (model-server, DistilBERT)
                        |  |  +----> Widget service (PUT/GET enabled_tools, CSP)
                        |  +-------> Auth (fastapi-users JWT)
                        +----------> Vault (boot-time secrets)

                  Every call is a Langfuse span; every log line is redacted.
```

See [docs/ARCH.md](docs/ARCH.md) for layer rules, data flow per tool, and the failure-isolation contract.

---

## Required reading before changing anything

1. **[CLAUDE.md](CLAUDE.md)** — the operating contract. Architecture, secrets, redaction, exceptions, async, DI, code style, branches/commits, testing — all non-negotiable.
2. **[docs/ARCH.md](docs/ARCH.md)** — strict layers inside `backend/app/` (`api/` → `services/` → `repositories/`). A new endpoint = a new route file + Pydantic + service method + repository method. Zero churn elsewhere.
3. **[docs/DECISIONS.md](docs/DECISIONS.md)** — why DistilBERT not Gemini for classification, why merge 4 classes → 3, why hybrid 0.6/0.4 weighting, why bge-small embeddings, etc.

---

## Day-by-day status

| Day | Goal | Status |
|---|---|---|
| **MON** | Repo + compose + Vault + Alembic + redaction + fetch MONAI issues + start fine-tune | ✅ |
| **TUE** | DistilBERT + classical + LLM baselines + NER + summarizer + classification golden set | ✅ |
| **WED** | Hybrid retrieval + cross-encoder rerank + RAG golden + RAGAS eval | ✅ |
| **THU** | Auth + tool-calling LLM + memory + Streamlit + React widget + demo host + CI evals | ✅ |
| **FRI AM** | Widget polish + all 5 `.md` files + tag `v0.1.0-week7` | ✅ |

---

## Submission acceptance checklist

The [13-item demo gate](CLAUDE.md#acceptance-checklist-run-before-friday-demo) is what we test against:

| # | Check |
|---|---|
| 1 | Fresh-clone `docker compose up` — every service green in ~60s |
| 2 | `ruff` + `mypy --strict` + widget `tsc --noEmit` all clean |
| 3 | `grep -ri 'sk-' backend/` empty outside Vault-reading code; gitleaks clean |
| 4 | `uv run pytest -q` — unit + integration green |
| 5 | `uv run pytest -q -m eval` — both eval suites above thresholds |
| 6 | Redaction test asserts fake `sk-...` never appears in logs/traces/memory |
| 7 | Kill Vault → API exits non-zero with structured `refuse_to_boot` error |
| 8 | Langfuse shows a full trace tree with a tool-failure path |
| 9 | New login surfaces previous conversation's stored memory in the system prompt |
| 10 | Demo host 8090: widget loads + chats |
| 11 | Demo host 8091: browser console shows CSP `frame-ancestors` block |
| 12 | `wc -c frontend/widget/dist/widget.js` ≤ 200 KB gzipped |
| 13 | `v0.1.0-week7` tag on `main`; CI green |

Full ops runbook → [docs/RUNBOOK.md](docs/RUNBOOK.md). Eval methodology → [docs/EVALS.md](docs/EVALS.md). Security posture → [docs/SECURITY.md](docs/SECURITY.md).

---

## License

Project for AIE Week 7 Bootcamp submission. Source code MIT unless otherwise noted in subdirectories.
