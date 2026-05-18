# Maintainer's Copilot — Week 7 Build Plan

## Context

**Why this plan exists.** The user is shipping the AIE Week 7 solo project: a "Maintainer's Copilot" — an authenticated chatbot for open-source maintainers that classifies issues (bug/feature/docs/question) via a fine-tuned model, retrieves answers from project docs via advanced RAG, calls tools (classifier, NER, summarizer, RAG), carries conversation memory (Redis + pgvector), and is embeddable as a React widget in any host app. The architecture is graded; the evals are graded; layers, secrets, traces, and redaction are graded.

**Deadline.** End of week (Friday) with a 10-minute demo. 5 days. Solo. No scope creep.

**Working directory.** `/home/user/workplace/aie_sef_bootcamp/project7/` — currently only contains `docs/resources/`. Starting from scratch.

**Locked decisions from clarifying round:**
- **Dataset repo:** **`Project-MONAI/MONAI`** — PyTorch framework for medical imaging. Chosen over `commaai/openpilot` because openpilot has **0 closed `questions`** labels (fatal for a 4-class problem), and MONAI's docs are crawlable + structured for honest RAG eval. The chosen-over-rejected reasoning is documented in `DECISIONS.md`.
- **Classes — 3, not 4 (data-driven merge):** MONAI's label counts on closed issues are `bug=337`, `feature_request=535`, `documentation=28`, `questions=250`. A `documentation` class with 28 examples gives a ~5-example test set after a 70/15/15 stratified split — per-class F1 on 5 examples is statistical noise (one error ≈ 20-point swing) and cannot be defended. Solution: **merge `documentation` + `questions` → a single `support` class** (the maintainer's routing decision is identical for both — "point user to the right doc"). Final 3-class problem: `bug` (337), `feature` (535), `support` (278). Balanced, defensible, every per-class F1 is on a real test set. Document the merge in `DECISIONS.md` with the exact counts. The brief lists 4 classes as the target; the brief *also* says every choice in `DECISIONS.md` must be backed by a number — the merge is exactly that.
- **LLM:** Google **Gemini** primary (`gemini-2.5-pro` for chat, `gemini-2.5-flash` for LLM-baseline classifier), **Ollama** fallback (`llama3.1:8b-instruct` — tool-calling capable). One `LLMClient` abstraction so the fallback is real, not theatrical.
- **Fine-tuned classifier:** `distilbert-base-uncased`, 4-class head. Decide final model on Day 2 — DistilBERT placeholder lets the entire pipeline land Monday.
- **Widget:** full Vite + React + bundle + loader + demo host; styling polish deferred to Friday morning.

**Adopted tech stack (fills the brief's open slots):**
- Vector store: **pgvector** (already in Postgres; no second piece of infra).
- Tracing backend: **Langfuse** (self-hostable, OSS, native LLM-trace model — span attrs per the brief). Defended in `DECISIONS.md`.
- Long-term memory type: **episodic** (per-user conversation summaries) — easiest to demo cross-conversation recall.
- Eval framework: **RAGAS** for RAG (faithfulness, answer relevancy, context precision/recall), custom scikit-learn for classification (macro-F1, per-class F1, confusion matrix), 5 hand-labeled judge-agreement on RAG.
- Auth: `fastapi-users` + JWT (per brief). Signing key resolves from Vault at startup.
- Secrets: **HashiCorp Vault** (dev mode in compose). `.env` carries only Vault root token + ports.
- Blob: **MinIO** for model artifacts, `eval_report.json`, training plots, retrieved-chunks snapshots.
- Migrations: **Alembic**, dedicated `migrate` container runs `alembic upgrade head` and exits before `api` boots.
- Python: **3.12**. Package manager: **uv** (Astral) — `pyproject.toml` is source of truth, `uv.lock` is the committed lockfile (satisfies the bootcamp "use a lockfile" rule, better than `requirements.txt` because it's deterministic cross-platform). Linter/formatter: **ruff** (line-length 100; matches companion guide). Type checker: **mypy --strict**. Tests: **pytest + pytest-asyncio**. HTTP: **httpx**. Async ORM: **SQLAlchemy 2.x async** + **asyncpg**. Settings: **pydantic-settings** with `extra="forbid"`.

---

## Architecture (layered, same standard as Week 6)

**Top-level layout — three clear roots:**

```
project7/
├── backend/                          # FastAPI service + model-server inference + alembic + eval + scripts
├── frontend/                         # All user-facing surfaces: React widget, Streamlit UI, demo host
├── docker/                           # All container infra (Dockerfiles + compose)
├── docs/                             # ARCH/DECISIONS/RUNBOOK/EVALS/SECURITY + resources/
├── tests/e2e/                        # Cross-cutting browser tests (Playwright)
├── .github/workflows/
├── .env.example                      # ONLY vault root token + ports
├── .gitignore
├── .dockerignore
├── .pre-commit-config.yaml
├── README.md
└── CLAUDE.md                         # see "CLAUDE.md content" section at the end of this plan
```

**`backend/` — Python services (layered, same standard as Week 6):**

```
backend/
├── app/                              # FastAPI service — layered, strict boundaries
│   ├── api/                          # Routers ONLY. No SQLAlchemy, no Redis, no external calls.
│   │   ├── routes/
│   │   │   ├── auth.py               # /auth/* (fastapi-users)
│   │   │   ├── chat.py               # /chat — single tool-calling LLM entrypoint
│   │   │   ├── classify.py           # /classify
│   │   │   ├── ner.py                # /ner
│   │   │   ├── summarize.py          # /summarize
│   │   │   ├── rag.py                # /rag/search
│   │   │   ├── widgets.py            # /widgets (admin) + /widget/:id/config (public)
│   │   │   ├── embed.py              # /embed (sets CSP frame-ancestors from widget.allowed_origins)
│   │   │   └── memory.py             # /memory/* (inspector)
│   │   ├── dependencies.py           # Depends() — current_user, db_session, llm, vault, redaction, traces
│   │   ├── exceptions.py             # boundary handler (domain → HTTP, hides stack traces)
│   │   └── main.py                   # FastAPI app, lifespan, mount routers, refuse-to-boot policy
│   ├── services/                     # business logic, tx boundaries, cache/memory invalidation
│   │   ├── chatbot.py                # tool-calling loop, prompt assembly, memory read/write
│   │   ├── classification.py         # picks classifier model, runs predict, records eval
│   │   ├── rag.py                    # query rewrite → hybrid retrieval → rerank → context build
│   │   ├── memory.py                 # short-term (Redis TTL) + long-term (pgvector) read/write
│   │   ├── widgets.py                # CRUD, audit log row on every change
│   │   └── eval.py                   # runs golden sets, writes eval_report.json to MinIO
│   ├── repositories/                 # SQL only. No HTTP errors. No cache invalidation.
│   ├── domain/                       # Pydantic domain models (distinct from ORM)
│   │   ├── issue.py
│   │   ├── conversation.py
│   │   ├── widget.py
│   │   ├── memory.py
│   │   └── errors.py                 # NotFoundError, PermissionDenied, ToolFailure, ValidationError
│   ├── infra/                        # adapters: Vault, MinIO, Redis, LLM, tracing, REDACTION
│   │   ├── vault.py                  # boot-time secret resolve; refuse-to-boot if unreachable
│   │   ├── minio.py
│   │   ├── redis_client.py
│   │   ├── llm/
│   │   │   ├── base.py               # LLMClient protocol (chat, tool_call, embed)
│   │   │   ├── gemini.py             # primary
│   │   │   └── ollama.py             # fallback
│   │   ├── tracing.py                # Langfuse client; trace_id correlation; span helpers
│   │   ├── redaction.py              # patterns, redact_text(), used by logs/traces/memory writes
│   │   └── model_server_client.py    # HTTP client to model-server container
│   ├── ml/                           # fine-tuning + ML model loading
│   │   ├── train_classifier.py       # HF Trainer, run-logger, model card writer
│   │   ├── classifier.py             # singleton loader (lifespan), SHA-256 verification
│   │   ├── classical_ml.py           # TF-IDF + LogReg baseline
│   │   ├── llm_baseline.py           # Gemini few-shot baseline
│   │   ├── ner.py                    # spaCy + regex code-entity NER
│   │   └── summarizer.py             # pretrained (BART) or LLM-driven
│   ├── rag/                          # corpus build, chunking, embedding, retrieval
│   │   ├── ingest.py                 # docs + resolved issues → chunks → embeddings → pgvector
│   │   ├── chunking.py               # markdown-aware (NOT naive fixed-size)
│   │   ├── embeddings.py             # local sentence-transformers (BAAI/bge-small-en-v1.5)
│   │   ├── retrieval.py              # hybrid dense + BM25 + reranker
│   │   ├── rewrite.py                # query transformation (HyDE or multi-query)
│   │   └── metadata.py               # metadata filtering helpers
│   ├── tools/                        # LLM-callable tools that wrap services
│   │   ├── classify_tool.py
│   │   ├── ner_tool.py
│   │   ├── summarize_tool.py
│   │   ├── rag_tool.py
│   │   └── write_memory_tool.py      # explicit memory writes; NO auto-writes
│   └── config.py                     # one Settings class, extra="forbid", typed
├── model_server/                     # Separate FastAPI inference service (classifier + NER + summarizer)
│   ├── main.py
│   └── inference.py
├── alembic/                          # migrations (head includes pgvector extension + all tables)
│   ├── versions/
│   ├── env.py
│   └── alembic.ini
├── prompts/                          # version-controlled prompt files
│   ├── system_chatbot.md
│   ├── rag_query_rewrite.md
│   └── summarize.md
├── scripts/                          # one-shot CLIs
│   ├── fetch_issues.py
│   ├── build_splits.py
│   └── train_classifier.py           # entrypoint that calls app.ml.train_classifier
├── eval/                             # eval sets + runners + thresholds
│   ├── golden_classification.jsonl   # 25 hand-curated, NOT in training
│   ├── golden_rag.jsonl              # 25 question/ideal-answer/ground-truth-chunks
│   ├── run_classification_eval.py
│   ├── run_rag_eval.py
│   └── eval_thresholds.yaml          # committed thresholds; zero → refuse to boot
├── tests/                            # backend tests (mirrors app/ layout)
│   ├── unit/
│   ├── integration/                  # real Postgres + Redis via testcontainers
│   └── eval/                         # golden-set smoke tests, tagged @pytest.mark.eval
├── pyproject.toml                    # [project] deps + [dependency-groups.dev] + ruff/mypy/pytest config
├── uv.lock                           # committed lockfile (deterministic, the dependency rule's "lockfile")
└── .python-version                   # 3.12 (uv reads this)
```

**`frontend/` — every user-facing surface (both call the backend over HTTP, no shared imports):**

```
frontend/
├── widget/                           # React widget (Vite + TypeScript) — embeddable production surface
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts                    # talks to backend /chat
│   │   ├── Chat.tsx
│   │   └── types.ts
│   ├── public/
│   │   └── widget.js                 # loader script template (served as /widget.js)
│   ├── tests/                        # widget unit tests (Vitest)
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts                # outputs dist/widget.js — single bundle
├── streamlit_ui/                     # Streamlit UI — internal tool (auth, admin widget config, memory inspector, full chat)
│   ├── pages/
│   │   ├── 1_chat.py
│   │   ├── 2_admin_widgets.py
│   │   └── 3_memory_inspector.py
│   ├── lib/
│   │   ├── api_client.py             # httpx wrapper around backend (NO direct backend imports)
│   │   └── auth.py                   # JWT token storage in session_state
│   ├── tests/                        # streamlit unit tests
│   ├── app.py
│   ├── pyproject.toml                # [project] streamlit/httpx/pydantic + ruff/mypy (NO backend/ dep)
│   ├── uv.lock                       # committed lockfile
│   └── .python-version               # 3.12
└── demo_host/                        # nginx static demo host (Friday demo)
    ├── allowed/
    │   └── index.html                # pastes <script src="/widget.js" data-widget-id="..."></script>
    └── disallowed/
        └── index.html                # second host to prove CSP-block at the boundary
```

> **Architectural rule:** `frontend/streamlit_ui/` is a pure HTTP client of the backend, exactly like `frontend/widget/`. It must NOT `from app...` import anything from `backend/`. This keeps the boundary clean and means Streamlit can be built/run/deployed independently.

**`docker/` — all container infra in one place:**

```
docker/
├── docker-compose.yml                # api, chatbot, widget, model-server, host, migrate, db, redis, minio, vault
├── docker-compose.override.yml       # local-dev overrides (bind mounts, debug ports)
├── api.Dockerfile                    # builds backend/app
├── model_server.Dockerfile           # builds backend/model_server
├── streamlit.Dockerfile              # builds frontend/streamlit_ui
├── widget.Dockerfile                 # builds frontend/widget (node → nginx serves dist/)
├── host.Dockerfile                   # nginx for frontend/demo_host
└── nginx/
    ├── widget.conf
    └── host.conf
```

> Compose paths use `context: ../backend` for api/model-server, `context: ../frontend/streamlit_ui` for the Streamlit service, `context: ../frontend/widget` for the widget build, `context: ../frontend/demo_host` for the host. Each Dockerfile builds from the right root.

**`tests/e2e/`** — Playwright tests that exercise widget-in-host in a real browser (cross-cuts both `frontend/` and `backend/`).

**Layer rule (Friday TA check: "add an endpoint live"):** new endpoint = new file in `backend/app/api/routes/` + Pydantic in `backend/app/domain/` + service method + repository method (if SQL). Zero changes elsewhere.

---

## Day-by-Day Plan (with tests at each step)

### MON — Foundations

**Goal:** Repo skeleton, compose with all services up, Vault wired, Alembic baseline, dataset fetched + split, fine-tune started.

**Branch:** `feature/foundations-skeleton`

**Steps:**
1. **Repo scaffold** — directory tree above; `backend/pyproject.toml` + `frontend/streamlit_ui/pyproject.toml` (each with `[project]` deps, `[dependency-groups.dev]`, `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`); `uv lock` in each → commit `uv.lock`; `.gitignore` (`.env`, `.venv/`, `__pycache__`, `.coverage`, `dist/`, `node_modules/`, MinIO data); `.dockerignore`; `.pre-commit-config.yaml` running `uv run ruff` / `uv run mypy` / `gitleaks` / `pytest -q -m "not eval"`.
2. **docker-compose.yml** — services: `api`, `chatbot`, `widget`, `model-server`, `host`, `migrate`, `db` (postgres:16 + pgvector), `redis:7`, `minio/minio`, `hashicorp/vault` (dev mode). `migrate` runs `alembic upgrade head` and exits before `api` boots. `.env.example` has ONLY `VAULT_ROOT_TOKEN`, ports, and dev-mode flags — `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` must return zero outside Vault-reading code.
3. **Vault adapter** — `backend/app/infra/vault.py` resolves: `GEMINI_API_KEY`, `OLLAMA_HOST`, `JWT_SIGNING_KEY`, `DATABASE_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` at startup. App refuses to boot on `vault.Unreachable`.
4. **Settings class** — `backend/app/config.py`, `pydantic-settings`, `extra="forbid"`, `@lru_cache(maxsize=1) def get_settings()`. No `os.getenv()` anywhere else in the codebase.
5. **FastAPI lifespan** — `backend/app/api/main.py` builds singletons (db engine, redis pool, embedding model, classifier, llm client, vault client, langfuse) on startup; disposes on shutdown. Refuse-to-boot policy: missing classifier weights, missing SHA-256 match, missing tracing config, any `eval_thresholds.yaml` value at zero → exit non-zero.
6. **Alembic baseline** — initial migration: `users`, `widgets`, `conversations`, `messages`, `memories` (pgvector(384)), `audit_log`. `pgvector` extension enabled in migration.
7. **Domain exceptions** — `backend/app/domain/errors.py`: `NotFoundError`, `PermissionDenied`, `ToolFailure`, `ValidationError`. `backend/app/api/exceptions.py`: single `add_exception_handlers(app)` mapping domain → HTTP with `{code, request_id, message}`. No stack traces to user.
8. **Redaction layer** — `backend/app/infra/redaction.py`: regex patterns for OpenAI/Anthropic/Gemini/GitHub PATs, JWT-shaped strings, emails, AWS/GCP keys. Patterns documented in `SECURITY.md`. Used by logger formatter, Langfuse span attribute hook, and the `write_memory_tool` before any write.
9. **Dataset fetch + label confirmation** — `scripts/fetch_issues.py` pulls closed issues from `Project-MONAI/MONAI` via GitHub API (paginated, cached to MinIO). `scripts/audit_labels.py` confirms the counts we already have: `bug≈337`, `feature_request≈535`, `documentation≈28`, `questions≈250`. **Locked mapping:** `bug → bug`, `feature_request → feature`, `documentation + questions → support`. Documented in `DECISIONS.md` with counts and rationale (28-example `documentation` makes 4-class F1 unscientific; routing decision is identical for docs and Q&A; merge yields a balanced 3-class problem). `scripts/build_splits.py`: stratified, time-aware split (test strictly more recent than train).
10. **Start fine-tune** — `backend/app/ml/train_classifier.py`: HF Trainer on DistilBERT, **3-class** head (`bug`, `feature`, `support`), run-logger writes to W&B (or local JSONL fallback), saves model + tokenizer + `model_card.json` (architecture, hyperparams, training data SHA-256, class mapping, final metrics, freeze policy). Artifacts uploaded to MinIO.

**Tests after MON:**
- `tests/unit/test_redaction.py` — a string containing a fake `sk-abc123...` Gemini key is redacted in: logger output, Langfuse span attrs, memory writer. **Must fail CI if any of the three leaks.** (This is the brief's explicit redaction test.)
- `tests/unit/test_settings.py` — `Settings(extra="forbid")` raises on unknown key; required fields raise `ValidationError` when missing.
- `tests/integration/test_compose_boot.py` — `docker-compose up`, assert all services healthy within 60s.
- `tests/integration/test_vault_refusal.py` — kill Vault, restart `api`, assert it exits non-zero with structured error.
- `tests/integration/test_migrations.py` — fresh DB → `alembic upgrade head` succeeds, schema matches Pydantic domain models.
- `tests/unit/test_label_mapping.py` — known maintainer labels map to the four canonical classes.
- **Manual gate:** `docker-compose up` from a fresh clone after `cp .env.example .env` and pasting Vault token — every service comes up green.

---

### TUE — DL Track (classifier, baselines, NER, summarizer)

**Goal:** Three classifiers compared on same test split. NER + summarizer as HTTP endpoints. Classification golden set hand-curated.

**Branch:** `feature/dl-track-classifiers`

**Steps:**
1. **Finish fine-tune** — pick best epoch on val-macro-F1, freeze, push artifact + model card to MinIO. Classifier loader verifies SHA-256 on boot.
2. **Classical baseline** — `backend/app/ml/classical_ml.py`: TF-IDF + LogisticRegression, trained on same train split, evaluated on same test split. Pickled to MinIO.
3. **LLM baseline** — `backend/app/ml/llm_baseline.py`: Gemini few-shot prompt over the same test split. Records cost + latency per call.
4. **Three-way comparison** — `eval/run_classification_eval.py` runs all three on test + golden set. Writes `eval_report.json` (accuracy, macro-F1, per-class F1, confusion matrix, p50/p95 latency, $/1000 predictions). Uploaded to MinIO every run.
5. **Classification golden set** — `eval/golden_classification.jsonl`: 25 hand-curated issues, separate from test split, edge cases prioritized (multi-label, ambiguous, non-English snippets, very short, very long).
6. **`DECISIONS.md` deployment choice** — pick one of three. Backed by numbers (F1, latency, cost). Defend in one line per dimension.
7. **Model-server** — `model-server` container loads winning classifier (FastAPI inference server), exposes `/predict`, `/healthz`. Loaded once via lifespan singleton.
8. **`/classify` endpoint** — `backend/app/api/routes/classify.py` → service → `model_server.classify(text)`. Pydantic in, Pydantic out (`ClassificationResult(label, confidence, model_version, latency_ms)`).
9. **NER endpoint** — `backend/app/ml/ner.py`: extract code-shaped entities (FunctionName, ClassName, FilePath, ErrorType, PackageName) via spaCy `en_core_web_sm` + regex post-pass. Integration only — wrapped behind `/ner`.
10. **Summarizer endpoint** — `backend/app/ml/summarizer.py`: choose LLM-driven (Gemini Flash) or pretrained (BART). Defended in DECISIONS.md. Behind `/summarize`.
11. **Eval CI gate** — `eval_thresholds.yaml`: `classification.macro_f1 >= 0.78` (placeholder, adjust after first run). `eval/run_classification_eval.py` exits non-zero if below.

**Tests after TUE:**
- `tests/unit/test_classifier_loads.py` — SHA-256 mismatch raises; absent file raises.
- `tests/unit/test_classification_pydantic.py` — `ClassificationResult` rejects invalid labels, negative confidence, etc.
- `tests/integration/test_classify_endpoint.py` — POST `/classify` with mocked model-server returns 200 + valid schema; unhappy paths return 4xx with `{code, request_id, message}` and no stack traces.
- `tests/integration/test_ner_extracts_code_entities.py` — fixed text → known entities.
- `tests/eval/test_classification_thresholds.py` — runs full eval and asserts thresholds met (slow; tagged `@pytest.mark.eval`, only on CI).
- `tests/unit/test_three_way_comparison_report.py` — `eval_report.json` schema valid + persisted to MinIO.

---

### WED — Advanced RAG + redaction + exceptions

**Goal:** Corpus built, hybrid retrieval with reranker working, RAG golden set hand-curated, RAG eval gating CI, exception handling refactored at boundary.

**Branch:** `feature/advanced-rag`

**Steps:**
1. **Corpus build** — `backend/app/rag/ingest.py`: project docs (FastAPI docs MD files) + resolved-issues slice (NOT in classifier training, NOT in held-out classification set). Stored as raw blobs in MinIO.
2. **Chunking** — `backend/app/rag/chunking.py`: recursive markdown-aware (not naive fixed-size). Defended in DECISIONS.md with chunk-size sweep on golden set.
3. **Embeddings** — local `BAAI/bge-small-en-v1.5` (384-dim, fast, fits CPU). Compared against `all-MiniLM-L6-v2` on the golden set; loser noted in DECISIONS.md.
4. **Hybrid retrieval** — `backend/app/rag/retrieval.py`: pgvector dense + BM25 (via Postgres `tsvector` + `ts_rank_cd` OR `pg_trgm`). Tuned weight (e.g., 0.6 dense + 0.4 sparse) backed by Hit@5 on golden set.
5. **Reranker** — `BAAI/bge-reranker-base` cross-encoder over hybrid top-20 → top-5. Loaded once via lifespan.
6. **Query rewrite** — `backend/app/rag/rewrite.py`: HyDE OR multi-query expansion (pick one, defend). Gemini Flash for the rewrite call.
7. **Metadata filtering** — every chunk has `{repo, source_type: docs|issue, created_at, labels[]}`. Retrieval supports filters.
8. **RAG golden set** — `eval/golden_rag.jsonl`: 25 question / ideal-answer / ground-truth-chunks triples. Hand-curated.
9. **RAG eval** — `eval/run_rag_eval.py` with RAGAS: faithfulness, answer relevancy, context precision, context recall. Plus retrieval metrics: Hit@5, MRR@10. **Hand-label 5 of 25 yourself; report agreement % with RAGAS judge in `EVALS.md`.**
10. **RAG CI gate** — `eval_thresholds.yaml`: `rag.faithfulness >= 0.85`, `rag.answer_relevancy >= 0.80`, `rag.hit_at_5 >= 0.70` (placeholders; tune after first run).
11. **Exception-handling refactor** — single boundary handler in `backend/app/api/exceptions.py`. Every route returns `{code, request_id, message}` on failure. Tool failures inside chatbot are caught and recovered (return structured `ToolError` to LLM, not 500). Every uncaught exception logged with trace_id + request_id.
12. **Redaction integration** — every log line, every Langfuse span attribute, every memory write goes through `redact_text()`. Add the redaction test from MON to CI required-checks.

**Tests after WED:**
- `tests/unit/test_chunking_not_naive.py` — chunking respects markdown structure.
- `tests/integration/test_hybrid_retrieval.py` — known query → known chunk in top-5.
- `tests/integration/test_rerank_improves_top1.py` — rerank moves a known-good chunk above a known-bad one.
- `tests/eval/test_rag_thresholds.py` — full golden-set eval; tagged slow.
- `tests/integration/test_classifier_down_chatbot_degrades.py` — kill model-server, hit `/chat`, assert chatbot answers "classification unavailable" and **does not 500**.
- `tests/unit/test_exception_handler_hides_stack.py` — uncaught exception → response body contains `code` + `request_id`, NOT a stack trace.
- `tests/unit/test_redaction_in_traces.py` — a Langfuse span carrying a fake API key is exported with the key redacted.

---

### THU — Chatbot + memory + auth + widget + embed

**Goal:** Auth working, single tool-calling LLM with all four tools, short-term + long-term memory, React widget bundled and embedded in demo host. Both eval suites green in CI.

**Branch:** `feature/chatbot-and-widget`

**Steps:**
1. **Auth** — `fastapi-users` with JWT, email + password registration. JWT signing key from Vault. Two roles: `user`, `admin`. Admin endpoints (widget config, user invite) gated by role dependency.
2. **Tool-calling LLM** — `backend/app/services/chatbot.py`: single Gemini call with `tools=[classify, ner, summarize, rag_search, write_memory]`. LLM decides; no hand-coded workflow, no multi-agent. Tools are real wrappers over the services built TUE/WED.
3. **Prompts as files** — `prompts/system_chatbot.md`, `prompts/rag_query_rewrite.md`. Loaded at startup, hot-reload disabled in prod.
4. **Short-term memory** — Redis. Key: `conv:{conversation_id}:history`. TTL: 24h (defended in `DECISIONS.md`: 24h covers a workday, expires before stale; at boundary, conversation continues but history is empty and chatbot asks for context). Stored as JSON array of `{role, content, tool_calls?}`.
5. **Long-term memory (episodic)** — at conversation end (or every N turns), service summarizes conversation, embeds it (same model as RAG), stores in `memories` table (`user_id, summary, embedding, created_at`). On new conversation, top-3 semantic-similarity memories are retrieved and injected into the system prompt.
6. **`write_memory` tool** — explicit tool the LLM must call to write memory. **No auto-writes.** Every call produces an `audit_log` row: `(actor=user_id, action='memory.write', target=memory_id, timestamp, redacted_content_hash)`.
7. **Tracing** — every LLM call, tool call, RAG retrieval is a Langfuse span. Conversation = trace tree rooted at the user message. Span attributes: `model`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `tool_name`, `tool_inputs_redacted`, `tool_outputs_redacted`. Trace ID logged with every structured log line for the same request.
8. **Streamlit UI** — `frontend/streamlit_ui/app.py`: login page (calls `/auth/jwt/login`), chat page (calls `/chat`), admin widget config page (admin only), memory inspector page (shows your stored memories, lets you delete with audit log). All backend calls go through `frontend/streamlit_ui/lib/api_client.py` (httpx) — no direct imports from `backend/`.
9. **Widget table + CRUD** — `widgets(id, public_widget_id, owner_id, allowed_origins[], theme jsonb, greeting, enabled_tools[], created_at)`. Admin Streamlit page generates a public `widget_id` and shows the embed snippet `<script src="{API_HOST}/widget.js" data-widget-id="..."></script>`.
10. **React widget** — `widget/`: Vite + React + TypeScript. Single bundle output. Reads its config at load from `/widget/:id/config`. Chat panel, input box, streamed messages, collapsed bubble that expands. `postMessage` channel for iframe-resize. Theme + greeting + position from config (NOT hardcoded). **No styling polish today — functional only.**
11. **Loader script** — `widget/public/widget.js`: reads `data-widget-id` from its own `<script>` tag, fetches config, injects an iframe pointing at `/embed?widget_id=...`, sets sandbox attributes.
12. **`/embed` route** — serves the bundled widget HTML; sets `Content-Security-Policy: frame-ancestors <allowed_origins>` header matching the widget's DB record. CORS allowlist also from `allowed_origins`, **not** from a hardcoded env var.
13. **Demo host** — `demo_host/index.html`: nginx-served page with `<script src="http://localhost:8000/widget.js" data-widget-id="..."></script>`. Plus a second host page on a different port with NOT-allowed origin (Friday demo's CSP-block proof).
14. **Both eval suites in CI** — `.github/workflows/ci.yml` runs both eval scripts on push; thresholds in `eval_thresholds.yaml` gate merge. `eval_report.json` from each run pushed to MinIO and diffed against the previous green build.

**Tests after THU:**
- `tests/integration/test_auth_jwt.py` — register, login, hit protected route with token (200), without token (401), with wrong role (403).
- `tests/integration/test_chat_tool_calling.py` — POST `/chat` with a question that requires RAG → assert `rag_search` tool was called (via Langfuse fake client recording calls).
- `tests/integration/test_chat_classifier_down.py` — model-server killed → chat says "classification unavailable", returns 200.
- `tests/integration/test_short_term_memory_ttl.py` — conversation set, TTL fast-forwarded → next message gets fresh context.
- `tests/integration/test_long_term_memory_retrieval.py` — write memory, new conversation, top-K retrieves it; user-A memories never returned for user-B (multi-tenant isolation).
- `tests/unit/test_write_memory_audit_log.py` — `write_memory` tool always produces an audit row; LLM cannot bypass it.
- `tests/integration/test_widget_config_admin_only.py` — non-admin gets 403.
- `tests/integration/test_csp_frame_ancestors.py` — `/embed?widget_id=X` returns `Content-Security-Policy: frame-ancestors a.com b.com` for X's allowlist.
- `tests/integration/test_cors_from_db.py` — origin in allowlist → CORS OK; origin not in allowlist → CORS denied.
- `tests/e2e/test_widget_loads_in_host.py` — Playwright (or Selenium): navigate to demo host, assert widget iframe loads, send a message, assert response renders.
- `tests/e2e/test_widget_blocked_on_disallowed_host.py` — navigate to disallowed host, assert browser blocks the iframe (console error).
- `tests/unit/test_widget_bundle_size.py` — `wc -c frontend/widget/dist/widget.js` < threshold (e.g., 200KB gzipped); CI fails if exceeded.

---

### FRI AM — Polish, README/docs, demo prep

**Goal:** All CI green, all five required `.md` files complete, widget styled, rehearsed demo.

**Branch:** `chore/ship-prep`

**Steps:**
1. **Widget styling** — Tailwind or vanilla CSS, match config theme. Verify bundle still under threshold after CSS.
2. **README** — project description, full architecture diagram (ASCII or mermaid), setup (`cp .env.example .env`, paste Vault token, `docker-compose up`), env vars list, API doc link (`/docs` FastAPI auto-gen), pointer to ARCH/DECISIONS/RUNBOOK/EVALS/SECURITY.
3. **`ARCH.md`** — layered diagram, layer rules, data flow (chat → tool → service → repo → DB / chat → tool → RAG service → pgvector / chat → memory service → Redis & pgvector).
4. **`DECISIONS.md`** — every choice with a number: embedding model (Hit@5: bge vs MiniLM), chunking (sweep), retrieval weights, reranker on/off, classifier deployment, tracing backend, memory type, query rewrite technique. One line "because" per choice.
5. **`RUNBOOK.md`** — how to start/stop the stack, where logs live, how to read a trace in Langfuse, how to rerun an eval, what to do when Vault becomes unreachable mid-flight (graceful degradation: cached secrets keep current requests going; new secret resolves fail; periodic re-resolve loop logs warnings; admin paged via configured webhook).
6. **`EVALS.md`** — both golden sets explained, the five hand-labeled vs judge-agreement numbers, what passing/failing means, how thresholds were chosen.
7. **`SECURITY.md`** — redaction patterns + rationale, secret list (names only), CSP/CORS posture, audit log fields, refuse-to-boot policy, response-shape policy.
8. **CI final pass** — lint, typecheck, build images, both eval suites against golden sets, redaction test, full smoke-test the stack from a fresh clone. All green.
9. **Git tag** — `git tag v0.1.0-week7 && git push --tags`. Repo public.
10. **Rehearse** — 10-minute demo: widget loading in allowed host, widget blocked in disallowed host, trace tree (including an error path), cross-conversation memory recall, one live "TA picks an issue and the chatbot triages it" run.

**Tests after FRI AM (final acceptance):**
- `docker-compose down -v && docker-compose up --build` from a fresh clone — every service green.
- `pytest -q` — all unit + integration tests pass.
- `pytest -q -m eval` — both eval suites pass, thresholds met.
- `wc -c frontend/widget/dist/widget.js` ≤ threshold.
- `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` outside Vault-reading code → empty.
- Manual: load demo host in browser, widget appears, chat works, trace appears in Langfuse, memory survives a new conversation.

---

## Critical files (reused functions/utilities to wire repeatedly)

These are the spines — touched almost daily (all paths relative to repo root):

- `backend/app/config.py::get_settings()` — lru_cached singleton, all config goes through here.
- `backend/app/infra/vault.py::resolve_secret(name)` — boot-time resolution.
- `backend/app/infra/redaction.py::redact_text(s)` — used by logger, tracer, memory writer.
- `backend/app/api/dependencies.py::get_current_user`, `get_db_session`, `get_llm`, `get_tracer` — every route declares its needs.
- `backend/app/api/exceptions.py::add_exception_handlers(app)` — one boundary handler, all routes.
- `backend/app/domain/errors.py::AppError` hierarchy — every service raises domain errors, never `HTTPException` directly.
- `backend/app/infra/llm/base.py::LLMClient` protocol — Gemini and Ollama both implement; tests override via `app.dependency_overrides`.
- `backend/eval/run_*_eval.py` — used in CI and locally; same code path.

---

## CLAUDE.md content (to be written immediately after plan approval)

This is the full text I will write to `/home/user/workplace/aie_sef_bootcamp/project7/CLAUDE.md` when we exit plan mode. It encodes the standards from both PDFs as machine-checkable rules for any future Claude session in this repo:

```markdown
# CLAUDE.md — Maintainer's Copilot

You are working inside the AIE Week 7 "Maintainer's Copilot" project. This file is your contract. Read it before every non-trivial change. The standards below come from the AIE Bootcamp Coding Guidelines and the Engineering Standards Companion Guide — they are graded.

## What this project is

Authenticated chatbot for open-source maintainers. Fine-tuned classifier (DistilBERT) + advanced RAG (hybrid + rerank) + tool-calling Gemini (Ollama fallback) + short/long memory + React widget. Layered codebase. Secrets in Vault. Blob in MinIO. Traces in Langfuse. Logs redacted. Two eval gates in CI.

## Non-negotiable rules

### Architecture (the grade)
- **Top-level roots:** `backend/` (all Python), `frontend/` (all JS/TS), `docker/` (all container infra), `docs/`, `tests/e2e/`. Do NOT add a new top-level dir without updating ARCH.md.
- **Inside `backend/app/` — layers:** `api/` (routers only, no SQLAlchemy/Redis/external calls) → `services/` (business logic, tx boundaries, cache/memory invalidation) → `repositories/` (SQL only) ; `domain/` Pydantic distinct from ORM ; `infra/` for Vault/MinIO/Redis/LLM/tracing/redaction. A new endpoint = new route file + Pydantic + service method + repo method. Zero churn elsewhere.
- **Refuse to boot** if: Vault unreachable, classifier weights missing, classifier SHA-256 mismatches `model_card.json`, tracing misconfigured, any `eval_thresholds.yaml` value is zero/disabled.

### Secrets (the grade)
- Every secret resolves from Vault at startup. **No `os.getenv` outside `backend/app/config.py` and `backend/app/infra/vault.py`.**
- `.env` carries only the Vault root token and ports. Never anything else.
- `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` MUST return zero matches outside Vault-reading code. Run before every commit.

### Logging & redaction (the grade)
- `backend/app/infra/redaction.py::redact_text()` runs before any log line, trace span attribute, or memory write leaves the service boundary.
- Never use `print()` for operational output. `structlog` everywhere. JSON-structured.
- Every log line carries the `trace_id` and `request_id` for the request.
- Never log: passwords, tokens, full request/response bodies with PII, secrets, connection strings.
- A redaction test asserts a fake `sk-...` key never appears unredacted in logs/traces/memory. **It must stay green.**

### Tracing (the grade)
- Every LLM call, tool call, and RAG retrieval is a Langfuse span. Conversation = trace tree rooted at user message.
- Span attributes: model, token counts, latency, tool I/O **after redaction**.
- Trace ID + request ID are joinable across logs and Langfuse.

### Exceptions (the grade)
- Domain exceptions live in `backend/app/domain/errors.py`. They are NOT `HTTPException`. Services raise them; the API boundary handler maps them to HTTP.
- **Never** bare `except:`. **Never** silently swallow. Never expose stack traces to users. Users see `{code, request_id, message}`.
- Tool failures inside the chatbot are caught and returned to the LLM as structured `ToolError(error, retryable)`. The agent decides what to do. It does not 500.
- Layer-3 failure isolation: classifier down ⇒ chatbot says so and continues; it does NOT return 500.

### Async (the grade)
- Every route, tool, and external call is async. **No `requests`** anywhere; use `httpx`. **No `time.sleep`**; use `await asyncio.sleep`.
- CPU-bound calls (`model.predict`) go through `asyncio.to_thread()` if they cost > a few ms.
- `await asyncio.gather()` independent calls in parallel.

### Dependency Injection (the grade)
- Every dependency (DB session, LLM, current user, tracer) is declared with `Depends()`. No globals, no module-level singletons constructed at import time.
- Heavy singletons (model, embedder, engine) are built in `lifespan()` on `app.state`, exposed through dependencies, disposed on shutdown.
- Tests use `app.dependency_overrides`. No monkey-patching of imports.

### Caching
- `functools.lru_cache(maxsize=1)` for `get_settings()` and other deterministic helpers. Never for things that take dicts, expire, or depend on time.
- `cachetools.TTLCache` + `asyncio.Lock` for external-API responses that have a justified freshness window. Document the TTL.

### Configuration
- One `Settings(BaseSettings)` in `backend/app/config.py`. `extra="forbid"`. All required fields use `Field(...)`. Tests construct `Settings(...)` directly with overrides.
- Never `os.getenv` outside this file and Vault adapter.

### Types & boundaries
- Type hints required on every function signature (`mypy --strict` enforced).
- Pydantic at every boundary: HTTP request bodies, LLM tool-call inputs/outputs, LLM structured outputs, webhook payloads.
- Inside the system, trust your types. Don't sprinkle defensive isinstance checks.

### Errors, retries, failure isolation
- Every external call has a `timeout`. Default 10s.
- Use `tenacity` with `stop_after_attempt(3)`, `wait_exponential`, `retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))`, `reraise=True`. Do not retry 4xx.
- Tools return `ToolError` (not raise) so the agent loop continues.

### Code style
- Python 3.12. **Ruff** is the linter + formatter (line-length 100, `select = ["E","F","I","B","UP","ASYNC","S"]`, `ignore = ["E501"]`). **mypy --strict**.
- PEP 8: 4-space indent, double quotes, trailing commas in multi-line. Imports: stdlib / third-party / local, blank-line-separated, isort/ruff sorted.
- Naming: `snake_case` variables/functions/modules; `PascalCase` classes; `UPPER_SNAKE_CASE` constants; `_leading_underscore` private; descriptive (`calculate_total_price`, not `calc`); booleans as questions (`is_active`, `has_permission`); collections plural; functions start with a verb.

### Branches, commits, PRs
- Branches: `<type>/<short-description>` — `feature/`, `bugfix/`, `hotfix/`, `refactor/`, `docs/`, `test/`, `chore/`. Lowercase + hyphens.
- Commits: **Conventional Commits** — `<type>(<scope>): <imperative summary under 72 chars>`. No trailing period. Body optional.
- PR title: `[TYPE] Short imperative description`. PR body uses the template (Summary / Changes / Testing / Screenshots / Checklist).
- PRs < 400 lines. One concern per PR. Squash on merge.
- Never commit directly to `main`. Never `--no-verify`. Never `--force-push` to `main`.

### Testing
- File names: `test_<thing>.py`. Function names: `test_<what>_<expected>` (e.g., `test_login_with_invalid_password_returns_401`).
- AAA pattern. One assertion-of-meaning per test.
- **Always test:** Pydantic schemas (valid + invalid), tool logic with mocked LLM, every API endpoint's status codes and shapes, error paths, auth & authz, redaction, multi-tenant isolation.
- **Coverage:** ≥80% line coverage overall; ≥95% on critical paths (auth, payments-equivalent, data mutations, redaction).
- `pytest --cov=app --cov-report=term-missing`. Coverage is a floor, not a target — meaningful tests over hitting a number.
- Slow tests tagged `@pytest.mark.eval`; CI runs them; local runs skip by default.
- Use `app.dependency_overrides` to inject fakes. Do not monkey-patch imports.

### Dependencies — `uv` only
- Source of truth: `pyproject.toml`. Lockfile: `uv.lock` (committed). Runtime: `uv sync --frozen`.
- Add a dep: `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev). Never edit `pyproject.toml` deps by hand without re-running `uv lock`.
- Run commands inside the env: `uv run ruff check .`, `uv run mypy .`, `uv run pytest -q`. No system `pip install`.
- Production Docker images: `uv sync --frozen --no-dev`. Dev images: `uv sync --frozen`.
- Review every new dependency: maintained? CVEs? Does stdlib already do it?
- Run `uv pip audit` (or `pip-audit` via `uvx`) in CI.

### Documentation
- Public modules / classes / functions: Google-style docstrings with Args / Returns / Raises.
- Inline comments only when WHY is non-obvious. Don't restate WHAT.
- `README.md` is the front door: description, setup, env vars list, link to `/docs`, architecture pointer. Plus the five required: `ARCH.md`, `DECISIONS.md`, `RUNBOOK.md`, `EVALS.md`, `SECURITY.md`.

### What I should NOT do in this repo

- Do not commit to `main`. Always branch.
- Do not run destructive git commands (`reset --hard`, `push --force`, branch deletion) without confirming with the user.
- Do not skip pre-commit hooks (`--no-verify`).
- Do not introduce a new top-level module without asking.
- Do not add a dependency without justifying it in the commit message and updating `requirements.txt` pin.
- Do not write code that bypasses Vault for secrets, redaction for logs, or the boundary handler for errors.
- Do not auto-write memory. The LLM must call `write_memory` explicitly; every write must produce an audit row.
- Do not regress any threshold in `eval_thresholds.yaml`. If a threshold needs to change, change it in a dedicated PR with the eval-report diff.
- Do not ship code I cannot defend orally on Friday. "No vibe coding."

### Pre-commit pipeline (runs on every commit, do not skip)

```
uv run ruff format → uv run ruff check → uv run mypy --strict → uv run pytest -q -m "not eval" → gitleaks
```

### Submission

Tag `v0.1.0-week7`. Repo public. `docker-compose up` succeeds from a fresh clone after `cp .env.example .env` and pasting the Vault root token. Both eval suites pass in CI.

— ship it.
```

---

## Verification (end-to-end acceptance)

A complete build is verified by running, in this order:

1. **Fresh clone smoke:** `git clone <repo> && cd <repo> && cp .env.example .env && <paste vault token> && docker-compose up --build` — every service green within ~60s.
2. **Static checks:** `cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy --strict app/` — all clean. Same in `frontend/streamlit_ui/`. In `frontend/widget/`: `npm run lint && npm run typecheck`.
3. **Secret hygiene:** `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` — empty outside Vault-reading code; `gitleaks detect` — clean.
4. **Unit + integration tests:** `cd backend && uv run pytest -q` — green.
5. **Eval suites:** `cd backend && uv run pytest -q -m eval` — both classification + RAG above committed thresholds.
6. **Redaction proof:** dedicated test asserts a fake `sk-...` key never appears unredacted in logs/traces/memory.
7. **Refuse-to-boot:** kill Vault, restart `api`, assert non-zero exit + structured error.
8. **Trace tree demo:** open Langfuse, walk a real `/chat` conversation including a tool-failure path.
9. **Cross-conversation memory:** new login → previous conversation's stored memory surfaces in the system prompt.
10. **Widget on allowed host:** demo-host page in browser → widget loads, chat round-trips, message renders.
11. **Widget blocked on disallowed host:** second host page → browser console shows CSP frame-ancestors block; widget does not render.
12. **Bundle size:** `wc -c frontend/widget/dist/widget.js` ≤ committed threshold (e.g., 200 KB gzipped).
13. **Tag:** `v0.1.0-week7` exists on `main`; CI is green on the tagged commit.

Passing all 13 = ready to demo Friday.
