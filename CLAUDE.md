# CLAUDE.md — Maintainer's Copilot

You are working inside the AIE Week 7 "Maintainer's Copilot" project. This file is your contract. Read it before every non-trivial change. The standards below come from the **AIE Bootcamp Coding Guidelines** and the **Engineering Standards Companion Guide** (in [docs/resources/](docs/resources/)) — they are graded.

The full build plan lives at [docs/PLAN.md](docs/PLAN.md). This file is the operating manual.

---

## What this project is

Authenticated chatbot for open-source maintainers. Fine-tuned classifier (DistilBERT, 3-class: `bug` / `feature` / `support`) + advanced RAG (hybrid retrieval + cross-encoder rerank) + tool-calling Gemini with Ollama fallback + short-term (Redis) and long-term (pgvector episodic) memory + embeddable React widget + Streamlit admin UI. Layered codebase. Secrets in Vault. Blob in MinIO. Traces in Langfuse. Logs redacted. Two eval gates in CI.

**Dataset:** `Project-MONAI/MONAI` closed issues. Classes merged from 4 → 3 because `documentation` has only 28 examples (would yield a 5-example test set after stratified split — statistical noise). Routing for `documentation` and `questions` is identical, so they merge into `support`. Documented in `docs/DECISIONS.md` with counts.

**Stack:** Python 3.12 + uv + FastAPI + SQLAlchemy 2.x async + asyncpg + Pydantic + pydantic-settings + structlog + Langfuse + sentence-transformers (bge-small + bge-reranker) + HuggingFace (DistilBERT) + spaCy + RAGAS. React + Vite + TypeScript. Postgres 16 + pgvector. Redis 7. MinIO. HashiCorp Vault. Alembic. Docker Compose.

---

## Top-level layout

```
project7/
├── backend/                          # FastAPI service + model-server + alembic + eval + scripts
│   ├── app/                          # Layered: api/ → services/ → repositories/ ; domain/ ; infra/ ; ml/ ; rag/ ; tools/
│   ├── model_server/                 # Separate FastAPI inference service
│   ├── alembic/                      # Migrations (head includes pgvector extension)
│   ├── prompts/                      # Version-controlled prompt files
│   ├── scripts/                      # One-shot CLIs (fetch_issues, build_splits, train_classifier)
│   ├── eval/                         # Golden sets + runners + thresholds.yaml
│   ├── tests/{unit,integration,eval}/
│   ├── pyproject.toml                # [project] + [dependency-groups.dev] + ruff/mypy/pytest config
│   └── uv.lock                       # Committed lockfile (the dependency rule's "lockfile")
├── frontend/                         # All user-facing surfaces (each talks to backend over HTTP only)
│   ├── widget/                       # React widget (Vite + TypeScript) — production-shaped embeddable
│   ├── streamlit_ui/                 # Streamlit internal tool (auth, admin, memory inspector, chat)
│   │                                 # MUST NOT import from backend/ — pure HTTP client
│   └── demo_host/{allowed,disallowed}/  # nginx static demo hosts (Friday CSP-block proof)
├── docker/                           # All container infra (Dockerfiles + compose)
├── docs/
│   ├── PLAN.md                       # full build plan
│   ├── ARCH.md, DECISIONS.md, RUNBOOK.md, EVALS.md, SECURITY.md
│   └── resources/                    # source PDFs (brief, coding guide, eng standards)
├── tests/e2e/                        # Playwright cross-cutting browser tests
├── .github/workflows/                # ci.yml (lint, typecheck, both eval suites, redaction, smoke)
├── .env.example                      # ONLY vault root token + ports
└── CLAUDE.md                         # this file
```

**Layer rule (Friday TA check: "add an endpoint live"):** new endpoint = new route file in `backend/app/api/routes/` + Pydantic in `backend/app/domain/` + service method + repository method (if SQL). **Zero churn elsewhere.**

---

## Day-by-day plan (summary; full version in `docs/PLAN.md`)

| Day | Branch | Goal | Test gate at end of day |
|---|---|---|---|
| **MON** | `feature/foundations-skeleton` | Repo + compose + Vault + Alembic + redaction + fetch MONAI issues + start fine-tune | redaction test, vault refusal, fresh-clone `docker-compose up`, migrations |
| **TUE** | `feature/dl-track-classifiers` | Finish fine-tune, classical + LLM baselines, three-way comparison, NER, summarizer, classification golden set | per-class F1 + macro-F1 above committed threshold; `/classify` endpoint integration test |
| **WED** | `feature/advanced-rag` | Corpus build, markdown chunking, hybrid retrieval, reranker, query rewrite, RAG golden, exception refactor | RAG thresholds (faithfulness, answer-relevancy, Hit@5); classifier-down chatbot stays 200 |
| **THU** | `feature/chatbot-and-widget` | Auth + tool-calling LLM + memory + Streamlit + React widget bundled + demo host + both eval suites in CI | e2e widget loads on allowed host, blocked on disallowed; cross-conv memory recall; bundle size |
| **FRI AM** | `chore/ship-prep` | Widget styling, all 5 `.md` files, CI green, `v0.1.0-week7` tag | full acceptance checklist (13 items in PLAN.md) |

---

## Non-negotiable rules

### Architecture (the grade)
- **Top-level roots:** `backend/`, `frontend/`, `docker/`, `docs/`, `tests/e2e/`. Do NOT add a new top-level dir without updating `docs/ARCH.md`.
- **Inside `backend/app/` — strict layers:** `api/` (routers only, no SQLAlchemy/Redis/external calls) → `services/` (business logic, tx boundaries, cache/memory invalidation) → `repositories/` (SQL only); `domain/` Pydantic distinct from ORM; `infra/` for Vault/MinIO/Redis/LLM/tracing/redaction.
- **Refuse to boot** if: Vault unreachable, classifier weights missing, classifier SHA-256 mismatches `model_card.json`, tracing misconfigured, any `eval_thresholds.yaml` value is zero/disabled.

### Secrets (the grade)
- Every secret resolves from Vault at startup. **No `os.getenv` outside `backend/app/config.py` and `backend/app/infra/vault.py`.**
- `.env` carries only the Vault root token and ports. Never anything else.
- `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` MUST return zero matches outside Vault-reading code. Run before every commit.
- Pre-commit `gitleaks` is required, never skipped.

### Logging & redaction (the grade)
- `backend/app/infra/redaction.py::redact_text()` runs before any log line, trace span attribute, or memory write leaves the service boundary.
- Never use `print()` for operational output. `structlog` everywhere. JSON-structured.
- Every log line carries `trace_id` and `request_id` for the request.
- Never log: passwords, tokens, full request/response bodies with PII, secrets, connection strings.
- A redaction test asserts a fake `sk-...` key never appears unredacted in logs/traces/memory. **It must stay green.**

### Tracing (the grade)
- Every LLM call, tool call, and RAG retrieval is a Langfuse span. Conversation = trace tree rooted at the user message.
- Span attributes: model, prompt/completion tokens, latency, tool I/O **after redaction**.
- Trace ID + request ID are joinable across logs and Langfuse.

### Exceptions (the grade)
- Domain exceptions live in `backend/app/domain/errors.py` (`NotFoundError`, `PermissionDenied`, `ToolFailure`, `ValidationError`). They are NOT `HTTPException`. Services raise them; the API boundary handler in `backend/app/api/exceptions.py` maps them to HTTP `{code, request_id, message}`.
- **Never** bare `except:`. **Never** silently swallow. Never expose stack traces to users.
- Tool failures inside the chatbot are caught and returned to the LLM as structured `ToolError(error, retryable)`. The agent decides what to do. It does NOT 500.
- Layer-3 failure isolation: classifier down ⇒ chatbot says so and continues; it does NOT return 500.

### Async (the grade)
- Every route, tool, and external call is async. **No `requests`** anywhere — use `httpx`. **No `time.sleep`** — use `await asyncio.sleep`.
- CPU-bound calls (`model.predict`) go through `asyncio.to_thread()` if they cost more than a few ms.
- `await asyncio.gather()` for independent calls in parallel.

### Dependency Injection (the grade)
- Every dependency (DB session, LLM, current user, tracer) is declared with `Depends()`. **No globals**, no module-level singletons constructed at import time.
- Heavy singletons (model, embedder, engine, LLM client) are built in `lifespan()` on `app.state`, exposed through dependencies, disposed on shutdown.
- Tests use `app.dependency_overrides` to inject fakes. **Do not monkey-patch imports.**

### Caching
- `functools.lru_cache(maxsize=1)` for `get_settings()` and other deterministic helpers. Never for things that take mutable args, expire, or depend on time.
- `cachetools.TTLCache` + `asyncio.Lock` for external-API responses that have a justified freshness window. **Document the TTL in `docs/DECISIONS.md`.**

### Configuration
- One `Settings(BaseSettings)` in `backend/app/config.py`. `extra="forbid"`. All required fields use `Field(...)`. Tests construct `Settings(...)` directly with overrides.
- Never `os.getenv` outside this file and the Vault adapter.

### Types & boundaries
- Type hints required on every function signature (`mypy --strict` enforced).
- Pydantic at every boundary: HTTP request bodies, LLM tool-call inputs/outputs, LLM structured outputs, webhook payloads.
- Inside the system, **trust your types**. Don't sprinkle defensive `isinstance` checks.

### Errors, retries, failure isolation
- Every external call has a `timeout`. Default 10s.
- Use `tenacity` with `stop_after_attempt(3)`, `wait_exponential`, `retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))`, `reraise=True`. **Do not retry 4xx.**
- Tools return `ToolError` (not raise) so the agent loop continues.

### Code style
- Python 3.12. **Ruff** is the linter + formatter (line-length 100, `select = ["E","F","I","B","UP","ASYNC","S"]`, `ignore = ["E501"]`). **mypy --strict**.
- PEP 8: 4-space indent, double quotes, trailing commas in multi-line. Imports: stdlib / third-party / local, blank-line-separated, isort/ruff sorted.
- Naming: `snake_case` variables/functions/modules; `PascalCase` classes; `UPPER_SNAKE_CASE` constants; `_leading_underscore` private; descriptive names (`calculate_total_price`, not `calc`); booleans read as questions (`is_active`, `has_permission`); collections plural; functions start with a verb.

### Branches, commits, PRs
- Branches: `<type>/<short-description>` — `feature/`, `bugfix/`, `hotfix/`, `refactor/`, `docs/`, `test/`, `chore/`. Lowercase + hyphens.
- Commits: **Conventional Commits** — `<type>(<scope>): <imperative summary under 72 chars>`. No trailing period. Body optional after blank line.
- PR title: `[TYPE] Short imperative description`. PR body uses the template (Summary / Changes / Testing / Screenshots / Checklist).
- PRs < 400 lines. One concern per PR. Squash on merge.
- **Never commit directly to `main`. Never `--no-verify`. Never `--force-push` to `main`.**

### Testing
- File names: `test_<thing>.py`. Function names: `test_<what>_<expected>` (e.g., `test_login_with_invalid_password_returns_401`).
- AAA pattern (Arrange, Act, Assert). One assertion-of-meaning per test.
- **Always test:** Pydantic schemas (valid + invalid), tool logic with mocked LLM, every API endpoint's status codes and shapes, error paths, auth & authz, redaction, multi-tenant isolation.
- **Coverage:** ≥80% line coverage overall; ≥95% on critical paths (auth, data mutations, redaction, memory writes).
- `uv run pytest --cov=app --cov-report=term-missing`. Coverage is a floor, not a target — meaningful tests over hitting a number.
- Slow tests tagged `@pytest.mark.eval`; CI runs them; local runs skip by default.
- Use `app.dependency_overrides` to inject fakes. Do not monkey-patch imports.

### Dependencies — `uv` only
- Source of truth: `pyproject.toml`. Lockfile: `uv.lock` (committed). Runtime: `uv sync --frozen`.
- Add a dep: `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (dev). Never hand-edit `pyproject.toml` deps without re-running `uv lock`.
- Run commands inside the env: `uv run ruff check .`, `uv run mypy .`, `uv run pytest -q`. **No system `pip install`.**
- Production Docker images: `uv sync --frozen --no-dev`. Dev images: `uv sync --frozen`.
- Review every new dependency: maintained? CVEs? Does stdlib already cover it?
- Run `uvx pip-audit` (or `safety check`) in CI.

### Documentation
- Public modules / classes / functions: Google-style docstrings with Args / Returns / Raises.
- Inline comments only when WHY is non-obvious. Don't restate WHAT.
- `README.md` is the front door: description, setup, env vars list, link to `/docs` (FastAPI auto-gen), architecture pointer. Plus the five required: `docs/ARCH.md`, `docs/DECISIONS.md`, `docs/RUNBOOK.md`, `docs/EVALS.md`, `docs/SECURITY.md`.

---

## What you must NOT do in this repo

- Do not commit to `main`. Always branch.
- Do not run destructive git commands (`reset --hard`, `push --force`, branch deletion) without explicit user confirmation.
- Do not skip pre-commit hooks (`--no-verify`).
- Do not introduce a new top-level dir without asking.
- Do not add a dependency without justifying it in the commit message and running `uv add` (never hand-edit `pyproject.toml`).
- Do not write code that bypasses Vault for secrets, redaction for logs, or the boundary handler for errors.
- Do not auto-write memory. The LLM must call the `write_memory` tool explicitly; every write must produce an audit row.
- Do not regress any threshold in `backend/eval/eval_thresholds.yaml`. If a threshold needs to change, change it in a dedicated PR with the `eval_report.json` diff attached.
- Do not ship code you cannot defend orally on Friday. **No vibe coding.**
- Do not import from `backend/` inside `frontend/streamlit_ui/` — Streamlit is an HTTP client of the backend, same as the React widget.

---

## Pre-commit pipeline (runs on every commit, do not skip)

```
uv run ruff format . → uv run ruff check . → uv run mypy --strict app/ → uv run pytest -q -m "not eval" → gitleaks detect
```

The same pipeline runs in CI, plus the eval suites and a smoke `docker-compose up`.

---

## Acceptance checklist (run before Friday demo)

Passing all 13 = ready to ship.

1. **Fresh clone smoke:** `git clone <repo> && cp .env.example .env && <paste vault token> && docker-compose up --build` — every service green within ~60s.
2. **Static checks:** `cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy --strict app/` — clean. Same in `frontend/streamlit_ui/`. In `frontend/widget/`: `npm run lint && npm run typecheck`.
3. **Secret hygiene:** `grep -ri 'sk-' backend/` and `grep -ri 'password' backend/` — empty outside Vault-reading code; `gitleaks detect` — clean.
4. **Unit + integration tests:** `cd backend && uv run pytest -q` — green.
5. **Eval suites:** `cd backend && uv run pytest -q -m eval` — both classification + RAG above committed thresholds.
6. **Redaction proof:** test asserts a fake `sk-...` key never appears unredacted in logs/traces/memory.
7. **Refuse-to-boot:** kill Vault, restart `api`, assert non-zero exit with structured error.
8. **Trace tree demo:** open Langfuse, walk a real `/chat` conversation including a tool-failure path.
9. **Cross-conversation memory:** new login → previous conversation's stored memory surfaces in the system prompt.
10. **Widget on allowed host:** demo-host page in browser → widget loads, chat round-trips, message renders.
11. **Widget blocked on disallowed host:** second host page → browser console shows CSP `frame-ancestors` block; widget does not render.
12. **Bundle size:** `wc -c frontend/widget/dist/widget.js` ≤ 200 KB gzipped.
13. **Tag:** `v0.1.0-week7` exists on `main`; CI is green on the tagged commit.

---

## Step completion reports (mandatory)

After every step is committed, write a brief Markdown report to `docs/reports/YYYY-MM-DD-<step-slug>.md` **without being asked**. The report must contain:

1. **What was built** — files created/modified and their purpose (2–4 bullet points).
2. **Tests written** — list each test file and what it asserts.
3. **Checks run** — ruff, mypy, pytest results.
4. **What's next** — one sentence naming the next step.

Keep it under 40 lines. Commit the report in the same commit as the step, or in a follow-up `docs:` commit immediately after. Never skip this.

---

## Submission

Tag `v0.1.0-week7`. Public repo. `docker-compose up` succeeds from a fresh clone after `cp .env.example .env` and pasting the Vault root token. Both eval suites pass in CI.

> *"Engineering is the discipline of writing code that other people can change without fear."* — Hasan, Engineering Standards Companion Guide

Ship it.
