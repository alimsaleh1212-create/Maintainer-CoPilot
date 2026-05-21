# Live Demo Guide — Maintainer's Copilot
**Audience:** Technical (engineers / TAs) | **Duration:** 10 minutes

---

## Pre-demo checklist (do this 10 min before)

```bash
# 1. Fresh stack — verify all services are healthy
cd docker && docker compose ps

# Expected: db, redis, vault, minio, api, model-server, chatbot, widget,
#           host-allowed, host-disallowed — all "healthy" or "exited 0"

# 2. Open these tabs in the browser now (not during demo):
#   A. Streamlit UI        → http://localhost:8501
#   B. Langfuse            → http://localhost:3000
#   C. FastAPI docs        → http://localhost:8000/docs
#   D. Allowed host        → http://localhost:8090
#   E. Disallowed host     → http://localhost:8091
#   F. pgAdmin             → http://localhost:5050  (optional, for DB live query)

# 3. Register a demo user (if starting fresh)
curl -s -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@maintainer.ai","password":"demo1234"}' | jq .

# 4. Seed a real MONAI issue into the chat history to demo cross-conv memory
#    (skip if you already have prior conversations)
```

---

## Act 1 — Architecture walk (2 min)

**Say:** "This is a layered FastAPI service with six independent components that the LLM calls as tools."

**Show:** `docs/ARCH.md` or describe the stack verbally:

```
HTTP request
   └─ FastAPI (api/) ← Depends(): db_session, current_user, tracer, llm
       └─ ChatbotService
           ├─ tool: classify  → DistilBERT (3-class) via model-server
           ├─ tool: ner       → spaCy + regex code-entity extractor
           ├─ tool: summarize → Gemini Flash with 150-word prompt
           ├─ tool: rag_search → hybrid dense+BM25 → bge-reranker → top-5
           └─ tool: write_memory → pgvector + audit_log row
                                    ↑ always explicit, never auto-written
```

**Key point for TAs:** "Every new endpoint is one file in `api/routes/`, one Pydantic type in `domain/`, one service method, one repo method. Zero churn elsewhere — that's the Friday TA live-add test."

---

## Act 2 — Live triage (2 min)

**In Streamlit UI (tab A):**

1. Sign in as `demo@maintainer.ai` / `demo1234`
2. Go to **Chat**
3. Paste this real MONAI issue:

   > "Getting CUDA out of memory error even with batch_size=1 on V100. Works fine on A100.
   > Using transforms.Compose with RandSpatialCrop. Stack trace:
   > `RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB`"

4. Hit send. Watch the tool calls stream in:
   - `classify` → `bug` (confidence shown)
   - `ner` → entities extracted: `transforms.Compose`, `RandSpatialCrop`, `RuntimeError`
   - `rag_search` → pulls relevant MONAI docs chunks
   - Response → structured triage with suggested fix

**Point out:** "The LLM decided which tools to call and in what order — no hand-coded workflow."

---

## Act 3 — Secrets & Vault (1 min)

**In terminal:**

```bash
# Show all secrets are in Vault, never in code
vault kv get -address=http://localhost:8200 secret/copilot
# Output: key names only (gemini_api_key, jwt_signing_key, database_url, etc.)

# Show that grepping the codebase returns nothing outside Vault adapter
grep -ri 'sk-' backend/ | grep -v vault.py | grep -v config.py
# Expected: empty

grep -ri 'password' backend/ | grep -v vault.py | grep -v test | grep -v alembic
# Expected: empty (comments only)
```

**Point for TAs:** "CLAUDE.md enforces this as a pre-commit rule. gitleaks runs on every commit."

---

## Act 4 — Redaction & Tracing (2 min)

**In Langfuse (tab B):**

1. Find the trace for the issue you just submitted
2. Open the trace tree:
   ```
   chat_turn
   ├── llm_call          (model, tokens, latency)
   ├── tool_classify     (input: text, output: {label, confidence})
   ├── tool_extract_entities (entities list)
   ├── tool_rag_search   (query, top_k chunks)
   └── llm_call          (final answer generation)
   ```

3. Click any span → show `tool_inputs` and `tool_outputs` — **no raw API keys, no passwords**

**Point for TAs:** "Every string goes through `redact_text()` before it reaches Langfuse. The unit test `test_redaction_in_traces.py` asserts a fake `AIza...` key never reaches the span attributes."

**Optional — show live redaction:**

```bash
# Run just that test to demonstrate
cd backend && uv run pytest tests/unit/test_redaction_in_traces.py -v
# 14 passed ✓
```

---

## Act 5 — Memory (1.5 min)

**In Streamlit → Chat:**

1. Ask the chatbot to remember something:

   > "Remember: I prefer concise answers with code snippets, no long explanations."

   The LLM should call `write_memory`. Show the audit log entry:

   ```bash
   # psql or pgAdmin: audit_log table
   SELECT action, actor_id, target_type, created_at
   FROM audit_log ORDER BY created_at DESC LIMIT 5;
   ```

2. **Open a new browser tab / incognito** → go to Streamlit → sign in again → Chat

3. Ask anything. Show the system prompt contains the prior memory.

**Point for TAs:** "Memory is explicit — `write_memory` tool must be called. Every write produces an audit row. The unit test `test_write_memory_audit_log.py` asserts this. No auto-writes."

---

## Act 6 — Widget + CSP (1.5 min)

**Tab D — Allowed host (http://localhost:8090):**

1. Show the widget loading in the iframe
2. Send a message → response renders inside the widget
3. Open browser DevTools → Network → find the `/embed` response → show the header:
   ```
   Content-Security-Policy: frame-ancestors http://localhost:8090
   ```

**Tab E — Disallowed host (http://localhost:8091):**

1. Open DevTools → Console
2. Refresh the page
3. Show the browser error:
   ```
   Refused to display '...' in a frame because an ancestor violates
   the following Content Security Policy directive: "frame-ancestors ..."
   ```

**Point for TAs:** "The allowlist lives in the `widgets` table, not in an env var. Change it via the Widget Admin page — it takes effect on the next embed request with no restart."

---

## Act 7 — Eval gates (30 sec)

```bash
cd backend

# Classification gate
uv run pytest tests/eval/test_classification_thresholds.py -v
# macro_f1=0.764 ≥ 0.75 ✓  per_class all ≥ 0.55 ✓

# RAG gate
uv run pytest tests/eval/test_rag_thresholds.py -v
# All 8 checks pass ✓

# Full suite
uv run pytest -q
# 142 passed
```

**Point for TAs:** "Both eval suites are tagged `@pytest.mark.eval`. CI runs them on every push. If macro-F1 drops below 0.75, the merge is blocked."

---

## Backup questions & answers

| Question | Answer |
|---|---|
| "Why 3 classes not 4?" | `documentation` had 28 examples → 5-example test set → ±20% F1 swing per error. Routing for docs and Q&A is identical. Merged to `support`. Counts in `DECISIONS.md`. |
| "Why pgvector not Pinecone?" | pgvector is already in Postgres — no second piece of infra. Embedding dim matches `bge-small-en-v1.5` (384). Defended in `DECISIONS.md`. |
| "Why Langfuse not W&B?" | Self-hostable OSS, native LLM trace model, dev-mode in compose. W&B is ML metrics; Langfuse is LLM spans. Defended in `DECISIONS.md`. |
| "Why episodic memory?" | Easiest to demo cross-conversation recall. User summaries are the natural unit for a per-maintainer copilot. |
| "What happens if Vault goes down?" | Cached secrets keep current requests going. New secret resolutions fail fast. API refuses to boot without Vault on cold start. See `RUNBOOK.md`. |
| "Add a new endpoint live" | `api/routes/new_route.py` + Pydantic in `domain/` + service method. Show it in FastAPI `/docs` immediately. |
| "What's the bundle size?" | `uv run pytest tests/unit/test_widget_bundle_size.py -v` — asserts ≤ 200 KB gzipped. |

---

## Things that can go wrong and quick fixes

| Symptom | Fix |
|---|---|
| Widget blank on allowed host | `docker compose restart widget` — volume `widget_dist` sometimes needs a rebuild |
| Langfuse shows no traces | Check `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in Vault; `TracingClient` auto-degrades to no-op on placeholder keys |
| `classify` tool returns error | `docker compose restart model-server` — weights load from MinIO on startup |
| CSP block doesn't show | Hard-refresh the disallowed host (Ctrl+Shift+R); DevTools must be open on Console tab |
| Memory not surfacing | Check Redis TTL — `redis-cli TTL conv:<id>:history`; long-term memory requires `write_memory` tool to have been called explicitly |

---

## Sign-off line

> "Every architectural choice is backed by a number in `DECISIONS.md`.
> Every secret lives in Vault. Every write is audited. Every eval gate is committed.
> From a fresh clone: `cp .env.example .env` → paste Vault token → `docker compose up` → ship."
