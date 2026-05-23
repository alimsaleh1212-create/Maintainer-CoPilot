# Widget Tool Filtering + Per-User Conversations + Tool Chips — Step Report
**Date:** 2026-05-22
**Branch:** feature/foundations-skeleton

## What was built

### Backend
- **`ChatRequest`** now accepts `widget_id` (optional). The `/chat` route fetches the widget, reads its `enabled_tools`, and passes the list to the chatbot. When the widget is missing → 404; when disabled → 403.
- **`ChatbotService`** filters the LLM tool-schema list by `enabled_tools` using a short-name→schema-name map (`classify→classify_issue`, `ner→extract_entities`, `summarize→summarize_text`). Empty list ⇒ no tools; the chatbot falls back to plain `chat()` with a system-prompt addendum so it doesn't promise to look anything up.
- **`MemoryService.touch_conversation` / `verify_conversation_owner` / `list_user_conversations`** — per-user conversation index in Redis: `user:{user_id}:conversations` sorted set + `conv:{cid}:meta` hash with `{user_id, title, created_at, updated_at}`. Ownership is enforced on every read.
- **`GET /chat/conversations`** and **`GET /chat/conversations/{id}`** — list and fetch the authenticated user's conversations. Cross-user access returns 403.
- **`/widgets/discover`** is now origin-aware: picks the enabled widget whose `allowed_origins` lists the calling `Origin`/`Referer`. Single-widget setups still work.
- **`gemini.chat()`** hardened against MAX_TOKENS / SAFETY responses that omit `content.parts[].text`.

### Frontend
- **Widget `Chat.tsx`** renders monochrome SVG+label tool chips under each assistant reply (RAG search / Classify / NER / Summarize / Memory). Dark-tech palette: `rgba(34,197,94,.10)` background, `rgba(34,197,94,.30)` border, `#22c55e` text, JetBrains Mono.
- **Streamlit chat sidebar** loads `GET /chat/conversations` on login. Session state is now keyed by `user_id` (`conversations_by_user`, `active_conv_by_user`) so a different user logging into the same browser sees only their own conversations. Lazy-loads message history when a sidebar item is clicked.

## Tests written / re-run
- `tests/e2e/conftest.py` — added `_widget_id_for_origin()` and a `disallowed_widget_id` fixture so each test gets the widget bound to its host (now that there are two widgets in the DB).
- `tests/e2e/test_widget_disallowed_host.py::test_embed_endpoint_has_csp_frame_ancestors` no longer hardcodes 8090; asserts the directive names exactly one demo-host origin.
- `pytest tests/e2e/ --browser chromium`: **11/11 passing in 45s**.

## Manual verification (curl, all green)
- `POST /chat widget_id=wgt_28bf2ded` (5 tools) + NER prompt → `tools_used: ["extract_entities"]`.
- `POST /chat widget_id=wgt_62aaf998` (0 tools) + same prompt → `tools_used: []`, plain-text response.
- `GET /chat/conversations` (user1) → 6 rows. (user2 registered) → `[]`.
- `GET /chat/conversations/<user1_conv>` as user2 → `403 permission_denied`.
- Widget on 8090 fetches discover with `Origin: http://localhost:8090` → returns the 8090-bound widget.

## Checks run
- `ruff check`: backend changes clean.
- `mypy --strict` on changed files: clean except for pre-existing redis-py async-typing warnings flagged across the file (unchanged from before).
- `tsc --noEmit` on widget: clean.

## What's next
Memory inspector page should also scope by user_id (currently does so via JWT but could benefit from a sidebar filter). The full chat round-trip in the widget (sign-in → ask → see chips) should be added as a Playwright test next.
