# E2E Tests, CORS Fix, Chat Sidebar — Step Report
**Date:** 2026-05-22  
**Branch:** feature/foundations-skeleton

## What was built

- **`frontend/streamlit_ui/pages/1_chat.py`** — Full rewrite. Added sidebar conversation history: each conversation stored in `st.session_state.conversations` keyed by UUID, with title (from first user message, 50 chars), timestamp, and message list. "New conversation" button + `›` switch buttons per inactive conversation in sidebar. Conversations persist across page re-runs via session state.

- **Source filter bug fixed** — `st.radio(..., key="source_filter")` replaces the previous pattern of manual `index=` calculation which silently reverted selections mid-conversation. Filter now applies to the next message immediately without needing a new conversation.

- **`backend/app/api/main.py`** — Added `CORSMiddleware(allow_origins=["*"])` so the demo host pages (ports 8090/8091) can cross-origin fetch `/widgets/discover` and `/widget.js`. Without this, browsers silently blocked the fetch and the widget ID was never discovered, preventing iframe injection.

- **`tests/e2e/conftest.py`** — Shared fixtures: `widget_id` (fetches from `/widgets/discover` once per session), `allowed_page` (navigates to 8090), `disallowed_page` (navigates to 8091).

- **`tests/e2e/test_widget_allowed_host.py`** — 6 Playwright tests verifying: page loads, discover resolves widget ID (snippet no longer shows "discovering"), loader script injected in DOM, iframe created with correct src, no CSP frame-ancestors violation in console, toggle button present.

- **`tests/e2e/test_widget_disallowed_host.py`** — 5 Playwright tests verifying: page loads, CSP explanation text visible, loader script still injected (embed is blocked at iframe level, not script level), /embed CSP header enforces `frame-ancestors http://localhost:8090` (8091 absent), browser console emits CSP violation error on disallowed host.

## Tests written

| File | Assertions |
|------|-----------|
| `tests/e2e/test_widget_allowed_host.py` | 6 tests — page title/badge, discover snippet update, script DOM injection, iframe src, no CSP console errors, iframe attached |
| `tests/e2e/test_widget_disallowed_host.py` | 5 tests — page title/badge, CSP explanation text, loader script present, /embed header has `frame-ancestors localhost:8090` and NOT `localhost:8091`, browser CSP violation in console |

All 11 tests pass: `pytest tests/e2e/ --browser chromium` in 33s.

## Checks run

```
pytest tests/e2e/ --browser chromium: 11 passed in 33.60s
```

Backend unit tests unaffected (CORS middleware is additive).

## What's next

System prompt is already updated (tool-selection rules) and wiki golden set eval is complete (0.893 faithfulness, 0.751 answer_relevancy, 0.760 Hit@5 — all PASS). Next: wire classifer + NER + summarizer endpoints integration tests, then auth flow + chatbot tool-calling end-to-end.
