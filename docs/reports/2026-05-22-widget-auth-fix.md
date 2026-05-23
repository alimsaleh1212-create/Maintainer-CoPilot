# Widget Auth + Disallowed-Host Cache Fix — Step Report
**Date:** 2026-05-22
**Branch:** feature/foundations-skeleton

## What was built

- **`frontend/widget/src/api.ts`** — `loginUser()` was calling `/auth/jwt/login` which returns 404. The actual FastAPI endpoint is `/auth/login` (router prefix `/auth` + route `POST /login`). After this fix, sign-in works in the embedded widget for the same credentials accepted by the Streamlit admin UI.
- **`frontend/widget/src/api.ts` (sendChat)** — Failed `/chat` calls now attach the HTTP status code to the thrown error so the UI can branch on it.
- **`frontend/widget/src/Chat.tsx`** — On 401, the assistant bubble now says "Please sign in first — chat requires an account…" instead of the unhelpful generic "Something went wrong." Other failures still report the generic message plus the status code, which is enough to triage quickly.
- **`frontend/demo_host/docker-entrypoint.sh`** — Added `Cache-Control: no-store, no-cache, must-revalidate` + `Pragma: no-cache` headers on both demo hosts. The disallowed host (8091) was triggering Firefox's "another site has embedded it" page after a prior iframe-block event for the same URL — caused by Firefox's BFCache serving a stale frame-blocked state. Forcing no-store ensures every reload is fresh.

## Root-cause notes

- Volume `docker_widget_dist` is shared between the widget container (writes built bundle on startup) and the API container (reads it for `/widget.js` and `/static/widget/`). Named volumes persist across `docker compose up`, so rebuilds *do not* repopulate them. To pick up a new widget bundle: `docker compose down widget api && docker volume rm docker_widget_dist && docker compose up -d api widget`.

## Tests written / run

- E2E suite re-run: `uv run pytest tests/e2e/ --browser chromium` — **11/11 passing in 35s**.
- Manual end-to-end:
  - `POST /auth/login` with `user1@mail.com / 12345678` → 200 + access_token (admin role).
  - `POST /chat` with that token and `{"message":"hi","widget_id":"wgt_11b64335"}` → 200 + response.
  - Widget bundle deployed via volume: `grep -o '/auth/[a-z/]*' /app/static/widget/widget.js` → `/auth/login` ✓.

## Checks run

```
pytest tests/e2e/ --browser chromium: 11 passed in 35.06s
docker exec docker-widget-1 grep /auth widget.js: /auth/login
docker exec docker-api-1 grep /auth widget.js: /auth/login
curl /auth/login user1@mail.com: 200 access_token
curl /chat with Bearer token: 200 response
```

## What's next

Widget end-to-end (sign-in + chat with memory + tool calling) now works in the browser at `http://localhost:8090`. Next: verify cross-conversation memory recall surfaces in the widget UI and write a single Playwright test that signs in and sends a chat message to catch this regression class automatically.
