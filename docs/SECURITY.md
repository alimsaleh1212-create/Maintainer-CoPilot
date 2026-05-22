# SECURITY.md — Maintainer's Copilot

## Redaction patterns

All patterns are compiled in `backend/app/infra/redaction.py`. `redact_text()` runs before any log line, Langfuse span attribute, or memory write leaves the service boundary.

| Pattern | Regex | Replacement |
|---------|-------|-------------|
| Anthropic API keys | `sk-ant-[a-zA-Z0-9\-_]{20,}` | `[REDACTED:anthropic-key]` |
| OpenAI / generic `sk-` keys | `sk-[a-zA-Z0-9\-_]{20,}` | `[REDACTED:api-key]` |
| Google AI / Gemini keys | `AIza[0-9A-Za-z\-_]{35}` | `[REDACTED:google-key]` |
| GitHub PATs (new format) | `github_pat_[a-zA-Z0-9_]{22,}` | `[REDACTED:github-pat]` |
| GitHub PATs (classic) | `ghp_[a-zA-Z0-9]{36}` | `[REDACTED:github-pat]` |
| Langfuse public/secret keys | `(?:pk\|sk)-lf-[a-zA-Z0-9\-]{20,}` | `[REDACTED:langfuse-key]` |
| AWS IAM access key IDs | `AKIA[0-9A-Z]{16}` | `[REDACTED:aws-key]` |
| JWT tokens | `ey[…].[…].[…]` (three base64url segments) | `[REDACTED:jwt]` |
| PEM private key headers | `-----BEGIN * PRIVATE KEY-----` | `[REDACTED:private-key]` |

**Why this order matters:** Anthropic keys (`sk-ant-`) are matched before the generic `sk-` pattern so the more-specific label is used.

**Test:** `tests/unit/test_redaction.py` asserts that every pattern above is scrubbed from all three emission paths: logger output, Langfuse span attributes, and memory writes. This test is in CI required checks.

---

## Secrets flow (Vault is the only runtime source)

```
.env  →  vault-init container  →  HashiCorp Vault (KV v2 at secret/copilot)
                                         │
                                         ▼
                              backend/app/infra/vault.py
                                         │
                                         ▼
                              backend/app/config.py::Settings
                                         │
                                         ▼
                              app code via Depends(get_settings)
```

**Hard rule:** No `os.getenv` lives outside `backend/app/config.py` and `backend/app/infra/vault.py`. Verified by grep at commit time.

**The `.env` file is a dev bootstrap convenience only.** It is gitignored. The `vault-init` container reads it on stack start, writes each secret into Vault, and exits. In production, secrets are loaded into Vault directly (e.g. `vault kv put secret/copilot @secrets.json`) and `.env` is replaced by `VAULT_ROOT_TOKEN=<token>` + ports only.

If `vault-init` cannot read `.env`, it fails the stack — Vault is never seeded from defaults.

---

## Secrets list (names only — values in Vault)

| Secret | Vault key | Used by |
|--------|-----------|---------|
| Gemini API key | `gemini_api_key` | `app.infra.llm.gemini` |
| Ollama host | `ollama_host` | `app.infra.llm.ollama` |
| JWT signing key | `jwt_signing_key` | `fastapi-users` auth |
| Database URL (with password) | `database_url` | SQLAlchemy engine |
| MinIO access key | `minio_access_key` | `app.infra.minio` |
| MinIO secret key | `minio_secret_key` | `app.infra.minio` |
| Langfuse public key | `langfuse_public_key` | `app.infra.tracing` |
| Langfuse secret key | `langfuse_secret_key` | `app.infra.tracing` |

---

## CSP / CORS posture

- `Content-Security-Policy: frame-ancestors <origins>` is set on the `/embed` route. The value comes from `widgets.allowed_origins` in the DB — **not** a hardcoded env var.
- CORS `allowed_origins` is also sourced from the widget record per request, not from a static list.
- The `host-disallowed` container (port 8091) exists solely to demonstrate that the browser rejects an unlisted origin during the Friday demo.

---

## Audit log fields

Every `audit_log` row contains: `actor_id`, `action` (e.g. `memory.write`, `widget.update`), `target_type`, `target_id`, `timestamp`, `extra_metadata` (redacted JSON context). Rows are immutable — no UPDATE or DELETE is issued against `audit_log`.

---

## Refuse-to-boot policy

`api` exits non-zero if any of the following are true at startup:
1. Vault is unreachable or the token is rejected.
2. Any required Vault secret key is missing.
3. The database is unreachable.
4. Redis is unreachable.
5. Classifier weights are missing or SHA-256 hash mismatches `model_card.json`. *(wired TUE)*
6. Any threshold in `eval_thresholds.yaml` is zero or disabled. *(wired TUE)*
7. Langfuse is misconfigured. *(wired WED)*

---

## Response shape policy

Users never see a stack trace. All error responses have shape:
```json
{"code": "<domain_code>", "message": "<safe_message>", "request_id": "<uuid>"}
```
Stack traces are logged server-side with the `trace_id` + `request_id` for correlation in Langfuse.
