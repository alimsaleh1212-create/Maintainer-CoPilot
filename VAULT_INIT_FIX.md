# Vault-Init Service — Critical Fix

## Problem Identified

Vault starts in dev mode (empty secret store), but the app requires 8 secrets to be present at startup:

```python
REQUIRED_KEYS = (
    "gemini_api_key",
    "ollama_host",
    "jwt_signing_key",
    "database_url",
    "minio_access_key",
    "minio_secret_key",
    "langfuse_public_key",
    "langfuse_secret_key",
)
```

**Without vault-init:** App boots, tries to resolve secrets from empty Vault, fails with `VaultSecretMissing` error → **app refuses to start**.

## Solution Implemented

Added `vault-init` service to docker-compose that:

1. **Waits for Vault to be healthy**
   ```bash
   while ! vault status -address=http://vault:8200 >/dev/null 2>&1; do sleep 1; done
   ```

2. **Seeds all 8 secrets with dev defaults**
   ```bash
   vault kv put -address=http://vault:8200 secret/copilot \
     gemini_api_key="dev-gemini-key-placeholder" \
     ollama_host="http://ollama:11434" \
     jwt_signing_key="dev-jwt-signing-key-..." \
     database_url="postgresql://copilot:copilot@db:5432/copilot" \
     minio_access_key="minioadmin" \
     minio_secret_key="minioadmin" \
     langfuse_public_key="dev-langfuse-public-key" \
     langfuse_secret_key="dev-langfuse-secret-key"
   ```

3. **Exits after completion** (`restart: "no"`)

## Updated Service Dependencies

**Before:**
```
migrate → vault (healthy)
model-server → vault (healthy)
```

**After:**
```
vault-init → vault (healthy) → completes
migrate → vault-init (completed successfully) → runs
model-server → vault-init (completed successfully) → starts
api → migrate (completed successfully) → starts
```

## Boot Sequence

1. ✅ Vault starts in dev mode
2. ✅ Vault-init waits for Vault health → seeds secrets → exits
3. ✅ Migrate starts → runs alembic migrations → exits
4. ✅ API starts → resolves secrets from Vault → boots successfully
5. ✅ Model-server starts → resolves secrets from Vault → boots successfully

## Dev vs. Production

**Dev:** vault-init container seeds hardcoded dev defaults

**Production:** Either:
- Use Kubernetes secret management to pre-populate Vault before app boots
- Or modify vault-init to read production secrets from a secure source (AWS Secrets Manager, etc.)
- Never use hardcoded production secrets

## Commit

```
fix(docker): add vault-init service to seed required secrets
```

This ensures the full stack boots successfully without manual Vault configuration.
