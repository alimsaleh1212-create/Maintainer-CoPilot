#!/bin/sh
# =============================================================================
# vault-init.sh — Seed application secrets from .env into HashiCorp Vault.
#
# This script is mounted into the vault-init container and executed once at
# stack startup. It:
#   1. Waits until Vault is healthy.
#   2. Reads /secrets/.env (mounted from the project root .env).
#   3. Maps each application secret (allowlist) to its Vault key.
#   4. Merges in compose-fixed infra connections (db, ollama URLs).
#   5. Writes everything to secret/copilot in a single atomic operation.
#   6. Verifies by listing the seeded keys (never values).
#
# Adding a new secret:
#   a. Add KEY=value to .env
#   b. Add one line to the case-statement mapping below
#   c. Restart vault-init: docker compose restart vault-init
#
# POSIX sh — works on Alpine (hashicorp/vault image is Alpine-based).
# =============================================================================
set -eu

ENV_FILE="/secrets/.env"
VAULT_PATH="secret/copilot"

# -----------------------------------------------------------------------------
# 1. Wait for Vault to be ready
# -----------------------------------------------------------------------------
echo "[vault-init] Waiting for Vault at $VAULT_ADDR ..."
while ! vault status -address="$VAULT_ADDR" >/dev/null 2>&1; do
  sleep 1
done
echo "[vault-init] Vault is healthy."

# -----------------------------------------------------------------------------
# 2. Verify .env was mounted
# -----------------------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
  echo "[vault-init] ERROR: $ENV_FILE not mounted." >&2
  echo "[vault-init]   Expected mount: ../.env:/secrets/.env:ro" >&2
  exit 1
fi
echo "[vault-init] Reading secrets from $ENV_FILE"

# -----------------------------------------------------------------------------
# 3. Build the kv-put argument list from .env (allowlist mapping)
#
#    Each iteration parses a KEY=value pair, strips surrounding quotes, and
#    maps it to its Vault key. Unknown keys are silently skipped — only the
#    keys listed below are ever written to Vault. This prevents accidental
#    secret leakage if .env accumulates unrelated variables.
# -----------------------------------------------------------------------------
ARGS=""

while IFS= read -r line || [ -n "$line" ]; do
  # Skip blank lines and comments
  case "$line" in
    ''|\#*) continue ;;
  esac

  # Split on the first '=' (handles values containing '=')
  key=$(printf '%s' "$line" | cut -d'=' -f1)
  value=$(printf '%s' "$line" | cut -d'=' -f2-)

  # Strip surrounding single or double quotes from the value
  value=$(printf '%s' "$value" | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")

  # Map .env key (UPPER_SNAKE) → Vault key (lower_snake). Allowlist only.
  case "$key" in
    GEMINI_API_KEY)      vault_key="gemini_api_key" ;;
    GEMINI_MODEL)        vault_key="gemini_model" ;;
    OLLAMA_CHAT_MODEL)   vault_key="ollama_chat_model" ;;
    OLLAMA_EMBED_MODEL)  vault_key="ollama_embed_model" ;;
    JWT_SIGNING_KEY)     vault_key="jwt_signing_key" ;;
    MINIO_ACCESS_KEY)    vault_key="minio_access_key" ;;
    MINIO_SECRET_KEY)    vault_key="minio_secret_key" ;;
    LANGFUSE_PUBLIC_KEY) vault_key="langfuse_public_key" ;;
    LANGFUSE_SECRET_KEY) vault_key="langfuse_secret_key" ;;
    GITHUB_TOKEN)        vault_key="github_token" ;;
    *)                   continue ;;
  esac

  # Skip empty values — Vault rejects empty strings for some operations.
  if [ -z "$value" ]; then
    echo "[vault-init]   skipping $vault_key (empty value in .env)"
    continue
  fi

  ARGS="$ARGS $vault_key=$value"
done < "$ENV_FILE"

# -----------------------------------------------------------------------------
# 4. Add compose-fixed infra connections (depend on the docker network, so
#    they belong in the compose file rather than .env)
# -----------------------------------------------------------------------------
ARGS="$ARGS ollama_host=http://ollama:11434"
ARGS="$ARGS database_url=postgresql+asyncpg://copilot:copilot@db:5432/copilot"

# -----------------------------------------------------------------------------
# 5. Apply defaults for any required key still missing
# -----------------------------------------------------------------------------
case "$ARGS" in
  *gemini_model=*)        ;;
  *) ARGS="$ARGS gemini_model=gemini-2.0-flash" ;;
esac
case "$ARGS" in
  *ollama_chat_model=*)   ;;
  *) ARGS="$ARGS ollama_chat_model=gemma3:4b" ;;
esac
case "$ARGS" in
  *ollama_embed_model=*)  ;;
  *) ARGS="$ARGS ollama_embed_model=nomic-embed-text" ;;
esac
case "$ARGS" in
  *jwt_signing_key=*)     ;;
  *) ARGS="$ARGS jwt_signing_key=dev-jwt-signing-key-change-in-production" ;;
esac
case "$ARGS" in
  *minio_access_key=*)    ;;
  *) ARGS="$ARGS minio_access_key=minioadmin" ;;
esac
case "$ARGS" in
  *minio_secret_key=*)    ;;
  *) ARGS="$ARGS minio_secret_key=minioadmin" ;;
esac
case "$ARGS" in
  *langfuse_public_key=*) ;;
  *) ARGS="$ARGS langfuse_public_key=dev-langfuse-public-key" ;;
esac
case "$ARGS" in
  *langfuse_secret_key=*) ;;
  *) ARGS="$ARGS langfuse_secret_key=dev-langfuse-secret-key" ;;
esac

# -----------------------------------------------------------------------------
# 6. Write to Vault atomically
# -----------------------------------------------------------------------------
echo "[vault-init] Seeding $VAULT_PATH ..."
# shellcheck disable=SC2086
vault kv put -address="$VAULT_ADDR" "$VAULT_PATH" $ARGS >/dev/null
echo "[vault-init] ✓ Secrets seeded successfully."

# -----------------------------------------------------------------------------
# 7. Print seeded keys (NEVER values — values may be secrets)
# -----------------------------------------------------------------------------
echo ""
echo "[vault-init] Keys now in $VAULT_PATH:"
# `vault kv get -field=...` returns a single field; we just want the key list.
# Use the table output and grep for the actual key=value rows.
vault kv get -address="$VAULT_ADDR" "$VAULT_PATH" \
  | awk '/^=+ Data =+/{flag=1; next} /^---/{next} flag && NF>=2 && $1!="Key" {print "  - " $1}'
