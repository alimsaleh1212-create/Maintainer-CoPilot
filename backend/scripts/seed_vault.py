"""Seed Vault with dev-mode placeholder secrets.

Run once after `docker-compose up` to populate the KV-v2 path that the app
reads at startup. Real values (Gemini key, etc.) must be filled in manually
or via CI secrets injection.

Usage:
    cd backend
    VAULT_ADDR=http://localhost:8200 VAULT_ROOT_TOKEN=<token> uv run python scripts/seed_vault.py
"""

from __future__ import annotations

import os
import sys

from app.infra.vault import VaultClient, VaultUnreachable

_DEV_DEFAULTS: dict[str, str] = {
    "gemini_api_key": "REPLACE_WITH_REAL_GEMINI_KEY",
    "ollama_host": "http://ollama:11434",
    "jwt_signing_key": "dev-jwt-signing-key-change-in-prod",
    "database_url": "postgresql+asyncpg://copilot:copilot@db:5432/copilot",
    "minio_access_key": "minioadmin",
    "minio_secret_key": "minioadmin",
    "langfuse_public_key": "REPLACE_WITH_LANGFUSE_PUBLIC_KEY",
    "langfuse_secret_key": "REPLACE_WITH_LANGFUSE_SECRET_KEY",
}


def main() -> None:
    vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
    vault_token = os.getenv("VAULT_ROOT_TOKEN", "")

    if not vault_token:
        print("ERROR: VAULT_ROOT_TOKEN is not set.", file=sys.stderr)
        sys.exit(1)

    client = VaultClient(addr=vault_addr, token=vault_token)

    try:
        client.health_check()
        print(f"Connected to Vault at {vault_addr}")
    except VaultUnreachable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    client.write_secrets(_DEV_DEFAULTS)
    print(f"Seeded {len(_DEV_DEFAULTS)} secrets at secret/copilot")
    print()
    print("Keys written:")
    for key in _DEV_DEFAULTS:
        placeholder = (
            "(placeholder — replace with real value)" if "REPLACE" in _DEV_DEFAULTS[key] else ""
        )
        print(f"  {key} {placeholder}")
    print()
    print("Next: replace REPLACE_* values with real credentials, then restart the api container.")


if __name__ == "__main__":
    main()
