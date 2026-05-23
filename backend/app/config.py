"""Application settings — the single source of truth for all configuration.

Rules (enforced by CLAUDE.md):
- os.getenv is ONLY allowed in this module and app.infra.vault.
- All secrets resolve from Vault via get_settings().
- Tests construct Settings(...) directly with explicit values.
- get_settings() is lru_cached so Vault is hit once per process.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Typed, validated application configuration.

    Construct via get_settings() in application code.
    Construct directly in tests (pass all required fields explicitly).
    """

    model_config = ConfigDict(extra="forbid")

    # --- Vault address (the one env var the app reads directly) ---------------
    vault_addr: str = Field(...)

    # --- Secrets (resolved from Vault at startup) -----------------------------
    gemini_api_key: str = Field(...)
    ollama_host: str = Field(...)
    jwt_signing_key: str = Field(...)
    database_url: str = Field(...)
    minio_access_key: str = Field(...)
    minio_secret_key: str = Field(...)
    langfuse_public_key: str = Field(...)
    langfuse_secret_key: str = Field(...)

    # --- Non-secret runtime config (sensible defaults for compose) ------------
    api_host: str = Field(default="0.0.0.0")  # noqa: S104
    api_port: int = Field(default=8000)
    model_server_base_url: str = Field(default="http://model-server:8001")
    redis_url: str = Field(default="redis://redis:6379/0")
    minio_endpoint: str = Field(default="minio:9000")
    minio_secure: bool = Field(default=False)
    langfuse_host: str = Field(default="http://langfuse:3000")
    redis_conversation_ttl_seconds: int = Field(default=86400)  # 24 h
    long_term_memory_top_k: int = Field(default=3)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    # --- LLM model names (non-secret; override via env vars) -----------------
    gemini_model: str = Field(default="gemini-2.0-flash")
    ollama_chat_model: str = Field(default="gemma3:4b")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # --- Cascading classifier -------------------------------------------------
    # When DistilBERT confidence < this value the service automatically
    # re-classifies using the LLM (Gemini).  Set to 0.0 to disable cascade.
    classify_cascade_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve secrets from Vault and return a fully-populated Settings instance.

    This is the only place os.getenv is called for VAULT_ADDR and
    VAULT_ROOT_TOKEN. lru_cache ensures Vault is contacted at most once per
    process lifetime (cleared between test cases via get_settings.cache_clear()).

    Raises:
        VaultUnreachable: if Vault cannot be contacted.
        VaultSecretMissing: if any required secret is absent.
    """
    # Local import avoids a circular dependency at module load time.
    from app.infra.vault import VaultClient

    vault_addr = os.getenv("VAULT_ADDR", "http://localhost:8200")
    vault_token = os.getenv("VAULT_ROOT_TOKEN", "")

    client = VaultClient(addr=vault_addr, token=vault_token)
    s = client.resolve_all()

    return Settings(
        vault_addr=vault_addr,
        gemini_api_key=s["gemini_api_key"],
        ollama_host=s["ollama_host"],
        jwt_signing_key=s["jwt_signing_key"],
        database_url=s["database_url"],
        minio_access_key=s["minio_access_key"],
        minio_secret_key=s["minio_secret_key"],
        langfuse_public_key=s["langfuse_public_key"],
        langfuse_secret_key=s["langfuse_secret_key"],
        # Model names live in Vault so they can be changed without rebuild.
        gemini_model=s["gemini_model"],
        ollama_chat_model=s["ollama_chat_model"],
        ollama_embed_model=s["ollama_embed_model"],
    )
