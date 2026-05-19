"""Unit tests for app.config.Settings.

Tests construct Settings directly — no Vault connection needed.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings

_REQUIRED: dict[str, str] = {
    "vault_addr": "http://localhost:8200",
    "gemini_api_key": "gemini-key",
    "ollama_host": "http://localhost:11434",
    "jwt_signing_key": "super-secret",
    "database_url": "postgresql+asyncpg://u:p@localhost/db",
    "minio_access_key": "access",
    "minio_secret_key": "minio-secret",
    "langfuse_public_key": "pk-test",
    "langfuse_secret_key": "sk-test",
}


def test_settings_valid_constructs() -> None:
    s = Settings(**_REQUIRED)
    assert s.vault_addr == "http://localhost:8200"
    assert s.gemini_api_key == "gemini-key"


def test_settings_extra_field_raises() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Settings(**_REQUIRED, unknown_field="bad")


def test_settings_missing_required_field_raises() -> None:
    incomplete = {k: v for k, v in _REQUIRED.items() if k != "gemini_api_key"}
    with pytest.raises(ValidationError):
        Settings(**incomplete)


def test_settings_defaults_are_correct() -> None:
    s = Settings(**_REQUIRED)
    assert s.api_port == 8000
    assert s.api_host == "0.0.0.0"  # noqa: S104
    assert s.redis_conversation_ttl_seconds == 86400
    assert s.long_term_memory_top_k == 3
    assert s.log_level == "INFO"
    assert s.environment == "development"
    assert s.minio_secure is False
