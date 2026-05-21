"""Vault adapter — boot-time secret resolution.

Reads secrets from HashiCorp Vault KV-v2. The app refuses to start if Vault
is unreachable or any required key is absent. This is the only module (besides
app.config) allowed to read VAULT_ADDR and VAULT_ROOT_TOKEN from the environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import hvac  # type: ignore[import-untyped]
import hvac.exceptions  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_MOUNT = "secret"
_PATH = "copilot"

# Canonical list of all secrets the rest of the app needs.
# Add new secrets here first; Settings will fail to construct if one is missing.
REQUIRED_KEYS: tuple[str, ...] = (
    "gemini_api_key",
    "ollama_host",
    "jwt_signing_key",
    "database_url",
    "minio_access_key",
    "minio_secret_key",
    "langfuse_public_key",
    "langfuse_secret_key",
    # LLM model names — not secrets, but stored in Vault so they can be changed
    # without a container rebuild.  Seed via vault-init or scripts/seed_vault.py.
    "gemini_model",
    "ollama_chat_model",
    "ollama_embed_model",
)


class VaultUnreachable(RuntimeError):
    """Raised when Vault cannot be contacted or the token is rejected at startup."""


class VaultSecretMissing(RuntimeError):
    """Raised when a required key is absent from the Vault secret."""


@dataclass(frozen=True)
class VaultClient:
    """Thin wrapper around hvac for KV-v2 secret resolution."""

    addr: str
    token: str

    def _hvac(self) -> hvac.Client:
        return hvac.Client(url=self.addr, token=self.token)

    def health_check(self) -> None:
        """Assert Vault is reachable and the token is authenticated.

        Raises:
            VaultUnreachable: if unreachable or token invalid.
        """
        try:
            client = self._hvac()
            if not client.is_authenticated():
                raise VaultUnreachable(f"Vault token is invalid or expired (addr={self.addr!r})")
        except VaultUnreachable:
            raise
        except hvac.exceptions.VaultError as exc:
            raise VaultUnreachable(f"Vault error at {self.addr!r}: {exc}") from exc
        except Exception as exc:
            raise VaultUnreachable(
                f"Vault unreachable at {self.addr!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def resolve_all(self) -> dict[str, str]:
        """Resolve all required secrets in a single Vault round-trip.

        Returns:
            Mapping of key → value for every key in REQUIRED_KEYS.

        Raises:
            VaultUnreachable: if Vault cannot be contacted.
            VaultSecretMissing: if any required key is absent from the secret.
        """
        try:
            client = self._hvac()
            response: dict[str, Any] = client.secrets.kv.v2.read_secret_version(
                path=_PATH,
                mount_point=_MOUNT,
                raise_on_deleted_version=True,
            )
            data: dict[str, str] = response["data"]["data"]
        except hvac.exceptions.InvalidPath as exc:
            raise VaultSecretMissing(
                f"Vault secret path {_PATH!r} does not exist in mount {_MOUNT!r}. "
                "Seed Vault first with scripts/seed_vault.py."
            ) from exc
        except VaultSecretMissing:
            raise
        except hvac.exceptions.VaultError as exc:
            raise VaultUnreachable(f"Vault error reading secrets: {exc}") from exc
        except Exception as exc:
            raise VaultUnreachable(f"Vault unreachable: {type(exc).__name__}: {exc}") from exc

        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise VaultSecretMissing(
                f"The following keys are missing from Vault path {_PATH!r}: {missing!r}. "
                "Re-run scripts/seed_vault.py to populate them."
            )

        logger.info("vault secrets resolved", extra={"keys": list(REQUIRED_KEYS)})
        return {k: data[k] for k in REQUIRED_KEYS}

    def write_secrets(self, secrets: dict[str, str]) -> None:
        """Write (or overwrite) secrets at the canonical path.

        Used by scripts/seed_vault.py — not called by the app at runtime.

        Args:
            secrets: key→value pairs to store.

        Raises:
            VaultUnreachable: if Vault cannot be contacted.
        """
        try:
            client = self._hvac()
            client.secrets.kv.v2.create_or_update_secret(
                path=_PATH,
                mount_point=_MOUNT,
                secret=secrets,
            )
        except hvac.exceptions.VaultError as exc:
            raise VaultUnreachable(f"Vault error writing secrets: {exc}") from exc
        except Exception as exc:
            raise VaultUnreachable(f"Vault unreachable: {type(exc).__name__}: {exc}") from exc
