"""Unit tests for app.infra.vault.VaultClient.

These tests exercise the refusal-to-boot behaviour when Vault is unreachable.
No real Vault instance is needed — we point at a port nothing is listening on.
"""

import pytest

from app.infra.vault import VaultClient, VaultUnreachable

_DEAD_VAULT = VaultClient(addr="http://localhost:19999", token="bad-token")  # noqa: S106


def test_health_check_unreachable_raises() -> None:
    with pytest.raises(VaultUnreachable):
        _DEAD_VAULT.health_check()


def test_resolve_all_unreachable_raises() -> None:
    with pytest.raises(VaultUnreachable):
        _DEAD_VAULT.resolve_all()


def test_write_secrets_unreachable_raises() -> None:
    with pytest.raises(VaultUnreachable):
        _DEAD_VAULT.write_secrets({"key": "value"})
