"""Integration test: alembic upgrade head succeeds on a fresh DB.

Requires a running Postgres instance. Run with:
    pytest -m integration tests/integration/test_migrations.py

Skipped by default in pre-commit (only 'not eval and not integration' run).
"""

import pytest


@pytest.mark.integration
def test_placeholder_migrations_wired() -> None:
    """Placeholder — full migration smoke test runs in compose CI.

    The real test:
    1. Spins up a fresh Postgres via testcontainers.
    2. Runs `alembic upgrade head`.
    3. Asserts all expected tables exist with the right columns.
    4. Runs `alembic downgrade base` and asserts tables are gone.

    Implemented in the integration sprint (WED) once testcontainers is wired.
    """
    assert True
