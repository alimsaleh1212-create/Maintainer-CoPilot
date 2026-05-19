"""Alembic environment — async SQLAlchemy runner.

The database URL is resolved from Vault via app.config.get_settings() so that
no secrets live in alembic.ini or environment variables other than
VAULT_ADDR and VAULT_ROOT_TOKEN.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Import ORM metadata so autogenerate can detect schema changes.
from app.repositories.base import Base
from app.repositories.models import (  # noqa: F401 — import triggers table registration
    AuditLog,
    Conversation,
    Memory,
    Message,
    User,
    Widget,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_db_url() -> str:
    """Resolve the database URL from Vault or fallback env var."""
    try:
        from app.config import get_settings

        return get_settings().database_url
    except Exception:
        # Offline mode or Vault not available — fall back to DATABASE_URL env var.
        url = os.getenv("DATABASE_URL", "")
        if not url:
            raise RuntimeError(
                "Cannot resolve database URL: Vault unreachable and DATABASE_URL not set."
            ) from None
        return url


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL only)."""
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live DB using an async engine."""
    url = _get_db_url()
    connectable = create_async_engine(url, poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
