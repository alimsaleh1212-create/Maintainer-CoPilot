"""FastAPI dependency providers.

Every dependency the routes need is declared here. Tests override these
via app.dependency_overrides — never monkey-patch imports.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import structlog
from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings

logger = structlog.get_logger(__name__)


def get_settings(request: Request) -> Settings:
    """Return the cached Settings singleton stored in app.state."""
    settings: Settings = request.app.state.settings
    return settings


async def get_db_session(
    request: Request,
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, rolling back on error."""
    session_factory: async_sessionmaker[AsyncSession] = request.app.state.session_factory
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_redis(request: Request) -> Redis:
    """Return the shared Redis client from app.state."""
    redis: Redis = request.app.state.redis
    return redis


# Annotated shorthand aliases for route signatures
SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]
