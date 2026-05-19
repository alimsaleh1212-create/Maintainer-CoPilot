"""FastAPI application factory and lifespan.

Lifespan contract (CLAUDE.md — Refuse to boot):
- Vault unreachable or secrets missing → sys.exit(1)
- DB unreachable → sys.exit(1)
- Redis unreachable → sys.exit(1)
- Classifier weights missing / SHA-256 mismatch → sys.exit(1)  [wired TUE]
- eval_thresholds.yaml threshold at zero → sys.exit(1)           [wired TUE]
- Langfuse misconfigured → sys.exit(1)                           [wired WED]

Singletons built here are stored on app.state and accessed via
app.api.dependencies — never as module-level globals.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.api.exceptions import add_exception_handlers
from app.api.routes import health
from app.config import Settings, get_settings
from app.infra.vault import VaultSecretMissing, VaultUnreachable

logger = structlog.get_logger(__name__)


def _configure_logging(log_level: str) -> None:
    import logging

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def _verify_db(engine: AsyncEngine) -> None:
    """Probe the DB with a cheap query; raises on failure."""
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _verify_redis(redis: Redis) -> None:
    """Ping Redis; raises on failure."""
    result = redis.ping()
    if hasattr(result, "__await__"):
        await result


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build singletons on startup; dispose on shutdown.

    Any failure here is fatal — the process exits non-zero so the container
    orchestrator knows the service failed to start cleanly.
    """
    # ------------------------------------------------------------------
    # 1. Resolve settings (Vault boot check)
    # ------------------------------------------------------------------
    try:
        settings: Settings = get_settings()
    except VaultUnreachable as exc:
        logger.critical("refuse_to_boot", reason="vault_unreachable", detail=str(exc))
        sys.exit(1)
    except VaultSecretMissing as exc:
        logger.critical("refuse_to_boot", reason="vault_secret_missing", detail=str(exc))
        sys.exit(1)

    _configure_logging(settings.log_level)
    logger.info("settings_resolved", environment=settings.environment)

    # ------------------------------------------------------------------
    # 2. Database engine
    # ------------------------------------------------------------------
    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )
    try:
        await _verify_db(engine)
    except Exception as exc:
        logger.critical("refuse_to_boot", reason="db_unreachable", detail=str(exc))
        await engine.dispose()
        sys.exit(1)

    session_factory: async_sessionmaker[Any] = async_sessionmaker(
        engine, expire_on_commit=False
    )

    # ------------------------------------------------------------------
    # 3. Redis
    # ------------------------------------------------------------------
    redis: Redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await _verify_redis(redis)
    except Exception as exc:
        logger.critical("refuse_to_boot", reason="redis_unreachable", detail=str(exc))
        await engine.dispose()
        await redis.aclose()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Store singletons on app.state
    # ------------------------------------------------------------------
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis

    logger.info("startup_complete", services=["db", "redis"])

    yield

    # ------------------------------------------------------------------
    # Shutdown — dispose connections gracefully
    # ------------------------------------------------------------------
    await engine.dispose()
    await redis.aclose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="Maintainer's Copilot API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    add_exception_handlers(app)
    app.include_router(health.router)

    return app


# Module-level singleton used by uvicorn entrypoint.
app = create_app()
