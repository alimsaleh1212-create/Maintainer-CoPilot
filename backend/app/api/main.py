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
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.api.exceptions import add_exception_handlers
from app.api.routes import auth, chat, classify, embed, health, memory, ner, rag, summarize, widgets
from app.config import Settings, get_settings
from app.infra.llm.gemini import GeminiClient
from app.infra.llm.ollama import OllamaClient
from app.infra.redaction import structlog_redaction_processor
from app.infra.vault import VaultSecretMissing, VaultUnreachable
from app.rag.embeddings import get_embedding_model

logger = structlog.get_logger(__name__)


def _configure_logging(log_level: str) -> None:
    import logging

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog_redaction_processor,  # scrub secrets before any emission
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

    session_factory: async_sessionmaker[Any] = async_sessionmaker(engine, expire_on_commit=False)

    # ------------------------------------------------------------------
    # 3. Embedding model (for RAG via Ollama)
    # ------------------------------------------------------------------
    try:
        embedder = get_embedding_model()
        await embedder.ensure_model_pulled()
        logger.info("embeddings_ready", model=embedder.model_name, host=embedder.ollama_host)
    except Exception as exc:
        logger.critical("refuse_to_boot", reason="embeddings_failed", detail=str(exc))
        await engine.dispose()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Redis
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
    # 5. LLM clients (primary Gemini + fallback Ollama)
    # ------------------------------------------------------------------
    ollama_http = httpx.AsyncClient(timeout=60.0)
    gemini_client = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        ollama_host=settings.ollama_host,
    )
    ollama_client = OllamaClient(
        base_url=settings.ollama_host,
        model=settings.ollama_chat_model,
        embed_model=settings.ollama_embed_model,
        http_client=ollama_http,
    )
    logger.info(
        "llm_clients_ready",
        gemini_model=settings.gemini_model,
        ollama_model=settings.ollama_chat_model,
    )

    # ------------------------------------------------------------------
    # 6. Store singletons on app.state
    # ------------------------------------------------------------------
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.redis = redis
    app.state.embedder = embedder
    app.state.gemini_client = gemini_client
    app.state.ollama_client = ollama_client

    logger.info("startup_complete", services=["db", "redis", "embeddings", "llm"])

    yield

    # ------------------------------------------------------------------
    # Shutdown — dispose connections gracefully
    # ------------------------------------------------------------------
    await engine.dispose()
    await redis.aclose()
    await embedder.close()
    await gemini_client.close()
    await ollama_http.aclose()
    logger.info("shutdown_complete")


_WIDGET_STATIC_DIR = Path("/app/static/widget")
_LOADER_SCRIPT = Path(__file__).parent.parent.parent.parent / "static" / "loader.js"


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
    app.include_router(rag.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(classify.router)
    app.include_router(ner.router)
    app.include_router(summarize.router)
    app.include_router(widgets.router)
    app.include_router(embed.router)
    app.include_router(memory.router)

    # Serve the widget JS loader at /widget.js (loaded by demo host pages).
    # The loader injects an iframe pointing to /embed?widget_id=...
    @app.get("/widget.js", include_in_schema=False)
    async def serve_loader() -> FileResponse:
        # In Docker: widget_dist volume mounted at /app/static/widget/
        docker_loader = _WIDGET_STATIC_DIR / "loader.js"
        if docker_loader.exists():
            return FileResponse(docker_loader, media_type="application/javascript")
        # Fallback for local dev: serve the public/loader.js from the frontend dir
        dev_loader = (
            Path(__file__).parent.parent.parent.parent.parent
            / "frontend"
            / "widget"
            / "public"
            / "loader.js"
        )
        return FileResponse(dev_loader, media_type="application/javascript")

    # Serve the React widget bundle (iframe src) at /static/widget/
    if _WIDGET_STATIC_DIR.exists():
        app.mount("/static/widget", StaticFiles(directory=str(_WIDGET_STATIC_DIR)), name="widget_static")

    return app


# Module-level singleton used by uvicorn entrypoint.
app = create_app()
