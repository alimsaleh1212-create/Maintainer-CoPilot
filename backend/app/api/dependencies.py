"""FastAPI dependency providers.

Every dependency the routes need is declared here. Tests override these
via app.dependency_overrides — never monkey-patch imports.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

import httpx
import structlog
from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.services.chatbot import ChatbotService
from app.services.classification import ClassificationService
from app.services.memory import MemoryService
from app.services.rag import RAGService
from app.services.widget import WidgetService

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


async def get_redis(request: Request) -> Redis:  # type: ignore[type-arg]
    """Return the shared Redis client from app.state."""
    redis: Redis = request.app.state.redis  # type: ignore[type-arg]
    return redis


# ---------------------------------------------------------------------------
# Service providers — one per service, built on demand from app.state
# ---------------------------------------------------------------------------


def get_classification_service(request: Request) -> ClassificationService:
    """Return a ClassificationService backed by the model-server HTTP client.

    The client is built fresh per request (lightweight — no connection pool
    state is held).  The model-server URL comes from Settings.

    Args:
        request: FastAPI request (used to access app.state.settings).

    Returns:
        ClassificationService instance.
    """
    settings: Settings = request.app.state.settings
    client = httpx.AsyncClient(
        base_url=settings.model_server_base_url,
        timeout=10.0,
    )
    return ClassificationService(client=client)


def get_memory_service(request: Request) -> MemoryService:
    """Return a MemoryService with Redis, session factory, and embedder.

    Args:
        request: FastAPI request (used to access app.state singletons).

    Returns:
        MemoryService instance.
    """
    return MemoryService(
        redis=request.app.state.redis,
        session_factory=request.app.state.session_factory,
        embedder=request.app.state.embedder,
    )


def get_widget_service(request: Request) -> WidgetService:
    """Return a WidgetService backed by the database session factory.

    Args:
        request: FastAPI request.

    Returns:
        WidgetService instance.
    """
    return WidgetService(session_factory=request.app.state.session_factory)


def get_rag_service(request: Request) -> RAGService:  # noqa: ARG001
    """Return a RAGService instance.

    Args:
        request: FastAPI request (present for DI consistency; RAGService is
            stateless and needs no app.state singletons currently).

    Returns:
        RAGService instance.
    """
    return RAGService()


def get_chatbot_service(request: Request) -> ChatbotService:
    """Return a ChatbotService initialised with LLM credentials from Vault.

    Args:
        request: FastAPI request (used to access app.state.settings).

    Returns:
        ChatbotService instance.
    """
    settings: Settings = request.app.state.settings
    return ChatbotService(
        gemini_api_key=settings.gemini_api_key,
        ollama_host=settings.ollama_host,
    )


# ---------------------------------------------------------------------------
# Annotated shorthand aliases for route signatures
# ---------------------------------------------------------------------------

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
RedisDep = Annotated[Redis, Depends(get_redis)]  # type: ignore[type-arg]
ClassificationServiceDep = Annotated[ClassificationService, Depends(get_classification_service)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
WidgetServiceDep = Annotated[WidgetService, Depends(get_widget_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
ChatbotServiceDep = Annotated[ChatbotService, Depends(get_chatbot_service)]
