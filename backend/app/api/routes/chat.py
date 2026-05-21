"""Chat endpoint: POST /chat

Delegates to ChatbotService which runs the tool-calling LLM loop.
Requires a valid JWT (Authorization: Bearer <token>).
"""

from __future__ import annotations

import uuid
from typing import Literal

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import (
    ChatbotServiceDep,
    ClassificationServiceDep,
    DbSessionDep,
    MemoryServiceDep,
    RAGServiceDep,
    SettingsDep,
)
from app.api.routes.auth import get_current_user
from app.repositories.models import User

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


SourceType = Literal["issue", "wiki"]


class ChatRequest(BaseModel):
    """Incoming chat message."""

    message: str = Field(..., min_length=1, max_length=8192, description="User message")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID; omit to start a new conversation",
    )
    rag_source_types: list[SourceType] | None = Field(
        default=None,
        description=(
            "If the LLM calls rag_search, restrict it to these sources. "
            "Empty/None = both issues and wiki."
        ),
    )
    rag_min_confidence: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Drop citations whose normalized score is below this threshold.",
    )


class CitationOut(BaseModel):
    """A single citation surfaced under a chat answer."""

    id: int
    source_type: str
    label: str
    url: str | None
    score: float
    chunk_id: str
    snippet: str


class ChatResponse(BaseModel):
    """Chat turn response."""

    response: str = Field(..., description="Assistant response text")
    conversation_id: str = Field(..., description="Conversation ID (use in subsequent turns)")
    tools_used: list[str] = Field(default_factory=list, description="Tool names called this turn")
    citations: list[CitationOut] = Field(
        default_factory=list,
        description="Citations from rag_search calls this turn, deduplicated by chunk_id",
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    settings: SettingsDep,
    db: DbSessionDep,
    chatbot_service: ChatbotServiceDep,
    memory_service: MemoryServiceDep,
    classification_service: ClassificationServiceDep,
    rag_service: RAGServiceDep,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """Run one user turn through the tool-calling chatbot loop.

    Creates a new conversation ID if one is not provided.  The conversation
    history is maintained in Redis (TTL 24 h) keyed by ``conversation_id``.

    Args:
        body: User message and optional conversation ID.
        settings: Application settings.
        db: Database session (passed to RAGService for retrieval).
        chatbot_service: Tool-calling chatbot service.
        memory_service: Redis + pgvector memory service.
        classification_service: DistilBERT classification service.
        rag_service: Hybrid retrieval service.
        current_user: Authenticated user (from JWT).

    Returns:
        ChatResponse with assistant text, conversation ID, and tools used.
    """
    conversation_id = body.conversation_id or str(uuid.uuid4())

    logger.info(
        "chat.request",
        user_id=str(current_user.id),
        conversation_id=conversation_id,
        message_length=len(body.message),
    )

    response_text, tools_used, citations = await chatbot_service.chat(
        user_message=body.message,
        conversation_id=conversation_id,
        user_id=current_user.id,
        memory_service=memory_service,
        classification_service=classification_service,
        rag_service=rag_service,
        db_session=db,
        top_k_memories=settings.long_term_memory_top_k,
        rag_source_types=body.rag_source_types,
        rag_min_confidence=body.rag_min_confidence,
    )

    return ChatResponse(
        response=response_text,
        conversation_id=conversation_id,
        tools_used=tools_used,
        citations=[CitationOut(**c) for c in citations],
    )
