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
    WidgetServiceDep,
)
from app.api.routes.auth import get_current_user
from app.domain.errors import NotFoundError, PermissionDenied
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
    widget_id: str | None = Field(
        default=None,
        description=(
            "Public widget ID (wgt_*) — when present, restricts the LLM to the "
            "tools enabled on that widget. Omit when calling from the Streamlit UI."
        ),
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
    widget_service: WidgetServiceDep,
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

    # If the request came from an embedded widget, look up the widget's
    # enabled_tools list and restrict the LLM accordingly.  When no widget_id
    # is supplied (e.g. Streamlit admin chat), the chatbot gets all tools.
    enabled_tools: list[str] | None = None
    if body.widget_id:
        widget = await widget_service.get_widget(body.widget_id)
        if widget is None:
            raise NotFoundError(f"Widget {body.widget_id} not found")
        if not widget.enabled:
            raise PermissionDenied(f"Widget {body.widget_id} is disabled")
        enabled_tools = list(widget.enabled_tools)

    # Verify the caller owns this conversation (multi-tenant isolation).
    # Returns False if the conversation has a different owner.
    if body.conversation_id:
        owner_ok = await memory_service.verify_conversation_owner(
            conversation_id=body.conversation_id,
            user_id=current_user.id,
        )
        if not owner_ok:
            raise PermissionDenied("Conversation belongs to a different user")

    logger.info(
        "chat.request",
        user_id=str(current_user.id),
        conversation_id=conversation_id,
        widget_id=body.widget_id,
        enabled_tools=enabled_tools,
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
        enabled_tools=enabled_tools,
    )

    # Persist conversation metadata so this user can list/resume it later.
    await memory_service.touch_conversation(
        conversation_id=conversation_id,
        user_id=current_user.id,
        first_message=body.message if not body.conversation_id else None,
    )

    return ChatResponse(
        response=response_text,
        conversation_id=conversation_id,
        tools_used=tools_used,
        citations=[CitationOut(**c) for c in citations],
    )


# ---------------------------------------------------------------------------
# Conversation listing / history (scoped to the authenticated user)
# ---------------------------------------------------------------------------


class ConversationSummary(BaseModel):
    """One row in the user's conversation list."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    """One message in a conversation history."""

    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    title: str
    messages: list[ConversationMessage]


@router.get("/chat/conversations", response_model=list[ConversationSummary])
async def list_my_conversations(
    memory_service: MemoryServiceDep,
    current_user: User = Depends(get_current_user),
) -> list[ConversationSummary]:
    """List the authenticated user's conversations, newest first."""
    convs = await memory_service.list_user_conversations(current_user.id)
    return [ConversationSummary(**c) for c in convs]


@router.get(
    "/chat/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
)
async def get_my_conversation(
    conversation_id: str,
    memory_service: MemoryServiceDep,
    current_user: User = Depends(get_current_user),
) -> ConversationHistoryResponse:
    """Return the history for a conversation owned by the authenticated user.

    Returns 403 if the conversation belongs to a different user, 404 if it
    does not exist.
    """
    owner_ok = await memory_service.verify_conversation_owner(
        conversation_id=conversation_id, user_id=current_user.id
    )
    if not owner_ok:
        raise PermissionDenied("Conversation belongs to a different user")

    history = await memory_service.get_history(conversation_id)
    if not history:
        # No meta + no history = doesn't exist for this user
        all_convs = await memory_service.list_user_conversations(current_user.id)
        if not any(c["conversation_id"] == conversation_id for c in all_convs):
            raise NotFoundError(f"Conversation {conversation_id} not found")

    convs = await memory_service.list_user_conversations(current_user.id)
    title = next(
        (c["title"] for c in convs if c["conversation_id"] == conversation_id),
        "Conversation",
    )
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        title=title,
        messages=[
            ConversationMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in history
        ],
    )
