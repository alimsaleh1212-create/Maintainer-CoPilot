"""Memory inspector routes.

Allows authenticated users to inspect and delete their long-term memories.
Every delete writes an AuditLog row (handled by MemoryService).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import MemoryServiceDep
from app.api.routes.auth import get_current_user
from app.repositories.models import Memory, User

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MemoryResponse(BaseModel):
    """A single long-term memory entry."""

    id: uuid.UUID
    user_id: uuid.UUID
    summary: str = Field(..., description="Redacted conversation summary")
    created_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=list[MemoryResponse])
async def list_memories(
    memory_service: MemoryServiceDep,
    current_user: User = Depends(get_current_user),
) -> list[MemoryResponse]:
    """Return all long-term memories for the authenticated user.

    Memories are ordered newest-first.

    Args:
        memory_service: Memory service (Redis + pgvector).
        current_user: Authenticated user.

    Returns:
        List of MemoryResponse objects.
    """
    memories: list[Memory] = await memory_service.list_memories(user_id=current_user.id)
    logger.info(
        "memory.list",
        user_id=str(current_user.id),
        count=len(memories),
    )
    return [MemoryResponse.model_validate(m) for m in memories]


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: uuid.UUID,
    memory_service: MemoryServiceDep,
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a specific long-term memory (must be owned by the caller).

    Writes an AuditLog row recording the deletion.

    Args:
        memory_id: UUID of the memory to delete.
        memory_service: Memory service.
        current_user: Authenticated user (must own the memory).

    Raises:
        NotFoundError: Memory not found or owned by another user.
    """
    await memory_service.delete_memory(
        memory_id=memory_id,
        user_id=current_user.id,
    )
    logger.info(
        "memory.deleted",
        memory_id=str(memory_id),
        user_id=str(current_user.id),
    )
    return None
