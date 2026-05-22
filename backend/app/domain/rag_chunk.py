"""RAG chunk domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RagChunkCreate(BaseModel):
    """Input model for creating a RAG chunk."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source: 'docs' or 'issue'")
    parent_id: str = Field(..., description="Parent document identifier")
    parent_text: str | None = Field(
        default=None,
        description="Full parent document text; None when chunk IS the parent",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., issue_id, file_path, labels)",
    )


class RagChunkResponse(BaseModel):
    """RAG chunk as returned from database."""

    id: UUID
    chunk_id: str
    text: str
    source: str
    parent_id: str | None
    parent_text: str | None
    metadata: dict[str, Any]
    created_at: datetime
