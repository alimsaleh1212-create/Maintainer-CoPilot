"""RAG chunk domain model."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RagChunkCreate(BaseModel):
    """Input model for creating a RAG chunk."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    text: str = Field(..., description="Chunk text content")
    source: str = Field(..., description="Source: 'docs' or 'issue'")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., issue_id, file_path)",
    )


class RagChunkResponse(BaseModel):
    """RAG chunk as returned from database."""

    id: UUID
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, str]
    created_at: datetime
