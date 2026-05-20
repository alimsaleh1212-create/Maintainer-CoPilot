"""SQLAlchemy ORM model for rag_chunks table."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class RagChunk(Base):
    """RAG chunk from corpus (docs or resolved issues)."""

    __tablename__ = "rag_chunks"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    text: Mapped[str] = mapped_column(Text(), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)  # "docs" | "issue"
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384), nullable=False
    )  # BAAI/bge-small: 384-dim
    tsvector: Mapped[object] = mapped_column(
        TSVECTOR(), nullable=False
    )  # For BM25 sparse search
    metadata: Mapped[dict[str, str]] = mapped_column(JSONB(), default={}, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<RagChunk {self.chunk_id} from {self.source}>"
