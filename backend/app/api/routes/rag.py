"""RAG search endpoint: query corpus for relevant context."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.services.rag import RAGService, SearchResults

router = APIRouter(prefix="/rag", tags=["rag"])

SourceType = Literal["issue", "wiki"]


class SearchRequest(BaseModel):
    """Request to search the corpus."""

    query: str = Field(..., description="User question or search text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    source_types: list[SourceType] | None = Field(
        default=None,
        description=(
            "Restrict retrieval to these source types. "
            "Empty/None = both issues and wiki."
        ),
    )
    min_citation_confidence: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Drop citations whose normalized score is below this threshold.",
    )


class ChunkResponse(BaseModel):
    """A retrieved chunk surfaced to clients."""

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    parent_text: str | None = None


class CitationResponse(BaseModel):
    """A single citation surfaced under a RAG answer."""

    id: int
    source_type: str
    label: str
    url: str | None
    score: float
    chunk_id: str
    snippet: str


class SearchResponse(BaseModel):
    """RAG search response."""

    query: str
    query_variations: list[str]
    chunks: list[ChunkResponse]
    citations: list[CitationResponse]
    total_retrieved: int


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """Search corpus for context chunks.

    Pipeline:
      1. Expand query into variations (multi-query, optional LLM rewrite).
      2. Hybrid retrieve (dense pgvector + sparse BM25) → top-k chunks.
      3. Parent-expand: enrich each chunk with its full parent document.
      4. Build citations server-side (deterministic — never LLM-generated).

    The ``source_types`` filter restricts retrieval at SQL time, so users
    asking "answer only from the wiki" never see issue chunks.
    """
    service = RAGService()
    results: SearchResults = await service.search(
        query=request.query,
        db_session=db_session,
        gemini_api_key=None,  # TODO: pass from Vault dependency
        top_k=request.top_k,
        source_types=request.source_types,
        min_citation_confidence=request.min_citation_confidence,
    )

    return SearchResponse(
        query=results.query,
        query_variations=results.query_variations,
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                text=c.text,
                source=c.source,
                score=c.score,
                metadata=c.metadata,
                parent_id=c.parent_id,
                parent_text=c.parent_text,
            )
            for c in results.chunks
        ],
        citations=[
            CitationResponse(**cite.to_dict()) for cite in results.citations
        ],
        total_retrieved=results.total_retrieved,
    )
