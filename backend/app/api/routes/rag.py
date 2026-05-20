"""RAG search endpoint: query corpus for relevant context."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.services.rag import RAGService, SearchResults

router = APIRouter(prefix="/rag", tags=["rag"])


class SearchRequest(BaseModel):
    """Request to search the corpus."""

    query: str = Field(..., description="User question or search text")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class ChunkResponse(BaseModel):
    """A retrieved chunk."""

    chunk_id: str
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    """RAG search response."""

    query: str
    query_variations: list[str]
    chunks: list[ChunkResponse]
    total_retrieved: int


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> SearchResponse:
    """Search corpus for context chunks.

    Takes a user question, expands it into query variations (multi-query),
    retrieves relevant chunks via hybrid search (dense + sparse), and
    returns top-k ranked results.

    Query expansion uses template-based patterns (80% of variations) +
    optional Gemini rewrite for complex queries.

    Hybrid retrieval: 60% dense (pgvector) + 40% sparse (BM25).

    Args:
        request: SearchRequest with query and top_k

    Returns:
        SearchResponse with expanded queries and ranked chunks
    """
    service = RAGService()
    results: SearchResults = await service.search(
        query=request.query,
        db_session=db_session,
        gemini_api_key=None,  # TODO: pass from Vault dependency
        top_k=request.top_k,
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
            )
            for c in results.chunks
        ],
        total_retrieved=results.total_retrieved,
    )
