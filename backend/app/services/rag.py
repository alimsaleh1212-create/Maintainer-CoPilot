"""RAG service: query rewrite → hybrid retrieval → ranked results.

Orchestrates multi-query expansion, embedding, hybrid retrieval, and reranking.
Returns chunks ranked by relevance for injection into chatbot context.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.errors import ToolFailure
from app.rag.embeddings import get_embedding_model
from app.rag.retrieval import HybridRetriever
from app.rag.rewrite import MultiQueryExpander

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single chunk returned from RAG search."""

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, str]


@dataclass
class SearchResults:
    """Top-k chunks ranked by relevance."""

    query: str
    query_variations: list[str]
    chunks: list[SearchResult]
    total_retrieved: int


class RAGService:
    """Query → retrieval orchestrator.

    Flow:
    1. Expand query into 3-5 variations (template + LLM fallback)
    2. Embed variations
    3. For each variation: hybrid dense + sparse retrieval
    4. Deduplicate + rerank → top-k final results
    """

    def __init__(self) -> None:
        self.expander = MultiQueryExpander()
        self.retriever = HybridRetriever()

    async def search(
        self,
        query: str,
        db_session: AsyncSession,
        gemini_api_key: str | None = None,
        top_k: int = 5,
    ) -> SearchResults:
        """Search corpus for relevant chunks.

        Args:
            query: User question or search text.
            db_session: SQLAlchemy session for DB access (retrieval).
            gemini_api_key: Vault-provided key for LLM-based query rewrite fallback.
            top_k: Number of final chunks to return.

        Returns:
            SearchResults with ranked chunks + metadata.

        Raises:
            ToolFailure if expansion or retrieval fails.
        """
        try:
            # 1. Expand query
            logger.info("rag_search_start", query=query[:100])
            query_variations = await self.expander.expand(query, gemini_api_key)
            logger.info("rag_query_expanded", variation_count=len(query_variations))

            # 2. Get embedding model
            embedder = get_embedding_model()

            # 3. Hybrid retrieval
            retrieved_chunks = await self.retriever.retrieve(
                query_variations=query_variations,
                embedding_fn=embedder.embed,
                db_session=db_session,
                reranker=None,  # TODO: wire reranker when docker ready
                top_k=top_k,
            )

            # 4. Convert to response format
            results = [
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=chunk.rerank_score or chunk.score,
                    metadata={"source_type": chunk.source},
                )
                for chunk in retrieved_chunks
            ]

            logger.info(
                "rag_search_complete",
                variations=len(query_variations),
                total_retrieved=len(retrieved_chunks),
                top_k=len(results),
            )

            return SearchResults(
                query=query,
                query_variations=query_variations,
                chunks=results,
                total_retrieved=len(retrieved_chunks),
            )

        except Exception as exc:
            logger.exception("rag_search_failed", query=query[:100], error=str(exc))
            raise ToolFailure(f"RAG search failed: {str(exc)}", retryable=True) from exc
