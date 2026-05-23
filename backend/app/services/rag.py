"""RAG service: query rewrite → hybrid retrieval → ranked results.

Orchestrates multi-query expansion, embedding, hybrid retrieval, and reranking.
Returns chunks ranked by relevance for injection into chatbot context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.errors import ToolFailure
from app.rag.citations import Citation, build_citations
from app.rag.embeddings import get_embedding_model
from app.rag.retrieval import HybridRetriever
from app.rag.rewrite import MultiQueryExpander

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single chunk returned from RAG search.

    ``text`` is the small, focused child used for matching; ``parent_text``
    is the full parent document used to expand context for the LLM. For
    issue sources, ``parent_text`` is None because the child IS the parent.
    """

    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any]
    parent_id: str | None = None
    parent_text: str | None = None


@dataclass
class SearchResults:
    """Top-k chunks ranked by relevance."""

    query: str
    query_variations: list[str]
    chunks: list[SearchResult]
    total_retrieved: int
    citations: list[Citation]


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
        source_types: list[str] | None = None,
        min_citation_confidence: float = 0.30,
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

            # 3. Hybrid retrieval — translate user-facing source labels
            #    ("issue", "wiki") to DB-level source column values
            #    ("issue", "docs"). Pass None to disable filtering.
            source_filter = _translate_source_types(source_types)
            # Lazy-import the reranker singleton so RAGService still works
            # in environments without sentence-transformers (e.g. unit tests).
            from app.rag.reranker import get_reranker

            retrieved_chunks = await self.retriever.retrieve(
                query_variations=query_variations,
                embedding_fn=embedder.embed,
                db_session=db_session,
                reranker=get_reranker(),
                top_k=top_k,
                source_filter=source_filter,
            )

            # 4. Convert to response format. Carry parent fields and the
            #    chunk's own metadata through — citations need it.
            results = [
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=chunk.rerank_score or chunk.score,
                    metadata=dict(chunk.metadata or {"source_type": chunk.source}),
                    parent_id=chunk.parent_id,
                    parent_text=chunk.parent_text,
                )
                for chunk in retrieved_chunks
            ]

            # 5. Build citations server-side (deterministic, not LLM-driven)
            citations = build_citations(
                retrieved_chunks,
                min_confidence=min_citation_confidence,
            )

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
                citations=citations,
            )

        except Exception as exc:
            logger.exception("rag_search_failed", query=query[:100], error=str(exc))
            raise ToolFailure(f"RAG search failed: {str(exc)}", retryable=True) from exc


def _translate_source_types(source_types: list[str] | None) -> list[str] | None:
    """Map user-facing source labels to DB ``source`` column values.

    The DB stores wiki/docs chunks under ``source="docs"``. The API surface
    speaks in user-friendly terms (``"issue"``, ``"wiki"``). This keeps both
    sides happy without renaming a populated column.
    """
    if not source_types:
        return None
    mapping = {"issue": "issue", "wiki": "docs", "docs": "docs"}
    translated = [mapping[s] for s in source_types if s in mapping]
    return translated or None
