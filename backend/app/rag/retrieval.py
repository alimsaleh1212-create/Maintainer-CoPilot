"""Hybrid retrieval: dense + sparse + reranking."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """Retrieved chunk with score and metadata."""

    chunk_id: str
    text: str
    source: str
    score: float  # Combined score after reranking
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None


class HybridRetriever:
    """Hybrid retrieval combining dense + sparse search.

    Strategy:
    1. Expand query into multiple variations (multi-query)
    2. For each variation:
       a. Dense search (pgvector) → top-50
       b. Sparse search (BM25 via Postgres tsvector) → top-50
       c. Combine scores: 0.6 * dense + 0.4 * sparse
    3. Deduplicate and rerank with cross-encoder
    4. Return top-5
    """

    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        top_k_before_rerank: int = 20,
        top_k_final: int = 5,
    ):
        """Initialize hybrid retriever.

        Args:
            dense_weight: Weight for dense (pgvector) scores
            sparse_weight: Weight for sparse (BM25) scores
            top_k_before_rerank: Candidates to rerank
            top_k_final: Final results returned
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k_before_rerank = top_k_before_rerank
        self.top_k_final = top_k_final

    async def retrieve(
        self,
        query_variations: list[str],
        embedding_fn,
        db_session,
        reranker=None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using hybrid approach.

        Args:
            query_variations: List of query phrasings from multi-query expander
            embedding_fn: Async function to embed text
            db_session: SQLAlchemy async session
            reranker: Optional cross-encoder for reranking

        Returns:
            Top-k retrieved chunks, scored and deduplicated
        """
        all_results = {}  # chunk_id → RetrievedChunk (for dedup)

        # For each query variation, retrieve
        for query_var in query_variations:
            logger.info(
                "retrieval.query_variation",
                query_original=query_variations[0],
                query_var=query_var,
            )

            # Dense search (pgvector)
            dense_results = await self._dense_search(
                query_var, embedding_fn, db_session
            )

            # Sparse search (BM25)
            sparse_results = await self._sparse_search(query_var, db_session)

            # Combine scores
            combined = self._combine_scores(dense_results, sparse_results)

            # Add/merge to all_results
            for chunk in combined:
                if chunk.chunk_id not in all_results:
                    all_results[chunk.chunk_id] = chunk
                else:
                    # Merge scores (take max)
                    existing = all_results[chunk.chunk_id]
                    existing.score = max(existing.score, chunk.score)

        # Sort and keep top-k before rerank
        ranked = sorted(all_results.values(), key=lambda c: c.score, reverse=True)[
            : self.top_k_before_rerank
        ]

        # Rerank if available
        if reranker:
            ranked = await self._rerank(
                query_variations[0], ranked, reranker
            )

        # Return top-k final
        return ranked[: self.top_k_final]

    async def _dense_search(
        self,
        query: str,
        embedding_fn,
        db_session,
        top_k: int = 50,
    ) -> list[RetrievedChunk]:
        """Dense search using pgvector similarity.

        Args:
            query: Query text
            embedding_fn: Function to embed query
            db_session: DB session
            top_k: Number of results

        Returns:
            List of chunks with dense scores
        """
        # Embed query
        query_embedding = await embedding_fn(query)

        # TODO: Query pgvector
        # This requires RAG corpus to be indexed first (WED step 1)
        # Placeholder SQL:
        # SELECT chunk_id, text, source,
        #        (1 - (embedding <=> query_embedding)) as score
        # FROM rag_chunks
        # ORDER BY embedding <=> query_embedding
        # LIMIT {top_k}

        logger.info("retrieval.dense_search_placeholder", query=query)
        return []  # Placeholder

    async def _sparse_search(
        self,
        query: str,
        db_session,
        top_k: int = 50,
    ) -> list[RetrievedChunk]:
        """Sparse search using BM25 (Postgres tsvector).

        Args:
            query: Query text
            db_session: DB session
            top_k: Number of results

        Returns:
            List of chunks with sparse scores
        """
        # TODO: Query Postgres tsvector
        # SELECT chunk_id, text, source,
        #        ts_rank_cd(tsvector, tsquery) as score
        # FROM rag_chunks
        # WHERE tsvector @@ tsquery
        # ORDER BY score DESC
        # LIMIT {top_k}

        logger.info("retrieval.sparse_search_placeholder", query=query)
        return []  # Placeholder

    def _combine_scores(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Combine dense and sparse scores.

        Args:
            dense_results: Results from pgvector
            sparse_results: Results from BM25

        Returns:
            Combined results with hybrid score
        """
        # Normalize scores to [0, 1]
        max_dense = max([r.score for r in dense_results]) if dense_results else 1.0
        max_sparse = max([r.score for r in sparse_results]) if sparse_results else 1.0

        combined = {}

        for result in dense_results:
            norm_score = result.score / max_dense if max_dense > 0 else 0
            combined[result.chunk_id] = RetrievedChunk(
                chunk_id=result.chunk_id,
                text=result.text,
                source=result.source,
                score=self.dense_weight * norm_score,
                dense_score=norm_score,
            )

        for result in sparse_results:
            norm_score = result.score / max_sparse if max_sparse > 0 else 0
            if result.chunk_id in combined:
                # Add sparse component
                combined[result.chunk_id].score += self.sparse_weight * norm_score
                combined[result.chunk_id].sparse_score = norm_score
            else:
                combined[result.chunk_id] = RetrievedChunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    source=result.source,
                    score=self.sparse_weight * norm_score,
                    sparse_score=norm_score,
                )

        return list(combined.values())

    async def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        reranker,
    ) -> list[RetrievedChunk]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Original query
            candidates: Chunks to rerank
            reranker: Cross-encoder model

        Returns:
            Reranked chunks
        """
        if not candidates:
            return candidates

        logger.info("retrieval.reranking", num_candidates=len(candidates))

        # TODO: Use BAAI/bge-reranker-base
        # scores = reranker.rank(query, [c.text for c in candidates])
        # Update candidates with rerank_score and resort

        logger.info("retrieval.reranking_placeholder")
        return candidates  # Placeholder: return as-is
