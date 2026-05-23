"""Hybrid retrieval: dense + sparse + reranking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

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
    parent_id: str | None = None
    parent_text: str | None = None
    metadata: dict[str, Any] | None = None


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
        embedding_fn: Callable[[str], Any],
        db_session: Any,
        reranker: Any | None = None,
        top_k: int | None = None,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using hybrid approach.

        Args:
            query_variations: List of query phrasings from multi-query expander
            embedding_fn: Async function to embed text
            db_session: SQLAlchemy async session
            reranker: Optional cross-encoder for reranking
            top_k: Number of final results to return (defaults to self.top_k_final)
            source_filter: Restrict retrieval to these ``rag_chunks.source`` values
                (e.g. ``["issue"]`` or ``["docs"]``). ``None`` = no filter.

        Returns:
            Top-k retrieved chunks, scored and deduplicated
        """
        if top_k is None:
            top_k = self.top_k_final
        all_results = {}  # chunk_id → RetrievedChunk (for dedup)

        # For each query variation, retrieve
        for query_var in query_variations:
            logger.info(
                "retrieval.query_variation",
                query_original=query_variations[0],
                query_var=query_var,
                source_filter=source_filter,
            )

            # Dense search (pgvector)
            dense_results = await self._dense_search(
                query_var, embedding_fn, db_session, source_filter=source_filter
            )

            # Sparse search (BM25)
            sparse_results = await self._sparse_search(
                query_var, db_session, source_filter=source_filter
            )

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
            ranked = await self._rerank(query_variations[0], ranked, reranker)

        final = ranked[:top_k]

        # Parent-expand: one extra round trip enriches each child with its
        # full parent document so the LLM gets surrounding context.
        await self._expand_parents(final, db_session)

        return final

    async def _dense_search(
        self,
        query: str,
        embedding_fn: Callable[[str], Any],
        db_session: AsyncSession,
        top_k: int = 50,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense search using pgvector similarity.

        Uses cosine distance (<=> operator) to find semantically similar chunks.
        Cosine distance ranges [0, 2], so we convert to similarity [0, 1]:
        similarity = 1 - distance

        Args:
            query: Query text
            embedding_fn: Function to embed query
            db_session: DB session
            top_k: Number of results

        Returns:
            List of chunks with dense scores (similarity 0-1)
        """
        # Embed query
        query_embedding = await embedding_fn(query)

        # Query pgvector: find nearest neighbors by cosine distance.
        # Use CAST(...AS vector) not ::vector — the :: shorthand confuses
        # SQLAlchemy's text() parameter parser when directly adjacent to the
        # parameter placeholder (e.g. :name::type is mis-tokenised).
        embedding_str = str(query_embedding)
        where_clause = ""
        params: dict[str, Any] = {"query_embedding": embedding_str, "top_k": top_k}
        if source_filter:
            where_clause = "WHERE source = ANY(:sources) "
            params["sources"] = source_filter
        # ``where_clause`` is a static literal chosen by an internal branch
        # ("" or "WHERE source = ANY(:sources) "); no user input interpolated.
        select_clause = "SELECT id, chunk_id, text, source, "
        score_clause = "1 - (embedding <=> CAST(:query_embedding AS vector)) as score "
        order_limit = "ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT :top_k"
        sql = f"{select_clause} {score_clause} FROM rag_chunks {where_clause}{order_limit}"  # noqa: S608
        stmt = text(sql).bindparams(**params)

        results = await db_session.execute(stmt)
        rows = results.fetchall()

        logger.info(
            "retrieval.dense_search",
            query=query[:100],
            returned=len(rows),
            top_k=top_k,
        )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                text=row.text,
                source=row.source,
                score=float(row.score),
                dense_score=float(row.score),
                sparse_score=None,
            )
            for row in rows
        ]

    async def _sparse_search(
        self,
        query: str,
        db_session: AsyncSession,
        top_k: int = 50,
        source_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Sparse search using BM25-like ranking (Postgres tsvector + ts_rank_cd).

        Converts the query to a tsvector-compatible format and uses ts_rank_cd
        for BM25-style relevance ranking. ts_rank_cd normalizes scores to [0, 1].

        Args:
            query: Query text
            db_session: DB session
            top_k: Number of results

        Returns:
            List of chunks with sparse scores (BM25-like, 0-1)
        """
        # Convert query to Postgres tsquery format (space-separated words become OR clauses)
        # to_tsquery returns a tsquery; plainto_tsquery makes it safer for user input
        where_extra = ""
        params: dict[str, Any] = {"query": query, "top_k": top_k}
        if source_filter:
            where_extra = "AND source = ANY(:sources) "
            params["sources"] = source_filter
        # ``where_extra`` is a static literal from an internal branch
        # ("" or "AND source = ANY(:sources) "); no user input interpolated.
        select_clause = "SELECT id, chunk_id, text, source, "
        score_clause = "ts_rank_cd(tsvector, plainto_tsquery('english', :query)) as score "
        where_base = "WHERE tsvector @@ plainto_tsquery('english', :query) "
        order_limit = "ORDER BY score DESC LIMIT :top_k"
        sql = (
            f"{select_clause} {score_clause} FROM rag_chunks {where_base}{where_extra}{order_limit}"  # noqa: S608
        )
        stmt = text(sql).bindparams(**params)

        results = await db_session.execute(stmt)
        rows = results.fetchall()

        logger.info(
            "retrieval.sparse_search",
            query=query[:100],
            returned=len(rows),
            top_k=top_k,
        )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                text=row.text,
                source=row.source,
                score=float(row.score),
                dense_score=None,
                sparse_score=float(row.score),
            )
            for row in rows
        ]

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

    async def _expand_parents(
        self,
        chunks: list[RetrievedChunk],
        db_session: AsyncSession,
    ) -> None:
        """Fill ``parent_id`` / ``parent_text`` on each chunk in place.

        Does a single ``SELECT chunk_id, parent_id, parent_text ...``
        keyed by chunk_id. Wiki chunks get a non-null parent_text; issue
        chunks get parent_id == chunk_id and parent_text remains None
        (the child already is the parent).
        """
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        stmt = text(
            "SELECT chunk_id, parent_id, parent_text, metadata "
            "FROM rag_chunks WHERE chunk_id = ANY(:ids)"
        ).bindparams(ids=ids)
        rows = (await db_session.execute(stmt)).fetchall()
        by_id = {row.chunk_id: row for row in rows}
        for c in chunks:
            row = by_id.get(c.chunk_id)
            if row is None:
                continue
            c.parent_id = row.parent_id
            c.parent_text = row.parent_text
            # metadata column is JSONB → driver returns dict
            c.metadata = row.metadata if isinstance(row.metadata, dict) else None

    async def _rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        reranker: Any,
    ) -> list[RetrievedChunk]:
        """Rerank candidates using a cross-encoder.

        Calls ``reranker.rerank(query, passages)`` which returns
        ``[(index, score), ...]`` sorted by score desc. We then attach the
        rerank score to each chunk and resort the list. The cross-encoder
        score replaces the hybrid score as the final ranking signal because
        BGE-reranker is far more accurate at semantic relevance than the
        dense+sparse linear combination.

        Args:
            query: Original user query (not the rewritten variations).
            candidates: Hybrid top-N to rerank.
            reranker: Object exposing async ``rerank(query, list[str]) -> list[(int, float)]``.

        Returns:
            Candidates resorted by cross-encoder score (descending).
        """
        if not candidates:
            return candidates

        logger.info("retrieval.reranking", num_candidates=len(candidates))
        passages = [c.text for c in candidates]
        ranked = await reranker.rerank(query, passages)

        for idx, score in ranked:
            candidates[idx].rerank_score = score
            candidates[idx].score = score
        candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        logger.info(
            "retrieval.reranking_complete",
            top_rerank_score=candidates[0].rerank_score if candidates else None,
        )
        return candidates
