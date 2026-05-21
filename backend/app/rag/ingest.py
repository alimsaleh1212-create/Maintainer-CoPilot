"""Corpus ingestion: chunk → embed → upsert into pgvector with parent context."""

from __future__ import annotations

import json
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking import Chunk, IssueChunker, MarkdownChunker
from app.rag.embeddings import EmbeddingModel

logger = structlog.get_logger(__name__)

# Batched embed cuts per-item latency from ~1 HTTP roundtrip to ~1/N.
# Ollama's /api/embed handles the batch on its end without recompute overhead.
EMBED_BATCH = 32


class CorpusIngestor:
    """Walks issues + wiki files, produces chunks, embeds, writes rows.

    Each row stores both the (small) child text *and* a reference back to
    the (big) parent — denormalized for one-roundtrip parent-document
    retrieval at query time.
    """

    def __init__(self) -> None:
        self.issue_chunker = IssueChunker()
        self.markdown_chunker = MarkdownChunker()

    # ── Issues ────────────────────────────────────────────────────────────────

    async def ingest_issues(
        self,
        issues: list[dict[str, Any]],
        embedder: EmbeddingModel,
        db_session: AsyncSession,
        exclude_issue_ids: set[int] | None = None,
    ) -> dict[str, int]:
        """Ingest closed issues as single-chunk parents.

        Args:
            issues: Decoded rows from raw_issues.jsonl.
            embedder: Embedding client.
            db_session: SQLAlchemy async session.
            exclude_issue_ids: Issue numbers to skip (e.g., RAG golden set IDs
                to prevent retrieval/eval leakage).

        Returns:
            Stats dict ``{ingested, skipped, errors}``.
        """
        excluded = exclude_issue_ids or set()
        stats = {"ingested": 0, "skipped": 0, "errors": 0}
        logger.info("corpus_ingest.issues_start", total=len(issues), excluded=len(excluded))

        pending: list[Chunk] = []
        for issue in issues:
            number = issue.get("number") or issue.get("id")
            if number in excluded:
                stats["skipped"] += 1
                continue
            chunk = self.issue_chunker.chunk(issue)
            if len(chunk.text) < 50:
                stats["skipped"] += 1
                continue
            pending.append(chunk)

        ok, errs = await self._embed_and_upsert_batched(pending, embedder, db_session)
        stats["ingested"] += ok
        stats["errors"] += errs

        logger.info("corpus_ingest.issues_done", **stats)
        return stats

    # ── Wiki pages ────────────────────────────────────────────────────────────

    async def ingest_wiki(
        self,
        wiki_files: list[tuple[str, str]],  # [(relative_path, content), ...]
        embedder: EmbeddingModel,
        db_session: AsyncSession,
    ) -> dict[str, int]:
        """Ingest wiki markdown files as parent-document chunks.

        Args:
            wiki_files: Pairs of (file_path_within_corpus, raw_markdown).
            embedder: Embedding client.
            db_session: SQLAlchemy async session.

        Returns:
            Stats dict ``{files, chunks, errors}``.
        """
        stats = {"files": 0, "chunks": 0, "errors": 0}
        logger.info("corpus_ingest.wiki_start", files=len(wiki_files))

        pending: list[Chunk] = []
        for file_path, content in wiki_files:
            try:
                chunks = self.markdown_chunker.chunk(content, file_path=file_path)
                pending.extend(chunks)
                stats["files"] += 1
            except Exception as exc:  # noqa: BLE001
                logger.exception("corpus_ingest.wiki_failed", file=file_path, error=str(exc))
                stats["errors"] += 1

        ok, errs = await self._embed_and_upsert_batched(pending, embedder, db_session)
        stats["chunks"] += ok
        stats["errors"] += errs

        logger.info("corpus_ingest.wiki_done", **stats)
        return stats

    # ── Embed + upsert in batches ────────────────────────────────────────────

    async def _embed_and_upsert_batched(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingModel,
        db_session: AsyncSession,
    ) -> tuple[int, int]:
        """Embed in batches of EMBED_BATCH, then upsert each row.

        Returns (ingested_count, error_count).
        """
        ingested = 0
        errors = 0
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            try:
                embeddings = await embedder.embed_batch([c.text for c in batch])
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "corpus_ingest.embed_batch_failed",
                    batch_index=i // EMBED_BATCH,
                    batch_size=len(batch),
                    error=str(exc),
                )
                errors += len(batch)
                continue
            for chunk, embedding in zip(batch, embeddings, strict=True):
                try:
                    await self._upsert(db_session, chunk, embedding)
                    ingested += 1
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "corpus_ingest.upsert_failed",
                        chunk_id=chunk.chunk_id,
                        error=str(exc),
                    )
                    errors += 1
        return ingested, errors

    # ── DB write ─────────────────────────────────────────────────────────────

    async def _upsert(
        self,
        db_session: AsyncSession,
        chunk: Chunk,
        embedding: list[float],
    ) -> None:
        """Idempotent insert (ON CONFLICT chunk_id DO UPDATE)."""
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        metadata_json = json.dumps(_serialize_metadata(chunk.metadata))
        # tsvector uses the chunk's own text — search hits the same content
        # the LLM will eventually see.
        stmt = text(
            "INSERT INTO rag_chunks "
            "(chunk_id, text, source, embedding, tsvector, metadata, parent_id, parent_text) "
            "VALUES (:chunk_id, :text, :source, CAST(:embedding AS vector), "
            "to_tsvector('english', :tsvector_text), CAST(:metadata AS jsonb), "
            ":parent_id, :parent_text) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "  text = EXCLUDED.text, "
            "  embedding = EXCLUDED.embedding, "
            "  tsvector = EXCLUDED.tsvector, "
            "  metadata = EXCLUDED.metadata, "
            "  parent_id = EXCLUDED.parent_id, "
            "  parent_text = EXCLUDED.parent_text"
        ).bindparams(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            embedding=embedding_str,
            tsvector_text=chunk.text,
            metadata=metadata_json,
            parent_id=chunk.parent_id,
            parent_text=chunk.parent_text,
        )
        await db_session.execute(stmt)
        await db_session.commit()


def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop None values and coerce non-JSON types to strings."""
    out: dict[str, Any] = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool, list, dict)):
            out[k] = v
        else:
            out[k] = str(v)
    return out
