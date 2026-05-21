"""Corpus ingestion: fetch docs and issues, chunk, embed, store in pgvector."""

from __future__ import annotations

import hashlib
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.chunking import MarkdownChunker
from app.rag.embeddings import EmbeddingModel

logger = structlog.get_logger(__name__)


class CorpusIngestor:
    """Ingest MONAI docs and issues into pgvector corpus.

    Sources:
    1. Static markdown files (documentation)
    2. GitHub issues (resolved, from outside training set)
    """

    def __init__(self) -> None:
        self.chunker = MarkdownChunker()

    async def ingest_docs(
        self,
        docs: list[tuple[str, str]],  # [(title, content), ...]
        embedder: EmbeddingModel,
        db_session: AsyncSession,
    ) -> dict[str, int]:
        """Ingest documentation files.

        Args:
            docs: List of (title, content) tuples
            embedder: EmbeddingModel instance
            db_session: SQLAlchemy async session

        Returns:
            Ingestion stats: {docs_count, chunks_count, errors_count}
        """
        logger.info("corpus_ingest.docs_start", docs_count=len(docs))

        stats = {"docs_count": 0, "chunks_count": 0, "errors_count": 0}

        for title, content in docs:
            try:
                # Chunk the document
                chunks = self.chunker.chunk(content, source=title)

                for chunk in chunks:
                    try:
                        # Embed the chunk
                        embedding = await embedder.embed(chunk.text)

                        # Create tsvector content
                        tsvector_text = f"{chunk.text} {title}"

                        # Insert into database
                        chunk_id = self._generate_chunk_id(title, chunk.text)
                        doc_metadata: dict[str, Any] = {"title": title, **chunk.metadata}
                        await self._insert_chunk(
                            db_session=db_session,
                            chunk_id=chunk_id,
                            content=chunk.text,
                            source="docs",
                            embedding=embedding,
                            tsvector_text=tsvector_text,
                            metadata=doc_metadata,
                        )

                        stats["chunks_count"] += 1

                    except Exception as exc:
                        logger.exception(
                            "corpus_ingest.chunk_failed",
                            title=title,
                            error=str(exc),
                        )
                        stats["errors_count"] += 1

                stats["docs_count"] += 1

            except Exception as exc:
                logger.exception(
                    "corpus_ingest.doc_failed",
                    title=title,
                    error=str(exc),
                )
                stats["errors_count"] += 1

        logger.info("corpus_ingest.docs_complete", **stats)
        return stats

    async def ingest_issues(
        self,
        issues: list[dict[str, Any]],  # [{id, title, body, labels}, ...]
        embedder: EmbeddingModel,
        db_session: AsyncSession,
        exclude_issue_ids: set[int] | None = None,
    ) -> dict[str, int]:
        """Ingest resolved issues (NOT in training set).

        Args:
            issues: List of issue dicts
            embedder: EmbeddingModel instance
            db_session: SQLAlchemy async session
            exclude_issue_ids: Issue IDs to skip (training/test splits)

        Returns:
            Ingestion stats: {issues_count, chunks_count, errors_count}
        """
        logger.info("corpus_ingest.issues_start", total=len(issues))

        stats = {"issues_count": 0, "chunks_count": 0, "errors_count": 0}

        if exclude_issue_ids is None:
            exclude_issue_ids = set()

        for issue in issues:
            issue_id = issue.get("id")

            if issue_id in exclude_issue_ids:
                continue

            try:
                # Create document from issue title + body
                title = issue.get("title", "")
                body = issue.get("body", "")
                document = f"# {title}\n\n{body}"

                # Chunk the issue
                chunks = self.chunker.chunk(document, source=f"issue_{issue_id}")

                for chunk in chunks:
                    try:
                        # Embed the chunk
                        embedding = await embedder.embed(chunk.text)

                        # Create tsvector content
                        tsvector_text = f"{title} {chunk.text}"

                        # Create record with issue metadata
                        chunk_id = self._generate_chunk_id(f"issue_{issue_id}", chunk.text)
                        metadata = {
                            "issue_id": str(issue_id),
                            "labels": ",".join(issue.get("labels", [])),
                            **chunk.metadata,
                        }

                        await self._insert_chunk(
                            db_session=db_session,
                            chunk_id=chunk_id,
                            content=chunk.text,
                            source="issue",
                            embedding=embedding,
                            tsvector_text=tsvector_text,
                            metadata=metadata,
                        )

                        stats["chunks_count"] += 1

                    except Exception as exc:
                        logger.exception(
                            "corpus_ingest.issue_chunk_failed",
                            issue_id=issue_id,
                            error=str(exc),
                        )
                        stats["errors_count"] += 1

                stats["issues_count"] += 1

            except Exception as exc:
                logger.exception(
                    "corpus_ingest.issue_failed",
                    issue_id=issue_id,
                    error=str(exc),
                )
                stats["errors_count"] += 1

        logger.info("corpus_ingest.issues_complete", **stats)
        return stats

    async def verify_corpus(self, db_session: AsyncSession) -> dict[str, int]:
        """Verify corpus is indexed correctly.

        Args:
            db_session: SQLAlchemy async session

        Returns:
            Stats: {total_chunks, indexed_chunks, unindexed_chunks}
        """
        # Count total chunks
        total_result = await db_session.execute(text("SELECT COUNT(*) FROM rag_chunks"))
        total = total_result.scalar() or 0

        # Count chunks with embeddings
        indexed_result = await db_session.execute(
            text("SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL")
        )
        indexed = indexed_result.scalar() or 0

        stats = {
            "total_chunks": int(total),
            "indexed_chunks": int(indexed),
            "unindexed_chunks": int(total) - int(indexed),
        }

        logger.info("corpus_ingest.verify_complete", **stats)
        return stats

    async def _insert_chunk(
        self,
        db_session: AsyncSession,
        chunk_id: str,
        content: str,
        source: str,
        embedding: list[float],
        tsvector_text: str,
        metadata: dict[str, str],
    ) -> None:
        """Insert a single chunk into the database.

        Args:
            db_session: SQLAlchemy async session
            chunk_id: Unique chunk identifier
            content: Chunk text content
            source: "docs" or "issue"
            embedding: Vector embedding
            tsvector_text: Text for tsvector indexing
            metadata: Additional metadata
        """
        import json as _json

        # pgvector expects "[f1, f2, ...]" format; use CAST() not :: so SQLAlchemy
        # can parse the named parameters correctly.
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        stmt = text(
            "INSERT INTO rag_chunks (chunk_id, text, source, embedding, tsvector, metadata) "
            "VALUES (:chunk_id, :content, :source, CAST(:embedding AS vector), "
            "to_tsvector('english', :tsvector_text), CAST(:metadata AS jsonb)) "
            "ON CONFLICT (chunk_id) DO NOTHING"
        ).bindparams(
            chunk_id=chunk_id,
            content=content,
            source=source,
            embedding=embedding_str,
            tsvector_text=tsvector_text,
            metadata=_json.dumps(metadata),
        )

        await db_session.execute(stmt)
        await db_session.commit()

    @staticmethod
    def _generate_chunk_id(source: str, text: str) -> str:
        """Generate a unique chunk ID from source and text hash."""
        hash_str = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"{source}_{hash_str}"
