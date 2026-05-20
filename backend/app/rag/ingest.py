"""Corpus ingestion: fetch docs and issues, chunk, embed, store in pgvector."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger(__name__)


class CorpusIngestor:
    """Ingest MONAI docs and issues into pgvector corpus.

    Sources:
    1. MONAI docs (markdown files from repo)
    2. Resolved MONAI issues (NOT in training split)
    """

    def __init__(
        self,
        repo_root: Path | str = "/home/user/workplace/aie_sef_bootcamp/project7",
    ):
        self.repo_root = Path(repo_root)
        self.github_token = None  # From Vault

    async def ingest_docs(
        self,
        docs_source: str = "local",  # "local" or "github"
        chunker=None,
        embedder=None,
        db_session=None,
    ) -> dict[str, int]:
        """Ingest documentation files.

        Args:
            docs_source: Where to pull docs from ("local" or "github")
            chunker: MarkdownChunker instance
            embedder: EmbeddingModel instance
            db_session: SQLAlchemy session to store chunks

        Returns:
            Ingestion stats: {docs_count, chunks_count, errors_count}
        """
        logger.info("corpus_ingest.docs_start", source=docs_source)

        stats = {"docs_count": 0, "chunks_count": 0, "errors_count": 0}

        # TODO: Implement doc fetching
        # 1. If "local": find .md files in MONAI repo
        # 2. If "github": fetch from GitHub raw content
        # 3. For each doc:
        #    - Chunk with MarkdownChunker
        #    - Embed chunks with EmbeddingModel
        #    - Insert into rag_chunks table with pgvector

        logger.info(
            "corpus_ingest.docs_placeholder",
            source=docs_source,
        )
        return stats

    async def ingest_issues(
        self,
        exclude_issue_ids: set[int] | None = None,
        chunker=None,
        embedder=None,
        db_session=None,
    ) -> dict[str, int]:
        """Ingest resolved issues (NOT in training set).

        Args:
            exclude_issue_ids: Issue IDs to skip (training/test splits)
            chunker: MarkdownChunker instance
            embedder: EmbeddingModel instance
            db_session: SQLAlchemy session to store chunks

        Returns:
            Ingestion stats: {issues_count, chunks_count, errors_count}
        """
        logger.info("corpus_ingest.issues_start")

        stats = {"issues_count": 0, "chunks_count": 0, "errors_count": 0}

        if exclude_issue_ids is None:
            exclude_issue_ids = set()

        # TODO: Implement issue ingestion
        # 1. Query MONAI closed issues via GitHub API
        # 2. Filter out exclude_issue_ids
        # 3. For each issue:
        #    - Create document from title + body
        #    - Chunk
        #    - Embed
        #    - Insert with metadata: {issue_id, labels, created_at, source: "issue"}

        logger.info(
            "corpus_ingest.issues_placeholder",
            excluded_count=len(exclude_issue_ids),
        )
        return stats

    async def verify_corpus(self, db_session=None) -> dict[str, int]:
        """Verify corpus is indexed correctly.

        Args:
            db_session: SQLAlchemy session

        Returns:
            Stats: {total_chunks, indexed_chunks, unindexed_chunks}
        """
        # TODO: Query rag_chunks table
        # - Count total rows
        # - Count rows with non-null embedding vectors
        # - Verify pgvector index exists

        logger.info("corpus_ingest.verify_placeholder")
        return {
            "total_chunks": 0,
            "indexed_chunks": 0,
            "unindexed_chunks": 0,
        }
