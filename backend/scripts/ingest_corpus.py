"""One-shot script: ingest MONAI issues into the RAG corpus.

Run inside the api container:
    docker exec docker-api-1 python scripts/ingest_corpus.py

Or locally with DATABASE_URL and OLLAMA_HOST set:
    uv run python scripts/ingest_corpus.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def main() -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.rag.embeddings import EmbeddingModel
    from app.rag.ingest import CorpusIngestor

    # ── Database URL ─────────────────────────────────────────────────────────
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://copilot:copilot@localhost:5432/copilot",
    )
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    print(f"Connecting to database: {db_url[:40]}...")
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # ── Embedding model ───────────────────────────────────────────────────────
    print(f"Initializing embedding model via Ollama at {ollama_host}...")
    embedder = EmbeddingModel(model_name="nomic-embed-text", ollama_host=ollama_host)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Find raw issues — try inside container path first, then relative
    candidates = [
        Path("/ml/data/raw_issues.jsonl"),
        Path(__file__).resolve().parent / "raw_issues.jsonl",
        Path(__file__).resolve().parent.parent.parent.parent / "ml" / "data" / "raw_issues.jsonl",
    ]
    issues_path: Path | None = None
    for c in candidates:
        if c.exists():
            issues_path = c
            break

    if issues_path is None:
        print("ERROR: raw_issues.jsonl not found. Checked:")
        for c in candidates:
            print(f"  {c}")
        sys.exit(1)

    print(f"Loading issues from {issues_path}...")
    issues: list[dict] = []
    with issues_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))

    print(f"Loaded {len(issues)} issues")

    # ── Load training split to exclude those IDs from RAG corpus ──────────────
    train_path_candidates = [
        Path("/ml/data/train.jsonl"),
        Path(__file__).resolve().parent / "train.jsonl",
        issues_path.parent / "train.jsonl",
    ]
    train_ids: set[int] = set()
    for tp in train_path_candidates:
        if tp.exists():
            with tp.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        if "id" in d:
                            train_ids.add(int(d["id"]))
            print(f"Loaded {len(train_ids)} training IDs to exclude from corpus")
            break

    # ── Run ingestion ─────────────────────────────────────────────────────────
    ingestor = CorpusIngestor()

    async with async_session() as session:
        print("Starting issue ingestion...")
        stats = await ingestor.ingest_issues(
            issues=issues,
            embedder=embedder,
            db_session=session,
            exclude_issue_ids=train_ids,
        )
        print(f"Issue ingestion complete: {stats}")

    await engine.dispose()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
