"""One-shot corpus ingestion.

Walks ``corpus/raw_issues.jsonl`` and ``corpus/monai_wiki/*.md``, chunks each
according to its strategy, embeds via the Vault-configured Ollama embed
model, and upserts into the ``rag_chunks`` table.

Run inside the api container:
    docker exec docker-api-1 python scripts/ingest_corpus.py

The script reads ``DATABASE_URL`` and ``OLLAMA_HOST`` from the same Settings
the rest of the app uses — no ad-hoc ``os.getenv`` defaults that drift away
from the running stack.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Allow importing app.* when run with `python scripts/ingest_corpus.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _resolve_corpus_dir() -> Path:
    """Pick the first existing corpus location.

    Looks (in order):
      1. /corpus            — bind-mounted in dev / mounted in CI
      2. <repo-root>/corpus — local dev outside docker
      3. /app/corpus        — copied-in baseline (no bind mount)
    """
    candidates = [
        Path("/corpus"),
        Path(__file__).resolve().parents[2] / "corpus",
        Path("/app/corpus"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"corpus/ not found. Checked: {[str(c) for c in candidates]}")


async def main() -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import get_settings
    from app.rag.embeddings import EmbeddingModel
    from app.rag.ingest import CorpusIngestor

    settings = get_settings()
    corpus_dir = _resolve_corpus_dir()
    print(f"corpus dir: {corpus_dir}")

    # ── DB + embedder from Settings (no os.getenv) ────────────────────────────
    print(f"database: {settings.database_url[:60]}...")
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print(f"embedder: ollama @ {settings.ollama_host} model={settings.ollama_embed_model}")
    embedder = EmbeddingModel(
        model_name=settings.ollama_embed_model,
        ollama_host=settings.ollama_host,
    )

    ingestor = CorpusIngestor()

    # ── Issues ────────────────────────────────────────────────────────────────
    issues_path = corpus_dir / "raw_issues.jsonl"
    issues_stats: dict[str, int] = {"ingested": 0, "skipped": 0, "errors": 0}
    if issues_path.exists():
        print(f"loading issues: {issues_path}")
        issues: list[dict] = []
        with issues_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    issues.append(json.loads(line))
        print(f"  {len(issues)} issues loaded")

        # Optional leakage guard for RAG golden set
        exclude_path = corpus_dir / "rag_golden_issue_ids.txt"
        excluded: set[int] = set()
        if exclude_path.exists():
            excluded = {int(line) for line in exclude_path.read_text().splitlines() if line.strip()}
            print(f"  excluding {len(excluded)} RAG-golden issue IDs")

        async with async_session() as session:
            issues_stats = await ingestor.ingest_issues(
                issues=issues,
                embedder=embedder,
                db_session=session,
                exclude_issue_ids=excluded,
            )
        print(f"  issues done: {issues_stats}")
    else:
        print(f"  (no raw_issues.jsonl at {issues_path} — skipping issues)")

    # ── Wiki ──────────────────────────────────────────────────────────────────
    wiki_dir = corpus_dir / "monai_wiki"
    wiki_stats: dict[str, int] = {"files": 0, "chunks": 0, "errors": 0}
    if wiki_dir.exists():
        wiki_files: list[tuple[str, str]] = []
        for md in sorted(wiki_dir.glob("*.md")):
            try:
                content = md.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                print(f"  WARN: cannot read {md.name}: {exc}")
                continue
            if content.strip():
                wiki_files.append((f"monai_wiki/{md.name}", content))
        print(f"loading wiki: {len(wiki_files)} pages")

        async with async_session() as session:
            wiki_stats = await ingestor.ingest_wiki(
                wiki_files=wiki_files,
                embedder=embedder,
                db_session=session,
            )
        print(f"  wiki done: {wiki_stats}")
    else:
        print(f"  (no {wiki_dir} — skipping wiki)")

    await engine.dispose()
    print("ingestion complete.")
    print(f"  total: issues={issues_stats}, wiki={wiki_stats}")


if __name__ == "__main__":
    asyncio.run(main())
