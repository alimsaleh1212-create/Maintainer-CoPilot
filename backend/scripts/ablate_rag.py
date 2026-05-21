"""RAG ablation harness.

Runs four progressively-enriched RAG configurations against a fixed probe
set (split across wiki + issue questions) and writes a markdown report
suitable for DECISIONS.md / EVALS.md.

Configurations:

    A. Naive          — dense pgvector only, no expansion, no parent-expand.
    B. +Hybrid        — dense + sparse BM25 (60/40), no expansion, no parents.
    C. +Query rewrite — A's hybrid + multi-query expansion.
    D. +Parent-child  — full pipeline (C) + parent-text expansion.

For each (config × probe) we record:
    - top-1 chunk_id, source, score
    - top-5 sources (issue/wiki mix)
    - parent_id of top-1 (None if parent-expand disabled)
    - retrieval latency (ms, one cold + one warm call averaged)

Run inside the api container:
    docker exec docker-api-1 python scripts/ablate_rag.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Probe set — questions deliberately chosen so we know which source SHOULD win.
# "expected" is what we want to see in top-5: "wiki", "issue", or "any".
PROBES: list[dict[str, str]] = [
    # ── Wiki-expected (developer/conceptual questions) ──────────────────────
    {"q": "How do I write a custom MONAI transform?", "expected": "wiki"},
    {"q": "What does MetaTensor track during preprocessing?", "expected": "wiki"},
    {"q": "Explain lazy transforms in MONAI", "expected": "wiki"},
    {"q": "How is the Compose pipeline applied?", "expected": "wiki"},
    {"q": "What is the MONAI network design philosophy?", "expected": "wiki"},
    # ── Issue-expected (bug-style or "I have an error" questions) ──────────
    {"q": "GPU out of memory when training 3D UNet", "expected": "issue"},
    {"q": "CUDA out of memory with batch size 4", "expected": "issue"},
    {"q": "AttributeError in CropForegroundd transform", "expected": "issue"},
    {"q": "DataLoader hangs on multi-worker", "expected": "issue"},
    {"q": "Loss is NaN after first epoch", "expected": "issue"},
]


async def _run_query(
    query: str,
    *,
    use_sparse: bool,
    use_rewrite: bool,
    use_parents: bool,
    db_session,
    embedder,
) -> dict:
    """One retrieval call configured per the ablation flags."""
    from app.rag.retrieval import HybridRetriever
    from app.rag.rewrite import MultiQueryExpander

    retriever = HybridRetriever(
        dense_weight=1.0 if not use_sparse else 0.6,
        sparse_weight=0.0 if not use_sparse else 0.4,
        top_k_before_rerank=20,
        top_k_final=5,
    )

    if use_rewrite:
        expander = MultiQueryExpander()
        variations = await expander.expand(query, gemini_api_key=None)
    else:
        variations = [query]

    start = time.perf_counter()
    chunks = await retriever.retrieve(
        query_variations=variations,
        embedding_fn=embedder.embed,
        db_session=db_session,
        reranker=None,
        top_k=5,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    if not use_parents:
        # Strip parent fields so the comparison is honest
        for c in chunks:
            c.parent_id = None
            c.parent_text = None

    return {
        "query": query,
        "variations": variations,
        "latency_ms": round(latency_ms, 1),
        "top_5": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "score": round(c.score, 4),
                "parent_id": c.parent_id,
                "has_parent_text": bool(c.parent_text),
            }
            for c in chunks
        ],
    }


def _hit_in_top5(results: list[dict], expected: str) -> str:
    """Did the expected source show up in top-5? Returns ✓/✗/—."""
    if expected == "any":
        return "—"
    expected_db = "docs" if expected == "wiki" else "issue"
    for r in results:
        if r["source"] == expected_db:
            return "✓"
    return "✗"


def _format_top1(results: list[dict]) -> str:
    if not results:
        return "—"
    r = results[0]
    src_label = "wiki" if r["source"] == "docs" else "issue"
    return f"{src_label} ({r['score']:.3f})"


async def main() -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from app.config import get_settings
    from app.rag.embeddings import EmbeddingModel

    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    embedder = EmbeddingModel(
        model_name=settings.ollama_embed_model,
        ollama_host=settings.ollama_host,
    )

    configs = [
        ("A. Naive", dict(use_sparse=False, use_rewrite=False, use_parents=False)),
        ("B. +Hybrid", dict(use_sparse=True, use_rewrite=False, use_parents=False)),
        ("C. +Query rewrite", dict(use_sparse=True, use_rewrite=True, use_parents=False)),
        ("D. +Parent-child", dict(use_sparse=True, use_rewrite=True, use_parents=True)),
    ]

    # results[config][query_idx] = dict
    all_results: dict[str, list[dict]] = {}
    for name, flags in configs:
        print(f"\n=== {name} ===")
        runs: list[dict] = []
        async with async_session() as session:
            for probe in PROBES:
                run = await _run_query(
                    probe["q"], db_session=session, embedder=embedder, **flags
                )
                run["expected"] = probe["expected"]
                run["hit"] = _hit_in_top5(run["top_5"], probe["expected"])
                runs.append(run)
                print(
                    f"  {run['hit']} {probe['expected']:<6} {probe['q'][:50]:<52} "
                    f"top1={_format_top1(run['top_5'])} {run['latency_ms']:.0f}ms"
                )
        all_results[name] = runs

    # ── Write JSON dump (machine-readable) + Markdown report (human) ─────────
    out_json = Path("/corpus/rag_ablation.json")
    out_md = Path("/corpus/rag_ablation.md")
    out_json.write_text(json.dumps(all_results, indent=2, default=str))

    lines: list[str] = []
    lines.append("# RAG ablation study")
    lines.append("")
    lines.append(
        "Each row is one of 10 probe queries. Five are wiki-expected "
        "(developer/conceptual), five are issue-expected (bug-style). The "
        "`hit` column shows whether the expected source surfaced in top-5."
    )
    lines.append("")

    # Summary table — Hit@5 per config × expected source
    lines.append("## Hit@5 summary")
    lines.append("")
    lines.append("| Config | Wiki Hit@5 | Issue Hit@5 | Combined Hit@5 | Avg latency (ms) |")
    lines.append("|---|---|---|---|---|")
    for name, _ in configs:
        runs = all_results[name]
        wiki = [r for r in runs if r["expected"] == "wiki"]
        issue = [r for r in runs if r["expected"] == "issue"]
        wiki_hit = sum(1 for r in wiki if r["hit"] == "✓")
        issue_hit = sum(1 for r in issue if r["hit"] == "✓")
        total = wiki_hit + issue_hit
        avg_lat = sum(r["latency_ms"] for r in runs) / max(1, len(runs))
        lines.append(
            f"| {name} | {wiki_hit}/{len(wiki)} | {issue_hit}/{len(issue)} | "
            f"{total}/{len(runs)} | {avg_lat:.0f} |"
        )
    lines.append("")

    # Per-query detail
    lines.append("## Per-query detail")
    lines.append("")
    for probe in PROBES:
        lines.append(f"### {probe['expected']}: _{probe['q']}_")
        lines.append("")
        lines.append("| Config | hit | top-1 source | top-1 score | top-5 sources | latency |")
        lines.append("|---|---|---|---|---|---|")
        for name, _ in configs:
            for run in all_results[name]:
                if run["query"] != probe["q"]:
                    continue
                top5 = run["top_5"]
                top1 = top5[0] if top5 else None
                src_mix = ", ".join(
                    ("wiki" if c["source"] == "docs" else "issue") for c in top5
                )
                top1_src = (
                    ("wiki" if top1["source"] == "docs" else "issue") if top1 else "—"
                )
                top1_score = f"{top1['score']:.3f}" if top1 else "—"
                lines.append(
                    f"| {name} | {run['hit']} | {top1_src} | {top1_score} | "
                    f"{src_mix} | {run['latency_ms']:.0f}ms |"
                )
        lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_md}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
