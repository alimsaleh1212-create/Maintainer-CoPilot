"""Deterministic citation builder.

Constructs ``Citation`` objects from retrieved chunks, server-side. This is
explicitly NOT done by the LLM — letting the LLM build citations risks
hallucinated links and inconsistent labels. Instead, the LLM gets context
that already references stable citation IDs (``[1]``, ``[2]``...) and the
server returns the citation list alongside the answer.

Confidence:
    Each citation carries a normalized score in [0, 1] derived from the
    retriever's combined rerank/hybrid score. A ``min_confidence`` filter
    drops citations the UI shouldn't surface (defaults to 0.30 — anything
    below that is usually retrieval noise).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Citation:
    """A single citation surfaced under a RAG answer.

    The ``id`` is the stable in-answer reference (e.g. ``[1]``).
    The ``label`` is what the UI renders (e.g. ``"Issue #4521"`` or
    ``"MetaTensor-guide.md > Composition"``).
    """

    id: int
    source_type: str  # "issue" | "wiki"
    label: str
    url: str | None
    score: float
    chunk_id: str
    snippet: str  # short preview, ≤ 240 chars

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "label": self.label,
            "url": self.url,
            "score": round(self.score, 4),
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }


# Public, GitHub-style URL templates. Kept here so callers don't reinvent.
_ISSUE_URL_TEMPLATE = "https://github.com/Project-MONAI/MONAI/issues/{number}"
_WIKI_URL_TEMPLATE = "https://github.com/Project-MONAI/MONAI/wiki/{page}"

_DEFAULT_MIN_CONFIDENCE = 0.30
_SNIPPET_MAX_CHARS = 240


def build_citations(
    chunks: list[Any],
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    max_citations: int = 5,
) -> list[Citation]:
    """Build a deduplicated citation list from retrieved chunks.

    Args:
        chunks: List of objects exposing ``source``, ``score``, ``chunk_id``,
            ``text``, ``metadata`` (dict). Both ``RetrievedChunk`` and
            ``SearchResult`` shapes satisfy this — duck-typed on purpose so
            callers don't need a shared import.
        min_confidence: Drop citations whose score is below this threshold.
            Default 0.30 — below this is usually retrieval noise.
        max_citations: Hard cap on returned citations. Default 5.

    Returns:
        Citations ordered by score (highest first), de-duplicated by parent.
    """
    seen_parents: set[str] = set()
    citations: list[Citation] = []
    counter = 1

    # Stable ordering: sort by score desc, then by chunk_id for tie-break
    ordered = sorted(
        chunks,
        key=lambda c: (-_get_score(c), _get_chunk_id(c)),
    )

    for chunk in ordered:
        score = _get_score(chunk)
        if score < min_confidence:
            continue

        source = _get_source(chunk)
        meta = _get_metadata(chunk)
        parent_key = _parent_key(chunk, meta)
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)

        label, url = _label_and_url(source, meta)
        snippet = _build_snippet(_get_text(chunk))

        citations.append(
            Citation(
                id=counter,
                source_type="issue" if source == "issue" else "wiki",
                label=label,
                url=url,
                score=score,
                chunk_id=_get_chunk_id(chunk),
                snippet=snippet,
            )
        )
        counter += 1
        if counter > max_citations:
            break

    return citations


# ── Internals ───────────────────────────────────────────────────────────────


def _get_score(c: Any) -> float:
    raw: Any = getattr(c, "rerank_score", None) or getattr(c, "score", 0.0)
    return float(raw)


def _get_source(c: Any) -> str:
    return str(getattr(c, "source", ""))


def _get_chunk_id(c: Any) -> str:
    return str(getattr(c, "chunk_id", ""))


def _get_text(c: Any) -> str:
    return str(getattr(c, "text", ""))


def _get_metadata(c: Any) -> dict[str, Any]:
    meta = getattr(c, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _parent_key(c: Any, meta: dict[str, Any]) -> str:
    """Group chunks under the same source so we cite each parent once."""
    parent_id = getattr(c, "parent_id", None) or meta.get("file_path")
    if parent_id:
        return str(parent_id)
    # Issues fall back to issue_number; otherwise use chunk_id itself
    return str(meta.get("issue_number") or _get_chunk_id(c))


def _label_and_url(source: str, meta: dict[str, Any]) -> tuple[str, str | None]:
    if source == "issue":
        number = meta.get("issue_number")
        url = meta.get("url") or (
            _ISSUE_URL_TEMPLATE.format(number=number) if number is not None else None
        )
        return (f"Issue #{number}" if number is not None else "Issue", url)

    # Wiki / docs
    file_path = str(meta.get("file_path") or "")
    page_name = file_path.rsplit("/", 1)[-1].removesuffix(".md")
    heading = str(meta.get("heading_path") or "")
    heading_tail = heading.split(" > ", 1)[-1] if " > " in heading else heading
    label = page_name.replace("-", " ").replace("_", " ").strip()
    if heading_tail and heading_tail.lower() != label.lower():
        label = f"{label} — {heading_tail}"
    url = _WIKI_URL_TEMPLATE.format(page=page_name) if page_name else None
    return (label or "Wiki", url)


def _build_snippet(text: str) -> str:
    text = text.strip()
    if len(text) <= _SNIPPET_MAX_CHARS:
        return text
    cut = text[:_SNIPPET_MAX_CHARS].rsplit(" ", 1)[0]
    return cut + "…"
