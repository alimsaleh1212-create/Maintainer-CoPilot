"""Chunking strategies for the RAG corpus.

Two classes of source document, two strategies:

1. **Issues** (``IssueChunker``) → one ``Chunk`` per issue.
   An issue is a coherent unit (title + body + comments); splitting loses
   the bug-to-fix relationship that makes it useful for retrieval. The
   embedder will silently truncate at its 512-token cap for very long
   issues — acceptable for tail content.

2. **Wiki pages** (``MarkdownChunker``) → recursive header-aware chunking.
   Children are emitted with the heading breadcrumb prepended ("page > h2 > h3")
   and a reference back to the full parent document for small-to-big retrieval.

Both produce ``Chunk`` objects with the same shape so the ingest pipeline
treats them identically.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single retrievable chunk.

    Attributes:
        text: The text that gets embedded and matched against queries.
        chunk_id: Stable unique identifier (used as the row's chunk_id).
        source: "issue" | "docs".
        parent_id: Stable identifier for the document this chunk belongs to.
            For issues, equals ``chunk_id`` (the chunk IS the parent).
            For wiki, equals the source file path (relative to ``corpus/``).
        parent_text: Full parent document text. ``None`` when the chunk
            already contains the entire parent (issues).
        metadata: Free-form JSON-serializable annotations
            (heading_path, labels, issue_number, file_path, ...).
    """

    text: str
    chunk_id: str
    source: str
    parent_id: str
    parent_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Issues ───────────────────────────────────────────────────────────────────


class IssueChunker:
    """One chunk per issue. No structural splitting."""

    def chunk(self, issue: dict[str, Any]) -> Chunk:
        """Build a single ``Chunk`` from a raw issue dict.

        Args:
            issue: A row from ``corpus/raw_issues.jsonl`` with at least
                ``number`` (or ``id``), ``title``, ``body``. Optional:
                ``labels``, ``state``, ``created_at``, ``comments``.

        Returns:
            A single ``Chunk`` carrying the concatenated issue text.
        """
        issue_number = issue.get("number") or issue.get("id")
        title = (issue.get("title") or "").strip()
        body = (issue.get("body") or "").strip()
        labels = issue.get("labels") or []

        parts = [f"# {title}"]
        if body:
            parts.append(body)

        comments = issue.get("comments") or []
        if isinstance(comments, list) and comments:
            top = [c for c in comments[:3] if isinstance(c, dict) and c.get("body")]
            if top:
                parts.append("\n## Comments")
                for c in top:
                    parts.append(c["body"].strip())

        full_text = "\n\n".join(parts)
        parent_id = f"issue_{issue_number}"
        return Chunk(
            text=full_text,
            chunk_id=parent_id,
            source="issue",
            parent_id=parent_id,
            parent_text=None,  # the chunk IS the parent
            metadata={
                "source_type": "issue",
                "issue_number": issue_number,
                "labels": labels if isinstance(labels, list) else [],
                "state": issue.get("state"),
                "created_at": issue.get("created_at"),
                "url": issue.get("html_url") or issue.get("url"),
            },
        )


# ─── Markdown (wiki pages) ────────────────────────────────────────────────────


class MarkdownChunker:
    """Recursive header-aware splitter for wiki pages.

    Recursion order:
      1. Split by ``## ``  (h2)
      2. If still too big → split by ``### `` (h3)
      3. If still too big → split by paragraphs (blank lines)
      4. Last resort → character window with word boundary

    Hard rules:
      - Fenced code blocks (``` ... ```) are atomic — never split inside.
      - Every chunk text is prefixed with its heading breadcrumb.
      - Every chunk carries ``parent_id`` and ``parent_text`` pointing back
        to the full file.
    """

    _CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

    def __init__(self, max_chars: int = 2000, min_chars: int = 100) -> None:
        # ~2000 chars ≈ 512 tokens (4 chars/token heuristic). Matches the
        # 512-token embedder cap with a small safety margin.
        self.max_chars = max_chars
        self.min_chars = min_chars

    def chunk(self, text: str, file_path: str) -> list[Chunk]:
        """Chunk a wiki markdown document.

        Args:
            text: Full file contents.
            file_path: Relative path within ``corpus/`` (used as ``parent_id``).

        Returns:
            One or more ``Chunk`` objects; each carries the full parent text.
        """
        full_text = text
        # Page title from H1 (or filename)
        h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        page_title = (h1_match.group(1).strip() if h1_match else _filename_to_title(file_path))

        h2_sections = self._split_by_header(text, level=2)
        chunks: list[Chunk] = []

        if not h2_sections:
            # No headers — whole file is one section
            for piece in self._split_oversize(text):
                chunks.append(
                    self._make_chunk(
                        body=piece,
                        breadcrumb=page_title,
                        file_path=file_path,
                        parent_text=full_text,
                    )
                )
            return [c for c in chunks if len(c.text) >= self.min_chars] or self._fallback(
                full_text, file_path, page_title
            )

        for h2_title, h2_body in h2_sections:
            h3_sections = self._split_by_header(h2_body, level=3)
            if not h3_sections:
                h3_sections = [("", h2_body)]

            for h3_title, h3_body in h3_sections:
                breadcrumb_parts = [page_title, h2_title, h3_title]
                breadcrumb = " > ".join(p for p in breadcrumb_parts if p)
                for piece in self._split_oversize(h3_body):
                    chunks.append(
                        self._make_chunk(
                            body=piece,
                            breadcrumb=breadcrumb,
                            file_path=file_path,
                            parent_text=full_text,
                        )
                    )

        chunks = [c for c in chunks if len(c.text) >= self.min_chars]
        return chunks or self._fallback(full_text, file_path, page_title)

    # ── Internals ────────────────────────────────────────────────────────────

    def _split_by_header(self, text: str, level: int) -> list[tuple[str, str]]:
        marker = "#" * level
        pattern = rf"^{marker}\s+(.+)$"
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if not matches:
            return []

        sections: list[tuple[str, str]] = []
        # Preserve any pre-header preamble (text before first match)
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("", preamble))

        for i, match in enumerate(matches):
            header = match.group(1).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((header, body))
        return sections

    def _split_oversize(self, text: str) -> list[str]:
        """If a section is too big, split — but never inside a fence."""
        if len(text) <= self.max_chars:
            return [text] if text.strip() else []

        # Identify code-fence spans; keep them atomic
        fences = [(m.start(), m.end()) for m in self._CODE_FENCE_RE.finditer(text)]

        def in_fence(idx: int) -> bool:
            return any(start <= idx < end for start, end in fences)

        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0

        # Split by blank lines (paragraphs) as primary boundary
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            para_size = len(para) + 2

            # If para alone is oversize and not a code fence, hard-split it
            if para_size > self.max_chars and not self._CODE_FENCE_RE.fullmatch(para):
                if buf:
                    pieces.append("\n\n".join(buf))
                    buf = []
                    buf_len = 0
                pieces.extend(self._word_split(para))
                continue

            if buf_len + para_size > self.max_chars and buf:
                pieces.append("\n\n".join(buf))
                buf = [para]
                buf_len = para_size
            else:
                buf.append(para)
                buf_len += para_size

        if buf:
            pieces.append("\n\n".join(buf))

        # Filter & dedupe trivial pieces
        out: list[str] = []
        for p in pieces:
            ps = p.strip()
            if ps and ps not in out:
                out.append(ps)
        return out or [text]

    def _word_split(self, text: str) -> list[str]:
        """Last-resort split — by words. Avoids splitting mid-word."""
        words = text.split()
        out: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for w in words:
            wl = len(w) + 1
            if buf_len + wl > self.max_chars and buf:
                out.append(" ".join(buf))
                buf = [w]
                buf_len = wl
            else:
                buf.append(w)
                buf_len += wl
        if buf:
            out.append(" ".join(buf))
        return out

    def _make_chunk(
        self,
        body: str,
        breadcrumb: str,
        file_path: str,
        parent_text: str,
    ) -> Chunk:
        prefixed = f"{breadcrumb}\n\n{body}" if breadcrumb else body
        # Stable chunk_id = file_path + content hash so re-ingest is idempotent
        digest = hashlib.sha256(prefixed.encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{file_path}#{digest}"
        return Chunk(
            text=prefixed,
            chunk_id=chunk_id,
            source="docs",
            parent_id=file_path,
            parent_text=parent_text,
            metadata={
                "source_type": "wiki",
                "file_path": file_path,
                "heading_path": breadcrumb,
            },
        )

    def _fallback(self, text: str, file_path: str, page_title: str) -> list[Chunk]:
        return [
            self._make_chunk(
                body=text.strip(),
                breadcrumb=page_title,
                file_path=file_path,
                parent_text=text,
            )
        ]


def _filename_to_title(path: str) -> str:
    name = path.rsplit("/", 1)[-1].removesuffix(".md")
    return name.replace("-", " ").replace("_", " ").strip()
