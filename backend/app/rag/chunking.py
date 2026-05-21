"""Markdown-aware recursive chunking for RAG corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk from corpus with metadata."""

    text: str
    metadata: dict[str, str | int | list[str]]
    source: str
    chunk_id: str


class MarkdownChunker:
    """Recursively chunk markdown respecting header hierarchy.

    Strategy:
    1. Split by headers (h1 → h4)
    2. If section > max_chunk_size, split by paragraphs
    3. If paragraph > max_chunk_size, split by sentences
    4. Preserve header context in chunk
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 50,
        overlap: int = 50,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        source: str,
        metadata: dict[str, str | int | list[str]] | None = None,
    ) -> list[Chunk]:
        """Chunk markdown text recursively.

        Args:
            text: Raw markdown content
            source: Source identifier (file path, issue ID, etc.)
            metadata: Additional metadata to attach to each chunk

        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}

        chunks = []
        chunk_id_counter = 0

        # Split by top-level headers (h1)
        h1_sections = self._split_by_header(text, level=1)

        for h1_header, h1_content in h1_sections:
            # Split each h1 section by h2
            h2_sections = self._split_by_header(h1_content, level=2)

            for h2_header, h2_content in h2_sections:
                # Split each h2 by h3
                h3_sections = self._split_by_header(h2_content, level=3)

                for h3_header, h3_content in h3_sections:
                    # Build header context
                    headers = [h for h in [h1_header, h2_header, h3_header] if h]
                    header_context = " > ".join(headers)

                    # Split h3 content by paragraphs
                    paragraphs = self._split_paragraphs(h3_content)

                    # Accumulate chunks
                    current_chunk = header_context + "\n\n" if header_context else ""
                    current_size = len(current_chunk)

                    for para in paragraphs:
                        # If a single paragraph is oversized, split it by words first
                        sub_paras = [para]
                        if len(para) > self.max_chunk_size:
                            sub_paras = self._split_large_paragraph(para)

                        for sub_para in sub_paras:
                            para_size = len(sub_para)

                            # If adding para exceeds max_chunk_size, flush current
                            if (
                                current_size + para_size > self.max_chunk_size
                                and current_chunk.strip()
                            ):
                                chunk_text = current_chunk.strip()
                                if len(chunk_text) >= self.min_chunk_size:
                                    chunks.append(
                                        Chunk(
                                            text=chunk_text,
                                            metadata=metadata.copy(),
                                            source=source,
                                            chunk_id=f"{source}#{chunk_id_counter}",
                                        )
                                    )
                                    chunk_id_counter += 1

                                # Start new chunk with overlap
                                if header_context:
                                    current_chunk = header_context + "\n\n"
                                    current_size = len(current_chunk)
                                else:
                                    current_chunk = ""
                                    current_size = 0

                            current_chunk += sub_para + "\n\n"
                            current_size += para_size + 2

                    # Flush remaining chunk
                    if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                        chunks.append(
                            Chunk(
                                text=current_chunk.strip(),
                                metadata=metadata.copy(),
                                source=source,
                                chunk_id=f"{source}#{chunk_id_counter}",
                            )
                        )
                        chunk_id_counter += 1

        return chunks if chunks else self._fallback_split(text, source, metadata)

    def _split_by_header(self, text: str, level: int) -> list[tuple[str, str]]:
        """Split markdown by header level.

        Returns:
            List of (header_text, content) tuples
        """
        pattern = rf"^{'#' * level}\s+(.+)$"
        matches = list(re.finditer(pattern, text, re.MULTILINE))

        if not matches:
            return [("", text)]

        sections = []
        for i, match in enumerate(matches):
            header = match.group(1)
            start = match.end() + 1  # After newline
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections.append((header, content))

        return sections

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text by paragraph (blank lines)."""
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _split_large_paragraph(self, text: str) -> list[str]:
        """Split an oversized paragraph by words into max_chunk_size pieces."""
        parts: list[str] = []
        words = text.split()
        current: list[str] = []
        current_size = 0
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > self.max_chunk_size and current:
                parts.append(" ".join(current))
                current = []
                current_size = 0
            current.append(word)
            current_size += word_size
        if current:
            parts.append(" ".join(current))
        return parts

    def _fallback_split(
        self,
        text: str,
        source: str,
        metadata: dict,
    ) -> list[Chunk]:
        """Fallback: split by fixed size if no structure found."""
        chunks = []
        chunk_id_counter = 0

        words = text.split()
        current_chunk = []
        current_size = 0

        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > self.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            metadata=metadata.copy(),
                            source=source,
                            chunk_id=f"{source}#{chunk_id_counter}",
                        )
                    )
                    chunk_id_counter += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(word)
            current_size += word_size

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata=metadata.copy(),
                        source=source,
                        chunk_id=f"{source}#{chunk_id_counter}",
                    )
                )

        return chunks
