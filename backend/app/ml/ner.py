"""Named Entity Recognition for GitHub issue text.

Combines spaCy ``en_core_web_sm`` with a regex post-pass to extract
code-shaped entities that a general NLP model misses:
``FunctionName``, ``ClassName``, ``FilePath``, ``ErrorType``,
``PackageName``, ``VersionNumber``.

Falls back to regex-only mode when spaCy is unavailable.

Public API
----------
extract_entities(text) → list[dict[str, str]]
    Async-safe; CPU work is dispatched via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for code-shaped entities
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # FilePath — path/to/file.py  or  ./src/foo/bar.ts
    (
        "FilePath",
        re.compile(
            r"""(?<!\w)                          # no word char before
            \.?/                                # optional leading ./
            (?:[a-zA-Z0-9_\-]+/)+              # one or more path segments
            [a-zA-Z0-9_\-]+                    # final segment (no extension required)
            (?:\.[a-zA-Z]{1,6})?               # optional extension
            (?!\w)                             # no word char after
            """,
            re.VERBOSE,
        ),
    ),
    # VersionNumber — v1.2.3, 1.2.3, 1.2.3.post4, 1.2.3a1
    (
        "VersionNumber",
        re.compile(
            r"""(?<!\w)
            v?                              # optional v prefix
            \d+\.\d+                        # major.minor
            (?:\.\d+)*                      # optional patch / micro parts
            (?:[._-]?(?:a|b|rc|post|dev)\d+)? # pre/post/dev release suffix
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
    # Python built-in exception types + user-defined *Error / *Exception
    (
        "ErrorType",
        re.compile(
            r"""(?<!\w)
            (?:
                ValueError|TypeError|KeyError|IndexError|AttributeError|
                ImportError|ModuleNotFoundError|RuntimeError|StopIteration|
                NotImplementedError|OSError|IOError|FileNotFoundError|
                PermissionError|TimeoutError|OverflowError|MemoryError|
                RecursionError|AssertionError|NameError|UnboundLocalError|
                ZeroDivisionError|UnicodeError|UnicodeDecodeError|
                UnicodeEncodeError|ConnectionError|BrokenPipeError|
                [A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning)
            )
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
    # PackageName — lowercase-with-dashes as used in pip/pypi context
    (
        "PackageName",
        re.compile(
            r"""(?<!\w)
            (?:pip\s+install\s+|import\s+|from\s+)   # leading context keyword
            ([a-zA-Z][a-zA-Z0-9_\-]*)                # the package name
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
    # ClassName — PascalCase identifier (≥2 upper-lower transitions, not all-caps)
    (
        "ClassName",
        re.compile(
            r"""(?<!\w)
            (?:[A-Z][a-z][a-zA-Z0-9]*){2,}   # at least two PascalCase "words"
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
    # FunctionName — snake_case with at least one underscore, not all uppercase
    (
        "FunctionName",
        re.compile(
            r"""(?<!\w)
            [a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+   # snake_case: at least one _
            (?=\s*\()                               # followed by opening paren
            (?!\w)
            """,
            re.VERBOSE,
        ),
    ),
]


def _apply_regex(text: str) -> list[dict[str, str]]:
    """Extract entities using regex patterns.

    Args:
        text: Raw issue/comment text.

    Returns:
        List of entity dicts with keys ``text``, ``label``, ``start``, ``end``.
    """
    entities: list[dict[str, str]] = []
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # PackageName pattern uses a capture group for the actual name
            start, end = (m.start(1), m.end(1)) if m.lastindex else (m.start(), m.end())
            span_text = text[start:end].strip()
            if span_text:
                entities.append(
                    {
                        "text": span_text,
                        "label": label,
                        "start": str(start),
                        "end": str(end),
                    }
                )
    return entities


# ---------------------------------------------------------------------------
# spaCy loader (lazy, cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_spacy() -> Any | None:
    """Load spaCy model, returning None if unavailable.

    Returns:
        Loaded spaCy ``Language`` object, or ``None`` if spaCy / the model
        is not installed.
    """
    try:
        import spacy  # type: ignore[import-untyped]

        nlp = spacy.load("en_core_web_sm")
        logger.info("spacy_model_loaded", model="en_core_web_sm")
        return nlp
    except Exception as exc:
        logger.warning(
            "spacy_unavailable_falling_back_to_regex_only",
            reason=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# spaCy entity extraction
# ---------------------------------------------------------------------------

_SPACY_LABEL_MAP: dict[str, str] = {
    "ORG": "PackageName",
    "PRODUCT": "PackageName",
    "WORK_OF_ART": "PackageName",
}


def _extract_spacy(text: str, nlp: Any) -> list[dict[str, str]]:
    """Run spaCy NER and translate labels to our schema.

    Args:
        text: Raw text.
        nlp: Loaded spaCy Language object.

    Returns:
        Entity dicts with our label vocabulary.
    """
    doc = nlp(text)
    entities: list[dict[str, str]] = []
    for ent in doc.ents:
        mapped_label = _SPACY_LABEL_MAP.get(ent.label_)
        if mapped_label:
            entities.append(
                {
                    "text": ent.text,
                    "label": mapped_label,
                    "start": str(ent.start_char),
                    "end": str(ent.end_char),
                }
            )
    return entities


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def _deduplicate(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove duplicate spans (same start/end/label).

    Prioritises entities that appear earlier in the list (spaCy results
    take precedence over regex when both are present).

    Args:
        entities: Combined entity list (may contain duplicates).

    Returns:
        Deduplicated list sorted by character offset.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for ent in entities:
        key = (ent["start"], ent["end"], ent["label"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
    unique.sort(key=lambda e: int(e["start"]))
    return unique


# ---------------------------------------------------------------------------
# Synchronous inner implementation (called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _extract_sync(text: str) -> list[dict[str, str]]:
    """CPU-bound extraction combining spaCy + regex.

    Args:
        text: Issue text to analyse.

    Returns:
        Deduplicated, sorted list of entity dicts.
    """
    entities: list[dict[str, str]] = []

    nlp = _load_spacy()
    if nlp is not None:
        entities.extend(_extract_spacy(text, nlp))

    entities.extend(_apply_regex(text))
    return _deduplicate(entities)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract code-shaped named entities from GitHub issue text.

    Dispatches CPU-bound work to a thread pool via ``asyncio.to_thread``
    so the event loop is not blocked.

    Args:
        text: Raw issue title + body (or any free text).

    Returns:
        List of entity dicts, each containing:
            - ``text``: The matched string.
            - ``label``: One of ``FunctionName``, ``ClassName``, ``FilePath``,
              ``ErrorType``, ``PackageName``, ``VersionNumber``.
            - ``start``: Character offset of the start of the span (as str).
            - ``end``: Character offset of the end of the span (as str).

    Example::

        entities = await extract_entities("ValueError in utils/helper.py v1.2.3")
        # [{"text": "ValueError", "label": "ErrorType", "start": "0", "end": "10"}, ...]
    """
    if not text:
        return []
    return await asyncio.to_thread(_extract_sync, text)
