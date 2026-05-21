"""Named Entity Recognition endpoint: POST /ner

Strategy (most-to-least efficient):
  1. Code-specific entities (FunctionName, ErrorType, FilePath) — always via regex.
     Regex is faster and more precise than any NLP model for structured patterns.
  2. General named entities (ORG, PERSON, PRODUCT, etc.) — via spaCy en_core_web_sm
     when available; silently omitted when spaCy is absent (e.g. unit-test env).

spaCy is ~12 MB and loaded once at module import time.  Regex runs on every call.
"""

from __future__ import annotations

import re

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ner", tags=["ner"])

# ---------------------------------------------------------------------------
# spaCy — optional; regex is the primary path for code entities
# ---------------------------------------------------------------------------
try:
    import spacy  # type: ignore[import-untyped]

    _nlp = spacy.load("en_core_web_sm")
    logger.info("ner.spacy_loaded", model="en_core_web_sm")
except (ImportError, OSError) as _spacy_err:
    _nlp = None
    logger.warning("ner.spacy_unavailable", reason=str(_spacy_err))

# spaCy entity labels we surface (the model also detects CARDINAL, DATE, etc.
# which are less useful for issue triage and add noise).
_SPACY_LABELS = frozenset({"ORG", "PERSON", "PRODUCT", "GPE", "NORP"})

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NERRequest(BaseModel):
    """Request body for /ner."""

    text: str = Field(..., min_length=1, max_length=4096, description="Text to extract entities from")


class Entity(BaseModel):
    """A single extracted entity."""

    text: str
    label: str  # FunctionName | ErrorType | FilePath | ORG | PERSON | PRODUCT | …
    start: int
    end: int


class NERResponse(BaseModel):
    """Response from /ner."""

    entities: list[Entity]
    spacy_available: bool


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# Regex patterns for code-specific entities — faster + more precise than NLP.
_FUNC_RE = re.compile(r"\b([a-z_][a-z0-9_]{1,})\s*\(")
_ERROR_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning))\b")
_PATH_RE = re.compile(r"([a-zA-Z0-9_./-]+\.(?:py|pyi|json|yaml|yml|txt|md|cfg|ini))")
_PKG_RE = re.compile(r"(?:^|\s)(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)", re.MULTILINE)


def _extract_regex_entities(text: str) -> list[Entity]:
    """Extract code-specific entities with regex patterns.

    Args:
        text: Raw text from the issue body.

    Returns:
        List of Entity objects (FunctionName, ErrorType, FilePath, PackageName).
    """
    seen: set[tuple[int, int]] = set()
    entities: list[Entity] = []

    def _add(match: re.Match[str], label: str, group: int = 1) -> None:
        start, end = match.start(group), match.end(group)
        if (start, end) not in seen:
            seen.add((start, end))
            entities.append(Entity(text=match.group(group), label=label, start=start, end=end))

    for m in _ERROR_RE.finditer(text):
        _add(m, "ErrorType")
    for m in _FUNC_RE.finditer(text):
        _add(m, "FunctionName")
    for m in _PATH_RE.finditer(text):
        _add(m, "FilePath")
    for m in _PKG_RE.finditer(text):
        _add(m, "PackageName")

    return entities


def _extract_spacy_entities(text: str) -> list[Entity]:
    """Extract general named entities via spaCy (ORG, PERSON, PRODUCT, …).

    Only called when spaCy is available.  Skips entities whose span overlaps
    with already-found code entities.

    Args:
        text: Raw text to process.

    Returns:
        List of Entity objects for general NER labels.
    """
    if _nlp is None:
        return []
    doc = _nlp(text)
    return [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
        if ent.label_ in _SPACY_LABELS
    ]


def extract_code_entities(text: str) -> list[Entity]:
    """Combine regex code-entity extraction with spaCy general NER.

    Args:
        text: Raw text to process.

    Returns:
        Deduplicated, position-sorted list of Entity objects.
    """
    regex_entities = _extract_regex_entities(text)
    spacy_entities = _extract_spacy_entities(text)

    # Merge: skip spaCy entities that overlap with regex hits.
    regex_spans = {(e.start, e.end) for e in regex_entities}
    merged: list[Entity] = list(regex_entities)
    for e in spacy_entities:
        if not any(e.start < r_end and e.end > r_start for r_start, r_end in regex_spans):
            merged.append(e)

    return sorted(merged, key=lambda e: e.start)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", response_model=NERResponse)
async def extract_entities(req: NERRequest) -> NERResponse:
    """Extract named entities from text.

    Code-specific labels (always present): FunctionName, ErrorType, FilePath, PackageName.
    General NER labels (when spaCy is available): ORG, PERSON, PRODUCT, GPE, NORP.

    Args:
        req: NERRequest with text field.

    Returns:
        NERResponse with entity list and spacy_available flag.
    """
    try:
        entities = extract_code_entities(req.text)
        logger.info(
            "ner.extracted",
            entity_count=len(entities),
            text_length=len(req.text),
            spacy=_nlp is not None,
        )
        return NERResponse(entities=entities, spacy_available=_nlp is not None)

    except Exception as e:
        logger.exception("ner.failed", error=str(e))
        raise HTTPException(status_code=500, detail="NER extraction failed") from e
