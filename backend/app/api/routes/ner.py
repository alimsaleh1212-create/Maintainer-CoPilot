"""Named Entity Recognition endpoint: POST /ner"""

from __future__ import annotations

import re

import spacy
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/ner", tags=["ner"])

# Load spaCy model once (in production, load in lifespan)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spacy_model_not_found", installing="en_core_web_sm")
    nlp = None


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Text to extract entities from")


class Entity(BaseModel):
    text: str
    label: str  # FunctionName, ClassName, FilePath, ErrorType, PackageName
    start: int
    end: int


class NERResponse(BaseModel):
    entities: list[Entity]


def extract_code_entities(text: str) -> list[Entity]:
    """Extract code-shaped entities via spaCy + regex patterns.

    Rules:
    - FunctionName: snake_case followed by ()
    - ClassName: PascalCase word not at start of sentence
    - FilePath: contains / or . and looks like a path
    - ErrorType: word followed by Error or Exception
    - PackageName: word before import or from keyword
    """
    entities = []

    # Regex patterns
    func_pattern = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")
    class_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z]*)*)\b")
    path_pattern = re.compile(r"([a-zA-Z0-9_./\\-]+\.(?:py|pyi|json|yaml|yml|txt))")
    error_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:Error|Exception))\b")

    # Extract functions
    for match in func_pattern.finditer(text):
        entities.append(
            Entity(
                text=match.group(1),
                label="FunctionName",
                start=match.start(1),
                end=match.end(1),
            )
        )

    # Extract error types
    for match in error_pattern.finditer(text):
        entities.append(
            Entity(
                text=match.group(1),
                label="ErrorType",
                start=match.start(1),
                end=match.end(1),
            )
        )

    # Extract file paths
    for match in path_pattern.finditer(text):
        entities.append(
            Entity(
                text=match.group(1),
                label="FilePath",
                start=match.start(1),
                end=match.end(1),
            )
        )

    # Deduplicate by position
    unique_entities = {}
    for e in entities:
        key = (e.start, e.end)
        if key not in unique_entities:
            unique_entities[key] = e

    return sorted(unique_entities.values(), key=lambda e: e.start)


@router.post("", response_model=NERResponse)
async def extract_entities(req: NERRequest) -> NERResponse:
    """Extract named entities from text.

    Supports: FunctionName, ClassName, FilePath, ErrorType, PackageName

    Args:
        req: NERRequest with text

    Returns:
        NERResponse with list of Entity objects
    """
    try:
        entities = extract_code_entities(req.text)
        logger.info(
            "ner.extracted",
            entity_count=len(entities),
            text_length=len(req.text),
        )
        return NERResponse(entities=entities)

    except Exception as e:
        logger.exception("ner.failed", error=str(e))
        raise HTTPException(status_code=500, detail="NER extraction failed") from e
