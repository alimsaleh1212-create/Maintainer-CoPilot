"""Classification endpoint: POST /classify

Routes issue text through the cascading ClassificationService.
DL (DistilBERT) is always tried first; the LLM fallback fires automatically
when confidence < Settings.classify_cascade_threshold.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.dependencies import ClassificationServiceDep

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["classification"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Issue text to classify")
    max_length: int = Field(default=512, ge=128, le=1024, description="Tokenizer max length")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted label: bug, feature, or support")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str = Field(
        ..., description="Model identifier (includes 'cascade:' prefix when LLM was used)"
    )
    latency_ms: float = Field(..., ge=0.0, description="Total inference latency in milliseconds")
    cascade_triggered: bool = Field(
        default=False,
        description="True when DistilBERT confidence was below threshold and the LLM was used",
    )
    cascade_model: str | None = Field(
        default=None,
        description="LLM used for cascade (e.g. 'gemini-2.0-flash'), or null if not triggered",
    )


@router.post("", response_model=ClassifyResponse)
async def classify(
    req: ClassifyRequest,
    classification_service: ClassificationServiceDep,
) -> ClassifyResponse:
    """Classify an issue via the cascading classification pipeline.

    Calls DistilBERT (model-server) first.  If confidence is below
    ``classify_cascade_threshold`` in Settings, the LLM is used automatically.

    Args:
        req: ClassifyRequest with issue text and optional tokenizer max_length.
        classification_service: Injected ClassificationService (cascade-enabled).

    Returns:
        ClassifyResponse including ``cascade_triggered`` and ``cascade_model``.
    """
    result = await classification_service.classify(req.text, req.max_length)
    return ClassifyResponse(
        label=result.label,
        confidence=result.confidence,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        cascade_triggered=result.cascade_triggered,
        cascade_model=result.cascade_model,
    )
