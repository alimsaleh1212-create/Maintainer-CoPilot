"""Domain model for classification results — API + tool boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.issue import ClassLabel


class ClassificationResult(BaseModel):
    """Result returned by the classifier tool and /classify endpoint."""

    label: ClassLabel = Field(..., description="One of: bug, feature, support")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    latency_ms: float = Field(..., ge=0.0)
    cascade_triggered: bool = Field(
        default=False,
        description="True when DistilBERT confidence was below threshold and the LLM was used instead",
    )
    cascade_model: str | None = Field(
        default=None,
        description="Name of the LLM used for cascade classification, or None if cascade was not triggered",
    )
