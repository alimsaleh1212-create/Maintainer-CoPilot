"""Domain model for classification results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.issue import CLASS_NAMES


class ClassificationResult(BaseModel):
    """Result returned by the classifier tool and /classify endpoint."""

    label: str = Field(..., description="One of: bug, feature, support")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    latency_ms: float = Field(..., ge=0.0)

    @classmethod
    def validate_label(cls, label: str) -> str:
        if label not in CLASS_NAMES:
            raise ValueError(f"label must be one of {CLASS_NAMES}, got {label!r}")
        return label
