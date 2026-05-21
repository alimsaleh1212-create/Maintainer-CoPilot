"""Classification endpoint: POST /classify"""

from __future__ import annotations

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/classify", tags=["classification"])


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Issue text to classify")
    max_length: int = Field(default=512, ge=128, le=1024, description="Tokenizer max length")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted label: bug, feature, or support")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    model_version: str = Field(..., description="Model version from model_card.json")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")


async def get_model_server_client() -> httpx.AsyncClient:
    """Return async HTTP client for model-server."""
    return httpx.AsyncClient(base_url="http://model-server:8001", timeout=10.0)


@router.post("", response_model=ClassifyResponse)
async def classify(
    req: ClassifyRequest,
    client: httpx.AsyncClient = Depends(get_model_server_client),
) -> ClassifyResponse:
    """Classify an issue by calling the model-server.

    Args:
        req: ClassifyRequest with issue text
        client: HTTP client to model-server

    Returns:
        ClassifyResponse with label, confidence, model_version, latency_ms
    """
    try:
        resp = await client.post(
            "/predict",
            json={"text": req.text, "max_length": req.max_length},
        )
        resp.raise_for_status()
        return ClassifyResponse(**resp.json())

    except httpx.TimeoutException as e:
        logger.error("classify.model_server_timeout", error=str(e))
        raise HTTPException(status_code=503, detail="Model server timeout") from e

    except httpx.ConnectError as e:
        logger.error("classify.model_server_unreachable", error=str(e))
        raise HTTPException(status_code=503, detail="Model server unreachable") from e

    except Exception as e:
        logger.exception("classify.failed", error=str(e))
        raise HTTPException(status_code=500, detail="Classification failed") from e
