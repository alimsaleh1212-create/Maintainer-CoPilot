"""LLM-callable tool: classify a GitHub issue into bug / feature / support."""

from __future__ import annotations

import time

import structlog

from app.domain.classification import ClassificationResult
from app.domain.errors import ToolFailure

logger = structlog.get_logger(__name__)

# Tool schema for Gemini function calling
CLASSIFY_TOOL_SCHEMA: dict[str, object] = {
    "name": "classify_issue",
    "description": (
        "Classify a GitHub issue as bug, feature, or support. "
        "Returns the predicted label and confidence score."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The issue title and body text to classify.",
            }
        },
        "required": ["text"],
    },
}


async def classify_issue(
    text: str,
    model_server_client: object,
) -> ClassificationResult:
    """Classify a GitHub issue using the fine-tuned DistilBERT model.

    Args:
        text: Issue title + body text.
        model_server_client: HTTP client to the model-server inference service.

    Returns:
        ClassificationResult with label, confidence, model_version, latency_ms.

    Raises:
        ToolFailure: If the model server is unreachable (retryable=True).
    """
    from app.infra.model_server_client import ModelServerClient

    client: ModelServerClient = model_server_client  # type: ignore[assignment]
    t0 = time.monotonic()
    try:
        result = await client.predict(text)
        logger.info(
            "classify_tool.success",
            label=result.label,
            confidence=result.confidence,
            latency_ms=round((time.monotonic() - t0) * 1000, 1),
        )
        return result
    except Exception as exc:
        logger.warning("classify_tool.failed", error=str(exc))
        raise ToolFailure(
            error="classification_unavailable",
            retryable=True,
            detail=str(exc),
        ) from exc
