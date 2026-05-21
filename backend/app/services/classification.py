"""Classification service: wraps the model-server classifier HTTP call.

Isolates the API boundary from classifier transport.  When the model-server
is unreachable, the service returns a graceful fallback result and logs a
warning — callers (chatbot, /classify route) stay 200.
"""

from __future__ import annotations

import time

import httpx
import structlog

from app.domain.classification import ClassificationResult
from app.domain.issue import ClassLabel

logger = structlog.get_logger(__name__)

# Fallback returned when the model-server is down.
_UNAVAILABLE: ClassificationResult = ClassificationResult(
    label="support",  # safest default: route as support, not a misclassified bug
    confidence=0.0,
    model_version="unavailable",
    latency_ms=0.0,
)


class ClassificationService:
    """Wraps the model-server /predict endpoint with graceful degradation.

    The service returns a ClassificationResult on every call.  When the
    model-server is unreachable the fallback result is returned and a warning
    is logged so callers can surface the degraded state without crashing.

    Args:
        client: Async HTTP client pre-configured with model-server base URL
                and timeout.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def classify(self, text: str, max_length: int = 512) -> ClassificationResult:
        """Classify issue text against the three-class taxonomy.

        Args:
            text: Raw issue text (title + body) — up to 4096 characters.
            max_length: Tokenizer max length forwarded to model-server.

        Returns:
            ClassificationResult with label, confidence, model_version, latency_ms.
            Returns the ``_UNAVAILABLE`` sentinel if the model-server is down,
            rather than raising — preserving chatbot liveness.
        """
        start = time.monotonic()
        try:
            response = await self._client.post(
                "/predict",
                json={"text": text, "max_length": max_length},
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.monotonic() - start) * 1000.0

            label_raw: str = data.get("label", "support")
            # Coerce to a valid ClassLabel; fall back to "support" on unknown value.
            label: ClassLabel = (
                label_raw if label_raw in ("bug", "feature", "support") else "support"
            )  # type: ignore[assignment]

            result = ClassificationResult(
                label=label,
                confidence=float(data.get("confidence", 0.0)),
                model_version=str(data.get("model_version", "unknown")),
                latency_ms=latency_ms,
            )
            logger.info(
                "classification.success",
                label=result.label,
                confidence=result.confidence,
                latency_ms=round(result.latency_ms, 1),
            )
            return result

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            logger.warning(
                "classification.model_server_unreachable",
                error=str(exc),
                fallback_label=_UNAVAILABLE.label,
            )
            return _UNAVAILABLE

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "classification.model_server_error",
                status_code=exc.response.status_code,
                error=str(exc),
                fallback_label=_UNAVAILABLE.label,
            )
            return _UNAVAILABLE

        except Exception as exc:
            logger.exception("classification.unexpected_error", error=str(exc))
            return _UNAVAILABLE
