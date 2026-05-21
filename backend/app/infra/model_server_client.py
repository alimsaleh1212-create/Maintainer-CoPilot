"""Model-server HTTP client — async proxy to the DistilBERT inference service.

The model-server runs as a separate FastAPI process (``http://model-server:8001``).
This client communicates with it over HTTP, keeping the heavy PyTorch runtime
out of the main API process.

Rules (CLAUDE.md):
- ``httpx.AsyncClient`` only — no ``requests``.
- Retry on network errors (not on 4xx) with exponential back-off.
- Timeout: 10 s per request.
- ``ToolFailure`` (not HTTP 500) is raised when the model server is down so
  the chatbot can continue degraded.
- structlog everywhere; no ``print()``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from app.domain.classification import ClassificationResult
from app.domain.errors import ToolFailure
from app.domain.issue import ClassLabel

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Retry constants — intentionally kept simple without tenacity so we don't
# need to add it as a dep yet.  Replace with tenacity once uv add tenacity.
# ---------------------------------------------------------------------------
_MAX_ATTEMPTS = 3
_BASE_BACKOFF_S = 0.5  # 0.5 s, 1 s, 2 s
_TIMEOUT_S = 10.0

_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


async def _retry_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    """Execute an httpx request with exponential retry on network errors.

    Args:
        client: The shared ``httpx.AsyncClient``.
        method: HTTP method string (``"GET"``, ``"POST"``, …).
        url: Full URL to request.
        **kwargs: Passed directly to ``client.request``.

    Returns:
        The successful ``httpx.Response``.

    Raises:
        ToolFailure: after all retry attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = await client.request(method, url, **kwargs)
            return response
        except _NETWORK_ERRORS as exc:
            last_exc = exc
            wait = _BASE_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                "model_server.retry",
                attempt=attempt,
                max_attempts=_MAX_ATTEMPTS,
                wait_s=wait,
                error=str(exc),
            )
            if attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(wait)

    raise ToolFailure(
        f"Model server unreachable after {_MAX_ATTEMPTS} attempts: {last_exc}",
        retryable=False,
    )


@dataclass
class ModelServerClient:
    """Async HTTP client for the DistilBERT model-server inference API.

    Constructed in ``lifespan()`` and injected via ``Depends()``.

    Args:
        base_url: Base URL of the model server (e.g. ``"http://model-server:8001"``).
        http_client: Shared ``httpx.AsyncClient`` (lifecycle managed by caller).
    """

    base_url: str
    http_client: httpx.AsyncClient = field(default_factory=lambda: httpx.AsyncClient(timeout=_TIMEOUT_S))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Check whether the model server is healthy.

        Returns:
            ``True`` if the server responds with HTTP 200, else ``False``.
        """
        try:
            response = await self.http_client.get(
                f"{self.base_url}/healthz",
                timeout=_TIMEOUT_S,
            )
            ok = response.status_code == 200
            logger.debug("model_server.health", status=response.status_code, ok=ok)
            return ok
        except _NETWORK_ERRORS as exc:
            logger.warning("model_server.health_failed", error=str(exc))
            return False

    async def predict(self, text: str) -> ClassificationResult:
        """Classify *text* into one of {bug, feature, support}.

        Args:
            text: The raw issue body / title to classify.

        Returns:
            ``ClassificationResult`` with label, confidence, version, latency.

        Raises:
            ToolFailure: if the model server is unreachable or returns an error.
        """
        t0 = time.perf_counter()
        try:
            response = await _retry_request(
                self.http_client,
                "POST",
                f"{self.base_url}/predict",
                json={"text": text},
                timeout=_TIMEOUT_S,
            )
        except ToolFailure:
            raise

        if response.status_code != 200:
            raise ToolFailure(
                f"Model server /predict returned HTTP {response.status_code}: {response.text}",
                retryable=False,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            payload: dict[str, Any] = response.json()
        except Exception as exc:
            raise ToolFailure(
                f"Model server /predict returned invalid JSON: {exc}",
                retryable=False,
            ) from exc

        _valid_labels: tuple[ClassLabel, ...] = ("bug", "feature", "support")
        try:
            raw_label = str(payload["label"])
            if raw_label not in _valid_labels:
                raise ValueError(f"unknown label {raw_label!r}")
            result = ClassificationResult(
                label=raw_label,  # type: ignore[arg-type]
                confidence=float(payload["confidence"]),
                model_version=str(payload.get("model_version", "unknown")),
                latency_ms=latency_ms,
            )
        except (KeyError, ValueError) as exc:
            raise ToolFailure(
                f"Model server /predict response missing or invalid fields: {exc}",
                retryable=False,
            ) from exc

        logger.info(
            "model_server.predict",
            label=result.label,
            confidence=round(result.confidence, 4),
            latency_ms=round(latency_ms, 1),
        )
        return result

    async def ner(self, text: str) -> list[dict[str, Any]]:
        """Run named-entity recognition on *text*.

        Args:
            text: Input text for NER.

        Returns:
            List of entity dicts, e.g. ``[{"text": "MONAI", "label": "ORG", ...}]``.

        Raises:
            ToolFailure: if the model server is unreachable or returns an error.
        """
        response = await _retry_request(
            self.http_client,
            "POST",
            f"{self.base_url}/ner",
            json={"text": text},
            timeout=_TIMEOUT_S,
        )
        if response.status_code != 200:
            raise ToolFailure(
                f"Model server /ner returned HTTP {response.status_code}: {response.text}",
                retryable=False,
            )

        try:
            entities: list[dict[str, Any]] = response.json()
        except Exception as exc:
            raise ToolFailure(
                f"Model server /ner returned invalid JSON: {exc}",
                retryable=False,
            ) from exc

        logger.debug("model_server.ner", entity_count=len(entities))
        return entities

    async def summarize(self, text: str) -> str:
        """Summarise *text* using the model server's summarisation endpoint.

        Args:
            text: Input text to summarise.

        Returns:
            Summary string.

        Raises:
            ToolFailure: if the model server is unreachable or returns an error.
        """
        response = await _retry_request(
            self.http_client,
            "POST",
            f"{self.base_url}/summarize",
            json={"text": text},
            timeout=_TIMEOUT_S,
        )
        if response.status_code != 200:
            raise ToolFailure(
                f"Model server /summarize returned HTTP {response.status_code}: {response.text}",
                retryable=False,
            )

        try:
            payload = response.json()
            summary: str = str(payload["summary"])
        except (KeyError, Exception) as exc:
            raise ToolFailure(
                f"Model server /summarize returned invalid response: {exc}",
                retryable=False,
            ) from exc

        logger.debug("model_server.summarize", summary_len=len(summary))
        return summary
