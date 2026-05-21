"""Classification service: wraps the model-server classifier with cascade fallback.

Primary path: call the model-server (DistilBERT).
Cascade path: if DistilBERT confidence < ``cascade_threshold``, re-classify
              with the LLM (Gemini).  This keeps high-confidence inferences
              fast and cheap while routing uncertain cases to the stronger model.

Both paths return a ``ClassificationResult``; ``cascade_triggered`` tells callers
which path was taken.  The service never raises — on any failure it returns the
``_UNAVAILABLE`` sentinel so callers (chatbot, /classify route) stay 200.
"""

from __future__ import annotations

import json
import re
import time

import httpx
import structlog

from app.domain.classification import ClassificationResult
from app.domain.issue import ClassLabel
from app.infra.llm.base import LLMClient

logger = structlog.get_logger(__name__)

# Returned when both the model-server and the LLM are unavailable.
_UNAVAILABLE: ClassificationResult = ClassificationResult(
    label="support",  # safest default: route as support, not a misclassified bug
    confidence=0.0,
    model_version="unavailable",
    latency_ms=0.0,
)

_VALID_LABELS: frozenset[str] = frozenset({"bug", "feature", "support"})

_LLM_SYSTEM_PROMPT = """\
You are a GitHub issue classifier for open-source projects.
Classify the issue into exactly one of these three categories:
- bug: a defect, regression, crash, unexpected behaviour, or error
- feature: a request for new functionality, enhancement, or improvement
- support: a question, how-to, documentation gap, or general help request

Reply with a JSON object and nothing else:
{"label": "<bug|feature|support>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}
"""


def _parse_llm_label(text: str) -> tuple[ClassLabel, float]:
    """Extract (label, confidence) from LLM JSON response.

    Falls back to a regex scan if the model wraps JSON in markdown fences.

    Args:
        text: Raw LLM response text.

    Returns:
        Validated (label, confidence) tuple.  Defaults to ("support", 0.5) on
        any parse failure so the cascade never hard-errors.
    """
    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        data = json.loads(cleaned)
        raw_label = str(data.get("label", "support")).lower().strip()
        label: ClassLabel = raw_label if raw_label in _VALID_LABELS else "support"  # type: ignore[assignment]
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return label, confidence
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("classification.llm_parse_failed", raw_text=text[:200])
        return "support", 0.5


class ClassificationService:
    """Wraps the model-server /predict endpoint with cascade LLM fallback.

    Primary path: DistilBERT via model-server (fast, free).
    Cascade path: LLM (Gemini) when DistilBERT confidence < ``cascade_threshold``.

    The service returns a ClassificationResult on every call and never raises,
    preserving chatbot liveness even when both paths fail.

    Args:
        client: Async HTTP client pre-configured with model-server base URL and timeout.
        llm_client: LLM client used for cascade (typically GeminiClient).
        cascade_threshold: DistilBERT confidence below which the LLM is used.
            Set to 0.0 to disable cascade entirely.
        llm_model_name: Human-readable name stored in ``cascade_model`` field.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        llm_client: LLMClient | None = None,
        cascade_threshold: float = 0.75,
        llm_model_name: str = "gemini",
    ) -> None:
        self._client = client
        self._llm_client = llm_client
        self._cascade_threshold = cascade_threshold
        self._llm_model_name = llm_model_name

    async def classify(self, text: str, max_length: int = 512) -> ClassificationResult:
        """Classify issue text, cascading to LLM if DL confidence is low.

        Args:
            text: Raw issue text (title + body) — up to 4096 characters.
            max_length: Tokenizer max length forwarded to model-server.

        Returns:
            ClassificationResult.  ``cascade_triggered=True`` when the LLM
            path was taken.  Returns the ``_UNAVAILABLE`` sentinel if all paths
            fail, rather than raising.
        """
        dl_result = await self._classify_dl(text, max_length)

        # If DL succeeded and confidence is above threshold, return immediately.
        if (
            dl_result.model_version != "unavailable"
            and dl_result.confidence >= self._cascade_threshold
        ):
            return dl_result

        # Cascade: DL was unavailable OR confidence is below threshold.
        if self._llm_client is None or self._cascade_threshold == 0.0:
            # No LLM configured or cascade disabled — return DL result as-is.
            return dl_result

        reason = (
            "model_server_unavailable"
            if dl_result.model_version == "unavailable"
            else f"confidence={dl_result.confidence:.3f}_below_threshold={self._cascade_threshold}"
        )
        logger.info(
            "classification.cascade_triggered",
            reason=reason,
            dl_confidence=dl_result.confidence,
            threshold=self._cascade_threshold,
        )

        llm_result = await self._classify_llm(text, dl_result)
        return llm_result

    async def _classify_dl(self, text: str, max_length: int) -> ClassificationResult:
        """Call the model-server and return a ClassificationResult.

        Returns the ``_UNAVAILABLE`` sentinel on any transport or server error.
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

            raw_label: str = data.get("label", "support")
            label: ClassLabel = raw_label if raw_label in _VALID_LABELS else "support"  # type: ignore[assignment]

            result = ClassificationResult(
                label=label,
                confidence=float(data.get("confidence", 0.0)),
                model_version=str(data.get("model_version", "unknown")),
                latency_ms=latency_ms,
            )
            logger.info(
                "classification.dl_success",
                label=result.label,
                confidence=result.confidence,
                latency_ms=round(result.latency_ms, 1),
            )
            return result

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            logger.warning("classification.model_server_unreachable", error=str(exc))
            return _UNAVAILABLE

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "classification.model_server_error",
                status_code=exc.response.status_code,
                error=str(exc),
            )
            return _UNAVAILABLE

        except Exception as exc:
            logger.exception("classification.dl_unexpected_error", error=str(exc))
            return _UNAVAILABLE

    async def _classify_llm(
        self, text: str, dl_result: ClassificationResult
    ) -> ClassificationResult:
        """Classify using the LLM and return a ClassificationResult.

        Args:
            text: Issue text to classify.
            dl_result: DL result (used for latency baseline + fallback if LLM fails).

        Returns:
            ClassificationResult with ``cascade_triggered=True`` and
            ``cascade_model`` set.  Falls back to the DL result on LLM failure.
        """
        if self._llm_client is None:
            return dl_result
        start = time.monotonic()
        try:
            # Truncate to avoid hitting LLM token limits.
            truncated_text = text[:3000]
            reply = await self._llm_client.chat(
                messages=[{"role": "user", "content": truncated_text}],
                system_prompt=_LLM_SYSTEM_PROMPT,
            )
            llm_latency_ms = (time.monotonic() - start) * 1000.0
            label, confidence = _parse_llm_label(reply)

            result = ClassificationResult(
                label=label,
                confidence=confidence,
                model_version=f"cascade:{self._llm_model_name}",
                latency_ms=dl_result.latency_ms + llm_latency_ms,
                cascade_triggered=True,
                cascade_model=self._llm_model_name,
            )
            logger.info(
                "classification.llm_success",
                label=result.label,
                confidence=result.confidence,
                llm_latency_ms=round(llm_latency_ms, 1),
                cascade_model=self._llm_model_name,
            )
            return result

        except Exception as exc:
            logger.exception("classification.llm_cascade_failed", error=str(exc))
            # Return DL result as best-effort rather than the hard-fail sentinel.
            return dl_result
