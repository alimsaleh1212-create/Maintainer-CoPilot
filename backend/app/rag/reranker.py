"""Async HTTP client for the model-server's cross-encoder rerank endpoint.

The actual BAAI/bge-reranker-base model lives in the ``model-server`` container
— it already has torch loaded for the DistilBERT classifier, so the reranker
is a free addition there and keeps the API container slim (no torch, no
transformers, no sentence-transformers in the API venv).

Failure mode: if model-server returns 503 (reranker_unavailable) or is
unreachable, ``rerank()`` returns an empty list. The hybrid retriever then
keeps its dense+sparse ordering and the call still succeeds — degradation,
not failure.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


class ModelServerReranker:
    """Thin async HTTP client around ``POST /rerank`` on model-server.

    Args:
        base_url: model-server base URL (defaults to Settings.model_server_base_url).
        timeout: per-request timeout in seconds.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        if base_url is None:
            base_url = get_settings().model_server_base_url
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def rerank(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Score (query, passage) pairs and return ``[(index, score), ...]`` desc.

        Args:
            query: User question used as the cross-encoder's left input.
            passages: Candidate texts. Each is scored against the query.
            top_k: Truncate to the top ``top_k`` results (None = return all).

        Returns:
            ``[(original_index, score), ...]`` sorted by score desc, or an
            empty list when the reranker is unavailable (caller should treat
            this as "skip the rerank step").
        """
        if not passages:
            return []

        payload: dict[str, object] = {"query": query, "passages": passages}
        if top_k is not None:
            payload["top_k"] = top_k

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base}/rerank", json=payload)
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # 503 = reranker_unavailable (model failed to load). Gracefully
            # fall back; do not raise — this is an enrichment, not core.
            if exc.response.status_code == 503:
                logger.warning("reranker.unavailable", detail=exc.response.text)
                return []
            logger.warning("reranker.http_error", status=exc.response.status_code)
            return []
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            logger.warning("reranker.network_error", error=str(exc))
            return []

        data = resp.json()
        return [(int(r["index"]), float(r["score"])) for r in data.get("results", [])]


@lru_cache(maxsize=1)
def get_reranker() -> ModelServerReranker:
    """Process-wide singleton — cheap to construct (httpx is created per call)."""
    return ModelServerReranker()
