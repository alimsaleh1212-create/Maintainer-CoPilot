"""Cross-encoder reranker for hybrid retrieval.

Wraps a local sentence-transformers CrossEncoder (BAAI/bge-reranker-base by
default). Inference runs in a thread (the model is sync / CPU-bound) so the
event loop stays unblocked. The model loads once at startup via the
``get_reranker()`` lazy-singleton — heavy CPU work belongs in lifespan, not
per-request.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """Thin async wrapper around sentence-transformers CrossEncoder.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            # Import lazily so the heavy torch/transformers dep doesn't get
            # pulled in on module load (matters for unit tests / scripts).
            from sentence_transformers import CrossEncoder

            logger.info("reranker.loading", model=self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("reranker.loaded", model=self.model_name)
        return self._model

    async def rerank(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Score (query, passage) pairs and return (index, score) sorted desc.

        Args:
            query: User question used as the cross-encoder's left input.
            passages: Candidate texts. Each is scored against the query.
            top_k: Truncate to the top ``top_k`` results (None = return all).

        Returns:
            List of ``(original_index, score)`` tuples sorted by score desc.
        """
        if not passages:
            return []

        model = self._load()
        pairs = [(query, p) for p in passages]
        # Cross-encoder inference is sync + CPU heavy. Push it off-thread so
        # we don't block the event loop for ~50-200ms per call.
        scores = await asyncio.to_thread(model.predict, pairs)

        indexed = sorted(
            enumerate(float(s) for s in scores), key=lambda t: t[1], reverse=True
        )
        return indexed[:top_k] if top_k else indexed


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Return the process-wide reranker singleton.

    Loaded once in the FastAPI lifespan via a warm-up call to ``_load()``.
    """
    return CrossEncoderReranker()
