"""Embedding model for RAG: calls Ollama's BAAI/bge-small-en-v1.5 endpoint."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Model constants
MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIM = 768  # nomic-embed-text produces 768-dim vectors
OLLAMA_HOST = "http://ollama:11434"
# nomic-embed-text has a 2048-token context. Code/markdown tokenizes denser
# than prose (~1 token per 2-3 chars vs 1 per 4), so cap conservatively at
# 3000 chars to stay under the limit and avoid Ollama 400s.
MAX_INPUT_CHARS = 3000


class EmbeddingModel:
    """Calls Ollama's embedding endpoint for BAAI/bge-small-en-v1.5.

    Uses HTTP calls to Ollama service: reduces API image size by removing
    torch/sentence-transformers dependencies. Network latency negligible
    for chatbot workloads (embeddings ~50ms vs LLM calls ~1-3s).
    """

    def __init__(self, model_name: str = MODEL_NAME, ollama_host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.ollama_host = ollama_host
        # 180s — large batches (up to EMBED_BATCH=32) on CPU Ollama can take
        # 30-60s. Query-time embeds use a single string and finish in <1s.
        self.client = httpx.AsyncClient(timeout=180.0)

    async def ensure_model_pulled(self) -> None:
        """Pull model from Ollama registry if not present."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.ollama_host}/api/pull",
                    json={"name": self.model_name},
                    timeout=300.0,  # Long timeout for model pull
                )
                resp.raise_for_status()
                logger.info("embeddings.model_pulled", model=self.model_name)
        except Exception as e:
            logger.warning(
                "embeddings.pull_failed",
                model=self.model_name,
                error=str(e),
                hint="Model may already be present in Ollama",
            )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama endpoint.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (each is list[float])
        """
        # Defensive truncate — model context is 2048 tokens, Ollama 400's
        # on inputs >~10K chars in practice.
        safe_texts = [t[:MAX_INPUT_CHARS] if t else " " for t in texts]
        try:
            # Ollama /api/embed accepts a list for batch embedding
            resp = await self.client.post(
                f"{self.ollama_host}/api/embed",
                json={"model": self.model_name, "input": safe_texts},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            # Response: {"model": ..., "embeddings": [[float, ...]]}
            embeddings: list[list[float]] = data["embeddings"]
            return embeddings
        except Exception as e:
            logger.error(
                "embeddings.embed_batch_failed",
                count=len(texts),
                error=str(e),
            )
            raise

    async def embed(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list[float]
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    def get_dimension(self) -> int:
        """Return embedding dimension (384 for bge-small)."""
        return EMBEDDING_DIM

    async def close(self) -> None:
        """Close HTTP client (called during lifespan shutdown)."""
        await self.client.aclose()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Get singleton embedding model instance."""
    return EmbeddingModel()
