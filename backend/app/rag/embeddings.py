"""Embedding model for RAG: BAAI/bge-small-en-v1.5."""

from __future__ import annotations

import asyncio
from functools import lru_cache

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

# Model constants
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small produces 384-dim vectors
BATCH_SIZE = 32


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model.

    Uses BAAI/bge-small-en-v1.5: fast, CPU-friendly, good for semantic search.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def load(self) -> None:
        """Load model (called during lifespan startup)."""
        if self.model is None:
            logger.info("embeddings.loading", model=self.model_name)
            self.model = SentenceTransformer(self.model_name)
            logger.info("embeddings.loaded", dim=EMBEDDING_DIM)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts asynchronously.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (each is list[float])
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
        )

        # Convert numpy to list[float] for JSON serialization
        return [emb.tolist() for emb in embeddings]

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


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Get singleton embedding model instance."""
    return EmbeddingModel()
