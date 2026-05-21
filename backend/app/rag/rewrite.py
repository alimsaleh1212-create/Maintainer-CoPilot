"""Query rewriting: multi-query expansion for RAG retrieval."""

from __future__ import annotations

import httpx
import structlog

logger = structlog.get_logger(__name__)


class MultiQueryExpander:
    """Expand a single query into multiple variations.

    Strategy: Template-based expansion for common patterns + LLM fallback.

    Examples:
    - "GPU memory error" → ["GPU memory error", "CUDA OOM", "out of memory GPU", ...]
    - "how to use transforms" → ["use transforms", "transforms tutorial", "apply transforms", ...]
    """

    # Template-based expansions for common MONAI/ML terms
    EXPANSIONS = {
        "gpu|cuda|gpu memory|device memory": [
            "GPU memory",
            "CUDA out of memory",
            "OOM",
            "device memory",
            "GPU memory overflow",
            "insufficient GPU memory",
        ],
        "error|bug|issue|problem": [
            "error",
            "fails",
            "doesn't work",
            "broken",
            "crash",
            "exception",
        ],
        "transform|augment|preprocessing": [
            "transform",
            "augmentation",
            "preprocessing",
            "pipeline",
            "data pipeline",
            "composition",
        ],
        "model|network|architecture": [
            "model",
            "neural network",
            "architecture",
            "pretrained model",
            "network design",
        ],
        "train|training|fine.?tune|finetune": [
            "training",
            "fine-tuning",
            "train",
            "fit",
            "model training",
            "learning",
        ],
        "inference|predict|forward|evaluation": [
            "inference",
            "prediction",
            "forward pass",
            "evaluation",
            "testing",
        ],
        "loss|metric|f1|accuracy|dice": [
            "loss function",
            "metric",
            "F1 score",
            "accuracy",
            "Dice coefficient",
            "evaluation metric",
        ],
    }

    def __init__(self, llm_fallback: bool = True, num_variations: int = 4):
        """Initialize multi-query expander.

        Args:
            llm_fallback: Use LLM for complex query rewrites
            num_variations: Target number of query variations
        """
        self.llm_fallback = llm_fallback
        self.num_variations = num_variations

    async def expand(self, query: str, gemini_api_key: str | None = None) -> list[str]:
        """Expand query into multiple variations.

        Args:
            query: Original query
            gemini_api_key: Optional API key for LLM fallback

        Returns:
            List of query variations (includes original)
        """
        variations = [query]

        # Try template-based expansion first
        template_variants = self._template_expand(query)
        variations.extend(template_variants)

        # If we have room and LLM available, use LLM for complex queries
        if self.llm_fallback and len(variations) < self.num_variations and gemini_api_key:
            try:
                llm_variants = await self._llm_expand(query, gemini_api_key)
                variations.extend(llm_variants)
            except Exception as e:
                logger.warning("rewrite.llm_fallback_failed", error=str(e))

        # Deduplicate and limit
        variations = list(dict.fromkeys(variations))[: self.num_variations]
        return variations

    def _template_expand(self, query: str) -> list[str]:
        """Template-based expansion for common terms.

        Returns:
            List of query variants (excludes original)
        """
        variants = []

        for pattern, expansions in self.EXPANSIONS.items():
            import re

            if re.search(pattern, query, re.IGNORECASE):
                # Pick 1-2 expansions and substitute
                for expansion in expansions[:2]:
                    variant = re.sub(pattern, expansion, query, flags=re.IGNORECASE)
                    if variant != query:
                        variants.append(variant)

        # Add synonym patterns (common MONAI-specific)
        if "normalize" in query.lower():
            variants.extend(
                [
                    query.replace("Normalize", "normalization"),
                    query.replace("normalize", "intensity scaling"),
                ]
            )

        if "batch" in query.lower():
            variants.extend(
                [
                    query.replace("batch", "mini-batch"),
                    query.replace("batch", "parallel"),
                ]
            )

        return variants[:3]  # Limit to avoid explosion

    async def _llm_expand(self, query: str, api_key: str) -> list[str]:
        """LLM-based query expansion (lightweight).

        Uses Gemini to generate 2-3 alternative phrasings.

        Args:
            query: Original query
            api_key: Gemini API key

        Returns:
            List of LLM-generated variants
        """
        prompt = f"""Generate 2-3 alternative phrasings of this question that maintain the same intent but use different wording:

Question: {query}

Alternative phrasings (one per line, no numbering):"""

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": 150},
                },
                timeout=10.0,
            )
            resp.raise_for_status()

            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]

            # Parse lines as variants
            variants = [line.strip() for line in content.split("\n") if line.strip()]
            return variants[:2]  # Limit to 2


class NoOpExpander:
    """Fallback: no expansion (just return original query)."""

    async def expand(self, query: str, gemini_api_key: str | None = None) -> list[str]:
        """Return original query as-is."""
        return [query]
