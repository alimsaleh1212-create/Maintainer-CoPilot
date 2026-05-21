"""LLM-callable tool: hybrid RAG search over project docs and resolved issues."""

from __future__ import annotations

import structlog

from app.domain.errors import ToolFailure

logger = structlog.get_logger(__name__)

RAG_TOOL_SCHEMA: dict[str, object] = {
    "name": "rag_search",
    "description": (
        "Search the project documentation and resolved GitHub issues for context "
        "relevant to the user's question. Returns top matching passages with sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query — what you want to find information about.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 5, max 10).",
            },
        },
        "required": ["query"],
    },
}


async def rag_search(
    query: str,
    rag_service: object,
    top_k: int = 5,
) -> list[dict[str, str]]:
    """Search the RAG corpus for relevant context.

    Args:
        query: Natural language search query.
        rag_service: RAG service instance.
        top_k: Number of results (capped at 10).

    Returns:
        List of dicts with keys: content, source, score.

    Raises:
        ToolFailure: If RAG retrieval fails.
    """
    from app.services.rag import RAGService

    service: RAGService = rag_service  # type: ignore[assignment]
    top_k = min(top_k, 10)
    try:
        results = await service.search(query=query, top_k=top_k)
        logger.info("rag_tool.success", query=query[:80], result_count=len(results))
        return results
    except Exception as exc:
        logger.warning("rag_tool.failed", query=query[:80], error=str(exc))
        raise ToolFailure(error="rag_unavailable", retryable=True, detail=str(exc)) from exc
