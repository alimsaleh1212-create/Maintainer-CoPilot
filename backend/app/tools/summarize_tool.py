"""LLM-callable tool: summarize a GitHub issue or conversation."""

from __future__ import annotations

import structlog

from app.domain.errors import ToolFailure

logger = structlog.get_logger(__name__)

SUMMARIZE_TOOL_SCHEMA: dict[str, object] = {
    "name": "summarize_issue",
    "description": (
        "Generate a concise 2-3 sentence technical summary of a GitHub issue "
        "or conversation thread. Focuses on: what the problem is, what was tried, "
        "and the outcome or current status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Issue or conversation text to summarize.",
            }
        },
        "required": ["text"],
    },
}


async def summarize_issue(text: str, llm_client: object) -> str:
    """Summarize an issue or conversation using the LLM.

    Args:
        text: Raw text to summarize.
        llm_client: LLM client instance.

    Returns:
        2-3 sentence summary string.

    Raises:
        ToolFailure: On LLM or model errors.
    """
    try:
        from app.ml.summarizer import Summarizer

        summarizer = Summarizer(llm_client=llm_client)
        summary = await summarizer.summarize(text)
        logger.info("summarize_tool.success", summary_length=len(summary))
        return summary
    except Exception as exc:
        logger.warning("summarize_tool.failed", error=str(exc))
        raise ToolFailure(
            error="summarizer_unavailable", retryable=False, detail=str(exc)
        ) from exc
