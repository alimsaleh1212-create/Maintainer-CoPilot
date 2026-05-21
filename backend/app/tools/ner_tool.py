"""LLM-callable tool: extract code-shaped entities from GitHub issue text."""

from __future__ import annotations

import structlog

from app.domain.errors import ToolFailure

logger = structlog.get_logger(__name__)

NER_TOOL_SCHEMA: dict[str, object] = {
    "name": "extract_entities",
    "description": (
        "Extract code-shaped named entities from GitHub issue text. "
        "Returns structured list of entities with type labels: "
        "FunctionName, ClassName, FilePath, ErrorType, PackageName, VersionNumber."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Issue text to extract entities from.",
            }
        },
        "required": ["text"],
    },
}


async def extract_entities(text: str) -> list[dict[str, str]]:
    """Extract named entities from issue text using the NER module.

    Args:
        text: Raw issue text.

    Returns:
        List of dicts with keys: text, label, start, end.

    Raises:
        ToolFailure: On unexpected errors.
    """
    try:
        from app.ml.ner import extract_entities as _extract

        entities = await _extract(text)
        logger.info("ner_tool.success", entity_count=len(entities))
        return entities
    except Exception as exc:
        logger.warning("ner_tool.failed", error=str(exc))
        raise ToolFailure(error="ner_unavailable", retryable=False, detail=str(exc)) from exc
