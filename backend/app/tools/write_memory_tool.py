"""LLM-callable tool: explicitly write an episodic memory for the user.

Every call produces an audit_log row. The LLM must call this tool explicitly
— no auto-writes. This enforces the "no auto-write" rule from CLAUDE.md.
"""

from __future__ import annotations

import uuid

import structlog

from app.domain.errors import ToolFailure
from app.infra.redaction import redact_text

logger = structlog.get_logger(__name__)

WRITE_MEMORY_TOOL_SCHEMA: dict[str, object] = {
    "name": "write_memory",
    "description": (
        "Store an important fact or summary in the user's long-term memory so it "
        "can be recalled in future conversations. Only write memories for genuinely "
        "important, durable facts — not ephemeral conversational context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "The fact or summary to remember. "
                    "Should be self-contained and useful out of context."
                ),
            }
        },
        "required": ["content"],
    },
}


async def write_memory(
    content: str,
    user_id: uuid.UUID,
    memory_service: object,
) -> dict[str, str]:
    """Persist an episodic memory for the user (explicit LLM call only).

    Redacts content before storing. Every call produces an audit_log row.

    Args:
        content: Text to remember (will be embedded + stored).
        user_id: Owner of the memory.
        memory_service: MemoryService instance.

    Returns:
        Dict with keys: memory_id, status.

    Raises:
        ToolFailure: If the memory write fails.
    """
    from app.services.memory import MemoryService

    service: MemoryService = memory_service  # type: ignore[assignment]
    safe_content = redact_text(content)
    try:
        memory = await service.save_memory(user_id=user_id, summary=safe_content)
        logger.info(
            "write_memory_tool.success",
            user_id=str(user_id),
            memory_id=str(memory.id),
        )
        return {"memory_id": str(memory.id), "status": "stored"}
    except Exception as exc:
        logger.warning("write_memory_tool.failed", user_id=str(user_id), error=str(exc))
        raise ToolFailure(
            error="memory_write_failed", retryable=False, detail=str(exc)
        ) from exc
