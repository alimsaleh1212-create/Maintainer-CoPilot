"""Tool error response model for LLM tool-calling loop.

When a tool fails, the chatbot loop catches the exception and returns
a structured ToolError to the LLM. The LLM decides whether to retry,
ask the user, or continue with an alternative approach.

This is distinct from domain errors (AppError) which are caught at the
API boundary and converted to HTTP responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ToolError(BaseModel):
    """Tool failure response returned to LLM.

    The LLM receives this in the tool-calling loop and can decide
    whether to retry, escalate, or choose an alternative.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_name": "classify",
                "error": "Model server unreachable (attempt 3/3)",
                "retryable": False,
            }
        }
    )

    tool_name: str = Field(..., description="Name of the tool that failed")
    error: str = Field(..., description="Human-readable error message")
    retryable: bool = Field(
        default=False,
        description="Whether the LLM should retry this tool call",
    )
