"""Unit tests for ToolError domain model.

ToolError is returned to the LLM when a tool call fails, allowing the LLM
to decide whether to retry, escalate, or continue.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.domain.tool_error import ToolError


def test_tool_error_valid():
    """ToolError with all fields is valid."""
    error = ToolError(
        tool_name="classify",
        error="Model server timeout",
        retryable=True,
    )
    assert error.tool_name == "classify"
    assert error.error == "Model server timeout"
    assert error.retryable is True


def test_tool_error_retryable_default_false():
    """ToolError retryable defaults to False."""
    error = ToolError(
        tool_name="classify",
        error="Invalid input",
    )
    assert error.retryable is False


def test_tool_error_missing_required_fields():
    """ToolError requires tool_name and error."""
    with pytest.raises(PydanticValidationError):
        ToolError(tool_name="classify")  # type: ignore

    with pytest.raises(PydanticValidationError):
        ToolError(error="timeout")  # type: ignore


def test_tool_error_serialization():
    """ToolError serializes to JSON for LLM."""
    error = ToolError(
        tool_name="rag_search",
        error="Database connection lost",
        retryable=True,
    )
    data = error.model_dump()

    assert data == {
        "tool_name": "rag_search",
        "error": "Database connection lost",
        "retryable": True,
    }

    # Should be JSON-serializable
    json_str = error.model_dump_json()
    assert "rag_search" in json_str
    assert "Database connection lost" in json_str


def test_tool_error_strict_types():
    """ToolError enforces strict types with Pydantic v2."""
    # tool_name must be string
    with pytest.raises(PydanticValidationError):
        ToolError(tool_name=123, error="test")  # type: ignore

    # error must be string
    with pytest.raises(PydanticValidationError):
        ToolError(tool_name="classify", error=123)  # type: ignore


def test_tool_error_different_tools():
    """ToolError tracks which tool failed."""
    tools = ["classify", "rag_search", "summarize", "ner"]

    for tool in tools:
        error = ToolError(
            tool_name=tool,
            error=f"{tool} failed",
            retryable=False,
        )
        assert error.tool_name == tool


def test_tool_error_chatbot_integration_pattern():
    """ToolError pattern for chatbot loop recovery.

    When a tool call fails during chatbot loop:
    1. Catch ToolFailure (domain exception)
    2. Convert to ToolError (response for LLM)
    3. Return to LLM with tool name + error + retryable flag
    4. LLM decides next action
    """
    # Simulate a tool failure scenario
    tool_name = "classify"
    error_message = "Model server unreachable (3/3 attempts)"
    is_retryable = False

    # Convert to ToolError for LLM
    tool_error = ToolError(
        tool_name=tool_name,
        error=error_message,
        retryable=is_retryable,
    )

    # Verify LLM gets the information it needs
    assert tool_error.tool_name == tool_name
    assert tool_error.error == error_message
    assert tool_error.retryable is is_retryable
