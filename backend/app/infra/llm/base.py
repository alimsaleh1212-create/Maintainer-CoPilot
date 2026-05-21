"""LLM client protocol and shared data types.

Defines the ``LLMClient`` Protocol that all concrete LLM adapters
(Gemini, Ollama, …) must satisfy.  The protocol is structural, so adapters
do not need to inherit from it — ``isinstance`` checks use
``runtime_checkable`` only in tests.

Rules (CLAUDE.md):
- Type hints required on every signature.
- Pydantic at every external boundary; dataclasses for internal value objects.
- No secrets, no logging here — adapters handle that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable


class LLMMessage(TypedDict, total=False):
    """A single message in an LLM conversation history.

    Attributes:
        role: One of ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``.
        content: Text content of the message (required for most roles).
        tool_calls: List of function calls emitted by the model (assistant role).
        tool_call_id: ID of the tool call this result is responding to (tool role).
        tool_name: Name of the tool that produced this result (tool role).
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    tool_name: str


@dataclass
class ToolCall:
    """A single function-call request from the LLM.

    Attributes:
        name: Name of the function to invoke (matches a registered tool).
        arguments: Parsed JSON arguments as a Python dict.
        id: Optional call ID returned by the model (used for result routing).
    """

    name: str
    arguments: dict[str, Any]
    id: str = field(default="")


@runtime_checkable
class LLMClient(Protocol):
    """Structural protocol for all LLM adapters.

    Concrete adapters (``GeminiClient``, ``OllamaClient``) implement these
    methods.  The chatbot service uses this protocol as its type annotation so
    that tests can inject fakes without monkey-patching imports.
    """

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a plain-text reply to *messages*.

        Args:
            messages: Conversation history in extended format (user/assistant/tool roles).
            system_prompt: Optional system instruction prepended to context.
            tools: Optional tool/function schemas made available to the model.

        Returns:
            The model's text reply.
        """
        ...

    async def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str = "",
    ) -> tuple[str, list[ToolCall]]:
        """Generate a reply that may include one or more tool calls.

        Args:
            messages: Conversation history in extended format (may include tool results).
            tools: Tool/function schemas the model may call.
            system_prompt: Optional system instruction.

        Returns:
            A 2-tuple of:
            - Text content from the model (may be empty string if it only
              emitted tool calls).
            - List of ``ToolCall`` objects requested by the model (may be
              empty if the model chose not to call any tool).
        """
        ...

    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*.

        Args:
            text: Input text to embed.

        Returns:
            A list of floats representing the embedding.
        """
        ...
