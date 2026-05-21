"""Gemini LLM client — uses the REST API directly via httpx.

No gRPC SDK needed — the Gemini API has a clean REST interface that is
lighter and avoids the ~200MB grpcio wheel.

Endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from app.infra.llm.base import LLMClient, LLMMessage, ToolCall  # noqa: F401

logger = structlog.get_logger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiClient:
    """Gemini LLM client using the REST API directly (no gRPC SDK).

    Implements the LLMClient protocol. Embeddings are delegated to Ollama
    because the Gemini embedding API requires a paid plan in many regions.

    Args:
        api_key: Gemini API key (from Vault).
        model: Primary model name (default: gemini-2.0-flash).
        ollama_host: Ollama host for embeddings.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        ollama_host: str = "http://ollama:11434",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._ollama_host = ollama_host
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _url(self, method: str) -> str:
        return f"{_GEMINI_BASE}/models/{self._model}:{method}?key={self._api_key}"

    def _messages_to_gemini(
        self, messages: list[LLMMessage], system_prompt: str | None
    ) -> dict[str, Any]:
        """Convert OpenAI-style messages to Gemini API format."""
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        body: dict[str, Any] = {"contents": contents}
        if system_prompt:
            body["system_instruction"] = {"parts": [{"text": system_prompt}]}
        return body

    async def chat(
        self,
        messages: list[LLMMessage],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Simple text generation (no tool calls).

        Args:
            messages: Conversation history.
            system_prompt: Optional system instruction.
            tools: Ignored in simple chat mode.

        Returns:
            Generated text string.
        """
        body = self._messages_to_gemini(messages, system_prompt)
        t0 = time.monotonic()
        resp = await self._client.post(self._url("generateContent"), json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - t0) * 1000

        text: str = data["candidates"][0]["content"]["parts"][0]["text"]
        logger.info(
            "gemini.chat",
            model=self._model,
            latency_ms=round(latency, 1),
            tokens=data.get("usageMetadata", {}).get("totalTokenCount", "?"),
        )
        return text

    async def tool_call(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Generate a response with possible function calls.

        Args:
            messages: Conversation history.
            tools: List of tool schemas in Gemini function-declaration format.
            system_prompt: Optional system instruction.

        Returns:
            Tuple of (text_response, list_of_tool_calls).
        """
        body = self._messages_to_gemini(messages, system_prompt)
        if tools:
            body["tools"] = [{"function_declarations": tools}]

        t0 = time.monotonic()
        resp = await self._client.post(self._url("generateContent"), json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - t0) * 1000

        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]

        text_parts = [p["text"] for p in parts if "text" in p]
        fn_parts = [p["functionCall"] for p in parts if "functionCall" in p]

        tool_calls = [
            ToolCall(
                name=fc["name"],
                arguments=fc.get("args", {}),
                id=f"tc_{i}",
            )
            for i, fc in enumerate(fn_parts)
        ]

        logger.info(
            "gemini.tool_call",
            model=self._model,
            latency_ms=round(latency, 1),
            tool_count=len(tool_calls),
        )
        return " ".join(text_parts), tool_calls

    async def embed(self, text: str) -> list[float]:
        """Embed text using Ollama (Gemini embed requires paid tier).

        Args:
            text: Text to embed.

        Returns:
            768-dim float vector (nomic-embed-text).
        """
        resp = await self._client.post(
            f"{self._ollama_host}/api/embed",
            json={"model": "nomic-embed-text", "input": [text]},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
