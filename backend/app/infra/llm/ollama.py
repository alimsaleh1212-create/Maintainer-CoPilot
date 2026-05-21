"""Ollama LLM adapter — local inference fallback for the chatbot.

Communicates with the Ollama HTTP API (``http://ollama:11434``) using
``httpx.AsyncClient``.  Serves as the fallback when Gemini is unavailable
(quota, credentials, or network issues).

Default model: ``llama3.1:8b-instruct`` (8B parameter instruction-tuned
variant — good balance of quality and speed on CPU).

Rules (CLAUDE.md):
- ``httpx.AsyncClient`` only — no ``requests``.
- ``ToolFailure`` on connection errors so the chatbot handles degradation.
- Timeout: 10 s per request (configurable via constructor).
- structlog everywhere; no ``print()``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from app.domain.errors import ToolFailure
from app.infra.llm.base import ToolCall

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "gemma3:4b"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_TIMEOUT_S = 60.0


@dataclass
class OllamaClient:
    """LLM adapter that routes requests to a local Ollama server.

    Implements the ``LLMClient`` protocol defined in ``app.infra.llm.base``.

    Args:
        base_url: Ollama server URL (e.g. ``"http://ollama:11434"``).
        model: Chat model tag to use.  Defaults to ``"llama3.1:8b-instruct"``.
        embed_model: Embedding model tag.  Defaults to ``"bge-small-en-v1.5"``.
        timeout: Request timeout in seconds.  Defaults to ``10.0``.
        http_client: Optional shared ``httpx.AsyncClient``.  A new client is
            created per-call if ``None`` (suitable for low-traffic usage).
    """

    base_url: str
    model: str = _DEFAULT_MODEL
    embed_model: str = _DEFAULT_EMBED_MODEL
    timeout: float = _DEFAULT_TIMEOUT_S
    http_client: httpx.AsyncClient | None = field(default=None)

    def _client(self) -> httpx.AsyncClient:
        """Return the shared client or a throw-away one.

        Returns:
            A configured ``httpx.AsyncClient``.
        """
        if self.http_client is not None:
            return self.http_client
        return httpx.AsyncClient(timeout=self.timeout)

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        own_client: bool,
    ) -> dict[str, Any]:
        """POST *body* to *path* and return the parsed JSON response.

        Args:
            path: URL path relative to ``base_url``.
            body: JSON-serialisable request payload.
            own_client: When ``True``, the created client is closed after use.

        Returns:
            Parsed JSON response dict.

        Raises:
            ToolFailure: on connection / timeout / non-2xx errors.
        """
        url = f"{self.base_url}{path}"
        client = self._client()
        try:
            response = await client.post(url, json=body, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise ToolFailure(
                f"Ollama {path} timed out after {self.timeout}s",
                retryable=True,
            ) from exc
        except httpx.NetworkError as exc:
            raise ToolFailure(
                f"Ollama {path} unreachable: {exc}",
                retryable=True,
            ) from exc
        except httpx.ConnectError as exc:
            raise ToolFailure(
                f"Ollama connection refused at {self.base_url}{path}: {exc}",
                retryable=True,
            ) from exc
        finally:
            if own_client:
                await client.aclose()

        if response.status_code >= 400:
            raise ToolFailure(
                f"Ollama {path} returned HTTP {response.status_code}: {response.text[:200]}",
                retryable=response.status_code >= 500,
            )

        try:
            return dict(response.json())
        except Exception as exc:
            raise ToolFailure(
                f"Ollama {path} returned invalid JSON: {exc}",
                retryable=False,
            ) from exc

    # ------------------------------------------------------------------
    # LLMClient protocol implementation
    # ------------------------------------------------------------------

    def _to_ollama_messages(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
    ) -> list[dict[str, Any]]:
        """Convert extended chatbot message format to Ollama /api/chat format.

        Handles all roles:
        - user/assistant: basic text turns
        - assistant with tool_calls: OpenAI-compatible function calls
        - tool (result): role="tool" with tool_call_id

        Args:
            messages: Extended message list from chatbot.
            system_prompt: Optional system message prepended to the list.

        Returns:
            Ollama-compatible message list.
        """
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role in ("user", "system"):
                result.append({"role": role, "content": content})

            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": content}
                raw_tcs = msg.get("tool_calls", [])
                if isinstance(raw_tcs, str):
                    raw_tcs = json.loads(raw_tcs)
                if raw_tcs:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc.get("arguments", {})},
                        }
                        for tc in raw_tcs
                    ]
                result.append(entry)

            elif role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": content,
                    }
                )

        return result

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a plain-text reply via Ollama's /api/chat endpoint.

        Args:
            messages: Conversation history (extended format).
            system_prompt: Optional system message prepended to the history.
            tools: Not used by the basic chat path; silently ignored.

        Returns:
            Model's text reply.

        Raises:
            ToolFailure: if Ollama is unreachable or returns an error.
        """
        ollama_messages = self._to_ollama_messages(messages, system_prompt)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
        }

        own = self.http_client is None
        payload = await self._post("/api/chat", body, own_client=own)

        try:
            text: str = payload["message"]["content"]
        except KeyError as exc:
            raise ToolFailure(
                f"Ollama /api/chat response missing 'message.content': {exc}",
                retryable=False,
            ) from exc

        logger.info("ollama.chat_ok", model=self.model, reply_len=len(text))
        return text

    async def tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str = "",
    ) -> tuple[str, list[ToolCall]]:
        """Submit messages to Ollama with tool/function-calling enabled.

        Ollama supports OpenAI-compatible function calling via the
        ``tools`` field in ``/api/chat``.

        Args:
            messages: Conversation history (extended format).
            tools: OpenAI-compatible tool schemas.
            system_prompt: Optional system message.

        Returns:
            A 2-tuple of (text_content, list_of_tool_calls).

        Raises:
            ToolFailure: if Ollama is unreachable or returns an error.
        """
        ollama_messages = self._to_ollama_messages(messages, system_prompt)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "tools": tools,
            "stream": False,
        }

        own = self.http_client is None
        payload = await self._post("/api/chat", body, own_client=own)

        try:
            msg: dict[str, Any] = payload["message"]
            text: str = msg.get("content", "") or ""
            raw_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
        except (KeyError, TypeError) as exc:
            raise ToolFailure(
                f"Ollama /api/chat tool-call response malformed: {exc}",
                retryable=False,
            ) from exc

        calls: list[ToolCall] = []
        for rc in raw_calls:
            try:
                fn = rc["function"]
                calls.append(
                    ToolCall(
                        name=str(fn["name"]),
                        arguments=dict(fn.get("arguments", {})),
                        id=str(rc.get("id", "")),
                    )
                )
            except (KeyError, TypeError) as exc:
                logger.warning("ollama.tool_call_parse_error", error=str(exc), raw=str(rc)[:200])

        logger.info(
            "ollama.tool_call_ok",
            model=self.model,
            tool_calls=len(calls),
            text_len=len(text),
        )
        return text, calls

    async def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector via Ollama's /api/embed endpoint.

        Args:
            text: Input text to embed.

        Returns:
            Dense float embedding vector.

        Raises:
            ToolFailure: if Ollama is unreachable or the response is malformed.
        """
        body: dict[str, Any] = {
            "model": self.embed_model,
            "input": [text],
        }

        own = self.http_client is None
        payload = await self._post("/api/embed", body, own_client=own)

        try:
            embeddings: list[list[float]] = payload["embeddings"]
            vector: list[float] = embeddings[0]
        except (KeyError, IndexError) as exc:
            raise ToolFailure(
                f"Ollama /api/embed response missing embeddings: {exc}",
                retryable=False,
            ) from exc

        logger.debug("ollama.embed_ok", model=self.embed_model, dims=len(vector))
        return vector
