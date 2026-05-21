"""Chatbot service: tool-calling LLM loop over Gemini (Ollama fallback).

Flow per user turn
------------------
1. Load short-term history from Redis.
2. Retrieve top-k long-term memories for context.
3. Build the messages list:
   [system_prompt, ...memories_as_context, ...history, user_message]
4. Call LLM with tools (loop, max MAX_TOOL_ROUNDS rounds).
5. For each tool call: execute the tool, append result, continue.
6. Append user message + final assistant response to Redis history.
7. Return final response text.

Tool failures are caught and returned as structured ToolError dicts so the
LLM can decide what to do next — the chatbot never 500s because of a tool.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import structlog

from app.domain.errors import ToolFailure
from app.domain.tool_error import ToolError
from app.infra.redaction import redact_text
from app.services.classification import ClassificationService
from app.services.memory import MemoryService
from app.services.rag import RAGService

logger = structlog.get_logger(__name__)

MAX_TOOL_ROUNDS = 5
_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "system_chatbot.md"


def _load_system_prompt() -> str:
    """Read the system prompt from the version-controlled file."""
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("chatbot.system_prompt_missing", path=str(_SYSTEM_PROMPT_PATH))
        return "You are the Maintainer's Copilot, an AI assistant for open-source maintainers."


# ---------------------------------------------------------------------------
# Tool registry — thin stubs that delegate to injected services
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "classify_issue",
        "description": "Classify a GitHub issue as bug, feature, or support.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Issue title + body text"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "rag_search",
        "description": "Search project documentation and resolved issues for relevant context.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_memory",
        "description": "Store important information from this conversation for future sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary to persist in long-term memory",
                },
            },
            "required": ["summary"],
        },
    },
    {
        "name": "extract_entities",
        "description": "Extract code entities (functions, classes, files, errors) from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract entities from"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "summarize_text",
        "description": "Generate a concise summary of an issue or conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
            },
            "required": ["text"],
        },
    },
]


class ChatbotService:
    """Tool-calling chatbot that orchestrates LLM + tools + memory.

    Args:
        gemini_api_key: API key for Gemini (primary LLM).
        ollama_host: Base URL for Ollama (fallback LLM).
    """

    def __init__(
        self,
        gemini_api_key: str,
        ollama_host: str,
    ) -> None:
        self._gemini_api_key = gemini_api_key
        self._ollama_host = ollama_host
        self._system_prompt = _load_system_prompt()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        conversation_id: str,
        user_id: uuid.UUID,
        memory_service: MemoryService,
        classification_service: ClassificationService,
        rag_service: RAGService,
        db_session: Any,
        gemini_api_key: str | None = None,
        top_k_memories: int = 3,
    ) -> tuple[str, list[str]]:
        """Run one user turn through the tool-calling loop.

        Args:
            user_message: Raw message from the user.
            conversation_id: Redis history key / trace grouping ID.
            user_id: Authenticated user's UUID.
            memory_service: Injected MemoryService for history + long-term.
            classification_service: Injected ClassificationService.
            rag_service: Injected RAGService.
            db_session: SQLAlchemy AsyncSession (passed to RAGService).
            gemini_api_key: Override API key (uses __init__ value if None).
            top_k_memories: Number of long-term memories to surface.

        Returns:
            Tuple of (final_response_text, tools_used_list).
        """
        api_key = gemini_api_key or self._gemini_api_key

        # 1. Short-term history
        history = await memory_service.get_history(conversation_id)

        # 2. Long-term memories
        memories = await memory_service.search_memories(user_id, user_message, top_k_memories)
        memory_context = "\n".join(
            f"[Past memory]: {m.summary}" for m in memories
        )

        # 3. Build messages
        system_content = self._system_prompt
        if memory_context:
            system_content = f"{system_content}\n\n{memory_context}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        # 4. Tool-call loop
        tools_used: list[str] = []
        final_response = ""

        for _round in range(MAX_TOOL_ROUNDS):
            response_data = await self._call_llm(messages, api_key)

            tool_calls = response_data.get("tool_calls", [])
            assistant_content: str = response_data.get("content", "")

            if not tool_calls:
                # No more tool calls — we have the final answer.
                final_response = assistant_content
                break

            # Append the assistant message (with tool_calls) to history.
            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": json.dumps(tool_calls),
            })

            # Execute each tool call.
            for call in tool_calls:
                tool_name: str = call.get("name", "unknown")
                tool_args: dict[str, Any] = call.get("arguments", {})
                tool_id: str = call.get("id", str(uuid.uuid4()))

                result = await self._execute_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_service=memory_service,
                    classification_service=classification_service,
                    rag_service=rag_service,
                    db_session=db_session,
                    gemini_api_key=api_key,
                )
                tools_used.append(tool_name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(result),
                })
        else:
            # Exceeded MAX_TOOL_ROUNDS — use last content as the response.
            logger.warning(
                "chatbot.max_tool_rounds_exceeded",
                conversation_id=conversation_id,
                max_rounds=MAX_TOOL_ROUNDS,
            )
            if not final_response:
                final_response = assistant_content

        # 5. Persist to Redis (redact before write)
        redacted_user = redact_text(user_message)
        redacted_response = redact_text(final_response)
        await memory_service.append_message(conversation_id, "user", redacted_user)
        await memory_service.append_message(conversation_id, "assistant", redacted_response)

        logger.info(
            "chatbot.turn_complete",
            conversation_id=conversation_id,
            user_id=str(user_id),
            tools_used=tools_used,
            response_length=len(final_response),
        )
        return final_response, tools_used

    # ------------------------------------------------------------------
    # LLM call (Gemini primary, stub — real integration wired THU)
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        api_key: str,
    ) -> dict[str, Any]:
        """Call the LLM with the current message list.

        Primary: Gemini via REST.  Falls back to a minimal Ollama call when
        the Gemini request fails.  The full multi-modal tool-call integration
        is wired on THU; for now the stub returns an echo-style response so
        the rest of the loop is exercisable.

        Args:
            messages: Full conversation history including system prompt.
            api_key: Gemini API key.

        Returns:
            Dict with ``content`` (str) and optional ``tool_calls`` (list).
        """
        try:
            return await self._call_gemini(messages, api_key)
        except Exception as exc:
            logger.warning(
                "chatbot.gemini_failed_falling_back",
                error=str(exc),
                fallback="ollama",
            )
            return await self._call_ollama_fallback(messages)

    async def _call_gemini(
        self,
        messages: list[dict[str, str]],
        api_key: str,
    ) -> dict[str, Any]:
        """Call Gemini 1.5 Flash via its REST API.

        Args:
            messages: Conversation messages.
            api_key: Google AI API key.

        Returns:
            Parsed response dict with ``content`` and ``tool_calls``.

        Raises:
            httpx.HTTPError: On transport or HTTP-level error.
        """
        # Build Gemini-format contents (system handled separately)
        system_instruction = ""
        gemini_contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [{"text": content}]})
            # tool results are handled as user turns in Gemini format

        payload: dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.2},
        }
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={api_key}"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p]
        response_text = " ".join(text_parts)

        return {"content": response_text, "tool_calls": []}

    async def _call_ollama_fallback(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Call Ollama (llama3 or similar) as a fallback.

        Args:
            messages: Conversation messages.

        Returns:
            Dict with ``content`` and empty ``tool_calls``.
        """
        # Convert messages to a single prompt string for Ollama
        prompt = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in messages
        )

        payload = {
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._ollama_host}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                return {"content": data.get("response", ""), "tool_calls": []}
        except Exception as exc:
            logger.exception("chatbot.ollama_fallback_failed", error=str(exc))
            return {
                "content": "I'm temporarily unavailable. Please try again in a moment.",
                "tool_calls": [],
            }

    # ------------------------------------------------------------------
    # Tool execution dispatcher
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        user_id: uuid.UUID,
        conversation_id: str,
        memory_service: MemoryService,
        classification_service: ClassificationService,
        rag_service: RAGService,
        db_session: Any,
        gemini_api_key: str,
    ) -> dict[str, Any]:
        """Dispatch a tool call and return a structured result.

        Failures are caught and converted to ToolError dicts so the LLM
        can decide what to do next — never raises.

        Args:
            tool_name: Name of the tool to invoke.
            tool_args: Arguments from the LLM's tool call.
            user_id: Authenticated user's UUID.
            conversation_id: Current conversation ID.
            memory_service: MemoryService instance.
            classification_service: ClassificationService instance.
            rag_service: RAGService instance.
            db_session: SQLAlchemy AsyncSession.
            gemini_api_key: API key for query rewrite in RAG.

        Returns:
            Dict with tool result or ToolError structure.
        """
        try:
            if tool_name == "classify_issue":
                text: str = tool_args.get("text", "")
                result = await classification_service.classify(text)
                return result.model_dump()

            if tool_name == "rag_search":
                query: str = tool_args.get("query", "")
                top_k: int = int(tool_args.get("top_k", 5))
                search_results = await rag_service.search(
                    query=query,
                    db_session=db_session,
                    gemini_api_key=gemini_api_key,
                    top_k=top_k,
                )
                return {
                    "query": search_results.query,
                    "chunks": [
                        {"text": c.text, "source": c.source, "score": c.score}
                        for c in search_results.chunks
                    ],
                }

            if tool_name == "write_memory":
                summary: str = tool_args.get("summary", "")
                memory = await memory_service.save_memory(user_id=user_id, summary=summary)
                return {"memory_id": str(memory.id), "status": "saved"}

            if tool_name == "extract_entities":
                # Stub: full NER integration wired WED/THU.
                text_ner: str = tool_args.get("text", "")
                logger.info("chatbot.tool_extract_entities_stub", text_length=len(text_ner))
                return {"entities": [], "note": "NER stub — full integration wired Wednesday"}

            if tool_name == "summarize_text":
                # Stub: summarizer integration wired WED/THU.
                text_sum: str = tool_args.get("text", "")
                logger.info("chatbot.tool_summarize_stub", text_length=len(text_sum))
                return {
                    "summary": text_sum[:200] + "..." if len(text_sum) > 200 else text_sum,
                    "note": "Summarizer stub — full integration wired Wednesday",
                }

            # Unknown tool
            logger.warning("chatbot.unknown_tool", tool_name=tool_name)
            return ToolError(
                tool_name=tool_name,
                error=f"Unknown tool: {tool_name}",
                retryable=False,
            ).model_dump()

        except ToolFailure as exc:
            logger.warning(
                "chatbot.tool_failure",
                tool_name=tool_name,
                error=exc.message,
                retryable=exc.retryable,
            )
            return ToolError(
                tool_name=tool_name,
                error=exc.message,
                retryable=exc.retryable,
            ).model_dump()

        except Exception as exc:
            logger.exception("chatbot.tool_unexpected_error", tool_name=tool_name, error=str(exc))
            return ToolError(
                tool_name=tool_name,
                error=f"Unexpected error in {tool_name}: {type(exc).__name__}",
                retryable=False,
            ).model_dump()
