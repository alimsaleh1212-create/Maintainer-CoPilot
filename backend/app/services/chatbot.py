"""Chatbot service: tool-calling LLM loop over a primary/fallback LLMClient pair.

Flow per user turn
------------------
1. Load short-term history from Redis.
2. Retrieve top-k long-term memories for context.
3. Build messages: [history..., user_message].
4. Call primary LLM (Gemini) with tools (loop, max MAX_TOOL_ROUNDS rounds).
5. For each tool call: execute the tool, append result, continue.
6. On primary failure fall back to Ollama plain chat (no tools).
7. Persist user message + final response to Redis (redacted).
8. Return (final_response_text, tools_used_list).

LLM coupling lives entirely in app.infra.llm — this module never touches httpx.
Tool failures are caught and returned as structured ToolError dicts so the LLM
can decide what to do next; the chatbot never 500s due to a tool error.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from app.domain.errors import ToolFailure
from app.domain.tool_error import ToolError
from app.infra.llm.base import ToolCall
from app.infra.redaction import redact_text
from app.infra.tracing import TracingClient
from app.ml.ner import extract_entities as _ner_extract_entities
from app.services.classification import ClassificationService
from app.services.memory import MemoryService
from app.services.rag import RAGService

if TYPE_CHECKING:
    from app.infra.llm.gemini import GeminiClient
    from app.infra.llm.ollama import OllamaClient

logger = structlog.get_logger(__name__)

MAX_TOOL_ROUNDS = 5
_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "system_chatbot.md"
_SUMMARIZE_PROMPT = (
    "Summarize the following text in 150 words or less. "
    "Be concise and preserve the key information.\n\n"
    "Text:\n{text}\n\nSummary:"
)


def _load_system_prompt() -> str:
    """Read the system prompt from the version-controlled file."""
    try:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("chatbot.system_prompt_missing", path=str(_SYSTEM_PROMPT_PATH))
        return "You are the Maintainer's Copilot, an AI assistant for open-source maintainers."


# ---------------------------------------------------------------------------
# Tool schemas — passed to the LLM on every turn
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
        primary_llm: Primary LLM client (Gemini) used for tool-calling loop.
        fallback_llm: Fallback LLM client (Ollama) used when primary fails.
    """

    def __init__(
        self,
        primary_llm: GeminiClient,
        fallback_llm: OllamaClient,
        tracer: TracingClient | None = None,
    ) -> None:
        self._primary = primary_llm
        self._fallback = fallback_llm
        self._tracer = tracer or TracingClient(None)
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
            top_k_memories: Number of long-term memories to surface.

        Returns:
            Tuple of (final_response_text, tools_used_list).
        """
        trace = self._tracer.start_trace(
            "chat_turn",
            user_id=str(user_id),
            metadata={"conversation_id": conversation_id},
        )

        # 1. Short-term history
        history = await memory_service.get_history(conversation_id)

        # 2. Long-term memories injected into system prompt
        memories = await memory_service.search_memories(user_id, user_message, top_k_memories)
        system_prompt = self._system_prompt
        if memories:
            memory_context = "\n".join(f"[Past memory]: {m.summary}" for m in memories)
            system_prompt = f"{system_prompt}\n\n{memory_context}"

        # 3. Build initial message list (history + new user turn)
        messages: list[dict[str, Any]] = list(history)
        messages.append({"role": "user", "content": user_message})

        # 4. Tool-call loop
        tools_used: list[str] = []
        final_response = ""
        assistant_content = ""

        for _round in range(MAX_TOOL_ROUNDS):
            text, tool_calls = await self._call_llm(messages, system_prompt, trace=trace)
            assistant_content = text

            if not tool_calls:
                final_response = text
                break

            # Append the assistant turn (with function calls) to history.
            messages.append({
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
                    for tc in tool_calls
                ],
            })

            # Execute each tool and append results.
            for tc in tool_calls:
                result = await self._execute_tool(
                    tool_call=tc,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_service=memory_service,
                    classification_service=classification_service,
                    rag_service=rag_service,
                    db_session=db_session,
                    trace=trace,
                )
                tools_used.append(tc.name)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "tool_name": tc.name,
                    "content": json.dumps(result),
                })
        else:
            logger.warning(
                "chatbot.max_tool_rounds_exceeded",
                conversation_id=conversation_id,
                max_rounds=MAX_TOOL_ROUNDS,
            )
            if not final_response:
                final_response = assistant_content

        # 5. Persist to Redis (redact before write)
        await memory_service.append_message(conversation_id, "user", redact_text(user_message))
        await memory_service.append_message(conversation_id, "assistant", redact_text(final_response))

        logger.info(
            "chatbot.turn_complete",
            conversation_id=conversation_id,
            user_id=str(user_id),
            tools_used=tools_used,
            response_length=len(final_response),
        )
        return final_response, tools_used

    # ------------------------------------------------------------------
    # LLM call — primary (Gemini) with Ollama fallback
    # ------------------------------------------------------------------

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        trace: Any = None,
    ) -> tuple[str, list[ToolCall]]:
        """Try primary LLM with tools; fall back to Ollama plain chat on failure.

        Args:
            messages: Full conversation history in extended format.
            system_prompt: System instruction to pass to the LLM.
            trace: Langfuse trace to attach the span to (optional).

        Returns:
            Tuple of (text_response, tool_calls). tool_calls is empty on fallback.
        """
        with self._tracer.span(trace, name="llm_call", input={"message_count": len(messages)}):
            try:
                return await self._primary.tool_call(
                    messages=messages,
                    tools=_TOOL_SCHEMAS,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                logger.warning(
                    "chatbot.primary_llm_failed_falling_back",
                    error=str(exc),
                    fallback="ollama",
                )

            try:
                text = await self._fallback.chat(
                    messages=messages,
                    system_prompt=system_prompt,
                )
                return text, []
            except Exception as exc:
                logger.exception("chatbot.fallback_llm_failed", error=str(exc))
                return "I'm temporarily unavailable. Please try again in a moment.", []

    # ------------------------------------------------------------------
    # Tool execution dispatcher
    # ------------------------------------------------------------------

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        user_id: uuid.UUID,
        conversation_id: str,
        memory_service: MemoryService,
        classification_service: ClassificationService,
        rag_service: RAGService,
        db_session: Any,
        trace: Any = None,
    ) -> dict[str, Any]:
        """Dispatch a tool call and return a structured result.

        Failures are caught and converted to ToolError dicts so the LLM
        can decide what to do next — never raises.

        Args:
            tool_call: ToolCall object from the LLM.
            user_id: Authenticated user's UUID.
            conversation_id: Current conversation ID.
            memory_service: MemoryService instance.
            classification_service: ClassificationService instance.
            rag_service: RAGService instance.
            db_session: SQLAlchemy AsyncSession.
            trace: Langfuse trace to attach the tool span to (optional).

        Returns:
            Dict with tool result or ToolError structure.
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments

        with self._tracer.span(
            trace,
            name=f"tool_{tool_name}",
            input={"tool_name": tool_name, "args": tool_args},
        ):
            try:
                if tool_name == "classify_issue":
                    result = await classification_service.classify(tool_args.get("text", ""))
                    return result.model_dump()

                if tool_name == "rag_search":
                    search_results = await rag_service.search(
                        query=tool_args.get("query", ""),
                        db_session=db_session,
                        gemini_api_key="",  # RAG uses Ollama embedder; key unused
                        top_k=int(tool_args.get("top_k", 5)),
                    )
                    return {
                        "query": search_results.query,
                        "chunks": [
                            {"text": c.text, "source": c.source, "score": c.score}
                            for c in search_results.chunks
                        ],
                    }

                if tool_name == "write_memory":
                    memory = await memory_service.save_memory(
                        user_id=user_id,
                        summary=tool_args.get("summary", ""),
                    )
                    return {"memory_id": str(memory.id), "status": "saved"}

                if tool_name == "extract_entities":
                    text_ner: str = tool_args.get("text", "")
                    entities = await _ner_extract_entities(text_ner)
                    logger.info("chatbot.tool_extract_entities", entity_count=len(entities))
                    return {"entities": entities}

                if tool_name == "summarize_text":
                    text_sum: str = tool_args.get("text", "")
                    prompt = _SUMMARIZE_PROMPT.format(text=text_sum)
                    summary = await self._primary.chat(
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return {"summary": summary.strip()}

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
                logger.exception(
                    "chatbot.tool_unexpected_error", tool_name=tool_name, error=str(exc)
                )
                return ToolError(
                    tool_name=tool_name,
                    error=f"Unexpected error in {tool_name}: {type(exc).__name__}",
                    retryable=False,
                ).model_dump()
