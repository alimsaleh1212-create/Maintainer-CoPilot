"""Issue/conversation summarizer using an LLM (Gemini Flash).

Loads the system prompt from ``backend/prompts/summarize.md`` at construction
time, falling back to an inline default if the file is absent.

Public API
----------
Summarizer.summarize(text) → str
    Async; truncates input to 4 000 characters before sending to the LLM.
summarize_text(text, llm_client) → str
    Convenience function for one-shot use without constructing the class.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_INPUT_CHARS = 4_000

_INLINE_SYSTEM_PROMPT = (
    "You are a technical summarizer for GitHub issue conversations.\n"
    "Summarize the following issue/conversation in 2-3 concise sentences.\n"
    "Focus on: what the problem is, what was tried, and the outcome.\n"
    "Be factual and technical. Do not editorialize."
)

_PROMPT_FILE = Path(__file__).parent.parent.parent / "prompts" / "summarize.md"


def _load_system_prompt() -> str:
    """Load the summarization system prompt from disk.

    Returns:
        Prompt text from ``prompts/summarize.md``, or the inline default
        when the file does not exist.
    """
    if _PROMPT_FILE.exists():
        content = _PROMPT_FILE.read_text(encoding="utf-8").strip()
        logger.debug("summarizer_prompt_loaded_from_disk", path=str(_PROMPT_FILE))
        return content
    logger.debug("summarizer_prompt_file_not_found_using_inline_default")
    return _INLINE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Summarizer class
# ---------------------------------------------------------------------------

class Summarizer:
    """LLM-backed summarizer for GitHub issue text.

    Args:
        llm_client: Any LLM client that exposes a
            ``generate_content(prompt: str) → <response with .text>``
            interface (e.g. a ``google.generativeai.GenerativeModel`` instance).
        system_prompt: Override the system prompt loaded from disk.
    """

    def __init__(
        self,
        llm_client: Any,
        system_prompt: str | None = None,
    ) -> None:
        self._client = llm_client
        self._system_prompt = system_prompt or _load_system_prompt()

    async def summarize(self, text: str) -> str:
        """Summarize issue or conversation text.

        Truncates input longer than 4 000 characters before sending to the LLM.

        Args:
            text: Raw issue title + body (or conversation transcript).

        Returns:
            A 2–3 sentence summary as a plain string.

        Raises:
            RuntimeError: If the LLM call fails after retries (callers should
                catch and return a ToolError).
        """
        return await summarize_text(text, self._client, system_prompt=self._system_prompt)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

async def summarize_text(
    text: str,
    llm_client: Any,
    *,
    system_prompt: str | None = None,
) -> str:
    """Summarize text using an LLM client in a single call.

    Args:
        text: Issue/conversation text to summarize.
        llm_client: LLM client with a ``generate_content`` or
            ``chat.completions.create`` interface. Tried in this order:
            1. ``generate_content(full_prompt)`` (Gemini SDK style).
            2. Any object with ``generate_content(full_prompt)`` as above.
        system_prompt: Override the default system prompt. Loaded from
            ``prompts/summarize.md`` when ``None``.

    Returns:
        A concise 2–3 sentence summary string.

    Raises:
        RuntimeError: Propagates any exception from the underlying LLM call.
    """
    if not text:
        return ""

    prompt_header = system_prompt or _load_system_prompt()

    # Truncate to keep within context budget
    truncated = text[:_MAX_INPUT_CHARS]
    if len(text) > _MAX_INPUT_CHARS:
        logger.debug(
            "summarizer_input_truncated",
            original_chars=len(text),
            truncated_to=_MAX_INPUT_CHARS,
        )

    full_prompt = f"{prompt_header}\n\n---\n\n{truncated}"

    t0 = time.perf_counter()
    try:
        response = llm_client.generate_content(full_prompt)
        summary: str = response.text.strip()
    except AttributeError:
        # Fallback: try OpenAI-style interface
        response = llm_client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": prompt_header},
                {"role": "user", "content": truncated},
            ],
        )
        summary = response.choices[0].message.content.strip()

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "summarizer_completed",
        input_chars=len(truncated),
        output_chars=len(summary),
        latency_ms=round(latency_ms, 1),
    )
    return summary
