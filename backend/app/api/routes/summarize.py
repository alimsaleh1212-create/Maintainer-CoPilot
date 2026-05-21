"""Summarization endpoint: POST /summarize

Uses the injected primary LLM (Gemini) to produce concise summaries.  All
network/credential plumbing is delegated to the LLMClient abstraction in
``app.infra.llm`` — this route just shapes the prompt and returns a Pydantic
response.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.dependencies import PrimaryLLMDep
from app.domain.errors import ToolFailure

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/summarize", tags=["summarization"])


class SummarizeRequest(BaseModel):
    """Request body for /summarize."""

    text: str = Field(..., min_length=10, max_length=4096, description="Text to summarize")
    max_length: int = Field(default=150, ge=50, le=500, description="Max summary length (words)")


class SummarizeResponse(BaseModel):
    """Response body for /summarize."""

    summary: str
    model: str


_SUMMARIZE_PROMPT = (
    "Summarize the following text in {max_length} words or less. "
    "Be concise and preserve the key information.\n\n"
    "Text:\n{text}\n\nSummary:"
)


@router.post("", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest, llm: PrimaryLLMDep) -> SummarizeResponse:
    """Summarize text using the primary LLM.

    Args:
        req: SummarizeRequest with text and max_length.
        llm: Injected primary LLM client (Gemini).

    Returns:
        SummarizeResponse with summary and model name.

    Raises:
        ToolFailure: If the LLM call fails (mapped to 503 by the boundary handler).
    """
    prompt = _SUMMARIZE_PROMPT.format(max_length=req.max_length, text=req.text)
    try:
        summary = await llm.chat(messages=[{"role": "user", "content": prompt}])
    except Exception as e:
        logger.exception("summarize.llm_failed", error=str(e))
        raise ToolFailure(f"Summarize LLM call failed: {e}") from e

    # Both GeminiClient and OllamaClient expose the model name as ``_model``.
    model_name = getattr(llm, "_model", "unknown")
    logger.info("summarize.success", input_length=len(req.text), output_length=len(summary))
    return SummarizeResponse(summary=summary.strip(), model=model_name)
