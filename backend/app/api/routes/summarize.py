"""Summarization endpoint: POST /summarize"""

from __future__ import annotations

import os

import httpx
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/summarize", tags=["summarization"])


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=4096, description="Text to summarize")
    max_length: int = Field(default=150, ge=50, le=500, description="Max summary length")


class SummarizeResponse(BaseModel):
    summary: str
    model: str = "gemini-2.5-flash"
    input_tokens: int = 0
    output_tokens: int = 0


async def summarize_with_gemini(
    text: str,
    max_length: int,
    api_key: str,
) -> dict[str, str | int]:
    """Summarize text using Gemini API."""
    async with httpx.AsyncClient() as client:
        prompt = f"""Summarize the following text in {max_length} words or less.
Be concise and preserve key information.

Text:
{text}

Summary:"""

        resp = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            params={"key": api_key},
            json={
                "contents": [
                    {
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {"maxOutputTokens": max_length},
            },
            timeout=15.0,
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]

        return {
            "summary": content,
            "input_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
            "output_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
        }


@router.post("", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest) -> SummarizeResponse:
    """Summarize text using LLM.

    Currently uses Gemini 2.5 Flash via API. Credentials from environment.

    Args:
        req: SummarizeRequest with text and max_length

    Returns:
        SummarizeResponse with summary, model, token counts
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("summarize.missing_api_key")
        raise HTTPException(
            status_code=503,
            detail="Summarization service misconfigured",
        )

    try:
        result = await summarize_with_gemini(req.text, req.max_length, api_key)

        logger.info(
            "summarize.success",
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )

        return SummarizeResponse(
            summary=result["summary"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )

    except httpx.TimeoutException as e:
        logger.error("summarize.timeout", error=str(e))
        raise HTTPException(status_code=504, detail="Summarization timeout") from e

    except Exception as e:
        logger.exception("summarize.failed", error=str(e))
        raise HTTPException(status_code=500, detail="Summarization failed") from e
