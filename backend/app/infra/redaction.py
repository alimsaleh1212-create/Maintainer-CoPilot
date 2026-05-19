"""Redaction layer — scrubs secrets from logs, trace spans, and memory writes.

Rules (CLAUDE.md):
- redact_text() runs before any log line, Langfuse span attribute, or memory
  write leaves the service boundary.
- Patterns are documented in docs/SECURITY.md.
- A unit test (tests/unit/test_redaction.py) asserts that fake API keys never
  appear unredacted through any of the three emission paths.

Add new patterns HERE and update SECURITY.md — not in callers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from typing import Any

# ---------------------------------------------------------------------------
# Compiled redaction patterns (order matters — more specific first).
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Anthropic keys — sk-ant- prefix (must precede generic sk- rule)
    (re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}"), "[REDACTED:anthropic-key]"),
    # OpenAI / generic sk- keys (covers any sk-<long-string> shape)
    (re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"), "[REDACTED:api-key]"),
    # Google AI / Gemini keys (AIza prefix, 35 chars of base64url)
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "[REDACTED:google-key]"),
    # GitHub PATs — new format (github_pat_)
    (re.compile(r"github_pat_[a-zA-Z0-9_]{22,}"), "[REDACTED:github-pat]"),
    # GitHub PATs — classic format (ghp_)
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "[REDACTED:github-pat]"),
    # Langfuse public and secret keys (pk-lf- / sk-lf-)
    (re.compile(r"(?:pk|sk)-lf-[a-zA-Z0-9\-]{20,}"), "[REDACTED:langfuse-key]"),
    # AWS IAM access key IDs (AKIA prefix, 16 uppercase alphanumerics)
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws-key]"),
    # JWT tokens (three base64url-encoded segments separated by dots)
    (
        re.compile(r"ey[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+"),
        "[REDACTED:jwt]",
    ),
    # PEM private key headers (RSA, EC, etc.)
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "[REDACTED:private-key]"),
]


def redact_text(text: str) -> str:
    """Scrub all known secret patterns from a string.

    Safe to call on every log message — patterns are pre-compiled.

    Args:
        text: Raw string that may contain secrets.

    Returns:
        String with every matched pattern replaced by a [REDACTED:*] token.
    """
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Redact all string values in a mapping (used for Langfuse span attributes).

    Non-string values (int, float, bool, list, …) are passed through unchanged
    so numeric metrics are never lost.

    Args:
        data: Arbitrary key→value mapping (e.g. span attributes).

    Returns:
        New dict with string values scrubbed.
    """
    return {k: redact_text(v) if isinstance(v, str) else v for k, v in data.items()}


def structlog_redaction_processor(
    logger: Any,
    method: str,
    event_dict: MutableMapping[str, Any],
) -> Mapping[str, Any]:
    """Structlog processor — redacts all string values before log emission.

    Wire this into structlog.configure(processors=[..., structlog_redaction_processor, ...])
    so it runs on every log record regardless of which module emits it.

    Args:
        logger: Structlog bound logger (unused, required by protocol).
        method: Log method name (unused, required by protocol).
        event_dict: Mutable event dictionary being assembled.

    Returns:
        event_dict with all string values redacted.
    """
    return {k: redact_text(v) if isinstance(v, str) else v for k, v in event_dict.items()}
