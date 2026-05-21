"""Langfuse tracing adapter — structured observability for LLM calls.

Every LLM call, tool call, and RAG retrieval must be a Langfuse span whose
attributes are redacted before emission (CLAUDE.md rules).

No-op mode:
    When Langfuse keys are absent / placeholder values, the client runs in
    no-op mode (all methods are silently skipped).  Tracing is observability,
    not security — a missing Langfuse key must NOT prevent the service from
    starting.

Rules (CLAUDE.md):
- ``redact_dict()`` from ``app.infra.redaction`` is called before writing
  any attribute to a span.
- structlog everywhere; no ``print()``.
- The ``TracingClient`` is constructed in ``lifespan()`` and injected via
  ``Depends()``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog

from app.infra.redaction import redact_dict, redact_text

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional Langfuse import — graceful degradation if not installed.
# ---------------------------------------------------------------------------
try:
    from langfuse import Langfuse  # type: ignore[import-untyped]
    from langfuse.api.resources.commons.errors.unauthorized_error import (  # type: ignore[import-untyped]
        UnauthorizedError,
    )

    _LANGFUSE_AVAILABLE = True
except ModuleNotFoundError:
    Langfuse = None  # type: ignore[assignment,misc]
    UnauthorizedError = Exception  # type: ignore[assignment,misc]
    _LANGFUSE_AVAILABLE = False

# Sentinel values that indicate Langfuse is not configured.
_PLACEHOLDER_PREFIXES = ("placeholder", "your-", "changeme", "")


def _is_placeholder(value: str) -> bool:
    """Return True if *value* looks like an unconfigured placeholder."""
    lower = value.lower().strip()
    return any(lower.startswith(p) for p in _PLACEHOLDER_PREFIXES) or len(lower) < 6


class _NoOpTrace:
    """Null-object trace — all methods are no-ops."""

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass

    @property
    def id(self) -> str:
        return "noop-trace"


class _NoOpSpan:
    """Null-object span — all methods are no-ops."""

    def end(self, output: Any = None, **kwargs: Any) -> None:  # noqa: ARG002
        pass

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass


class TracingClient:
    """Langfuse observability adapter.

    Wraps the Langfuse SDK and redacts all attributes before emission.
    Falls back to no-op logging when the package is not installed or keys
    are not configured.

    Args:
        langfuse_client: A fully-authenticated ``Langfuse`` instance, or
            ``None`` to run in no-op mode.
    """

    def __init__(self, langfuse_client: Any | None) -> None:
        self._client: Any | None = langfuse_client
        self._is_noop = langfuse_client is None
        if self._is_noop:
            logger.info("tracing.noop_mode", reason="langfuse_client is None")

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @classmethod
    def from_keys(
        cls,
        public_key: str,
        secret_key: str,
        host: str,
    ) -> "TracingClient":
        """Construct a ``TracingClient`` from Vault-resolved keys.

        Falls back to no-op mode if:
        - The ``langfuse`` package is not installed.
        - Either key looks like a placeholder.
        - Langfuse authentication fails.

        Args:
            public_key: Langfuse project public key (``pk-lf-...``).
            secret_key: Langfuse project secret key (``sk-lf-...``).
            host: Langfuse server URL.

        Returns:
            ``TracingClient`` instance (may be in no-op mode).
        """
        if not _LANGFUSE_AVAILABLE or Langfuse is None:
            logger.warning(
                "tracing.langfuse_not_installed",
                advice="Run `uv add langfuse` to enable tracing.",
            )
            return cls(None)

        if _is_placeholder(public_key) or _is_placeholder(secret_key):
            logger.warning(
                "tracing.placeholder_keys",
                advice="Set real Langfuse keys in Vault to enable tracing.",
            )
            return cls(None)

        try:
            client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("tracing.langfuse_configured", host=host)
            return cls(client)
        except UnauthorizedError as exc:
            logger.warning("tracing.langfuse_auth_failed", error=str(exc))
            return cls(None)
        except Exception as exc:
            logger.warning("tracing.langfuse_init_failed", error=str(exc))
            return cls(None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_trace(
        self,
        name: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create and return a new Langfuse trace.

        Args:
            name: Human-readable trace name (e.g. ``"chat_turn"``).
            user_id: Authenticated user identifier for Langfuse grouping.
            metadata: Arbitrary key→value pairs (redacted before emission).

        Returns:
            A Langfuse trace object, or a no-op trace in no-op mode.
        """
        safe_meta = redact_dict(metadata or {})
        if self._is_noop or self._client is None:
            logger.debug(
                "tracing.start_trace_noop",
                name=name,
                user_id=user_id,
            )
            return _NoOpTrace()

        trace = self._client.trace(
            name=name,
            user_id=user_id,
            metadata=safe_meta,
        )
        logger.debug("tracing.start_trace", name=name, trace_id=trace.id)
        return trace

    def start_span(
        self,
        trace: Any,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create and return a span within *trace*.

        Args:
            trace: Parent trace (or no-op trace) returned by ``start_trace``.
            name: Span name (e.g. ``"classify"``, ``"rag_retrieve"``).
            input: Span input — redacted if it is a string or dict.
            metadata: Arbitrary key→value pairs (redacted before emission).

        Returns:
            A Langfuse span object, or a no-op span in no-op mode.
        """
        safe_input: Any
        if isinstance(input, str):
            safe_input = redact_text(input)
        elif isinstance(input, dict):
            safe_input = redact_dict(input)
        else:
            safe_input = input

        safe_meta = redact_dict(metadata or {})

        if self._is_noop or not hasattr(trace, "span"):
            logger.debug("tracing.start_span_noop", name=name)
            return _NoOpSpan()

        span = trace.span(name=name, input=safe_input, metadata=safe_meta)
        logger.debug("tracing.start_span", name=name)
        return span

    def end_span(
        self,
        span: Any,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Finalise *span* with the given output and metadata.

        Args:
            span: Span returned by ``start_span``.
            output: Span output — redacted if string or dict.
            metadata: Additional metadata to attach (redacted).
        """
        safe_output: Any
        if isinstance(output, str):
            safe_output = redact_text(output)
        elif isinstance(output, dict):
            safe_output = redact_dict(output)
        else:
            safe_output = output

        safe_meta = redact_dict(metadata or {})

        if hasattr(span, "end"):
            with contextlib.suppress(Exception):
                span.end(output=safe_output, metadata=safe_meta)
        logger.debug("tracing.end_span")

    @contextmanager
    def span(
        self,
        trace: Any,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Context manager that starts and auto-ends a span.

        Args:
            trace: Parent trace returned by ``start_trace``.
            name: Span name.
            input: Span input (redacted).
            metadata: Optional metadata (redacted).

        Yields:
            The active span object.
        """
        s = self.start_span(trace, name=name, input=input, metadata=metadata)
        try:
            yield s
        except Exception:
            self.end_span(s, output={"error": "exception raised"})
            raise
        else:
            self.end_span(s)

    async def flush(self) -> None:
        """Flush all pending Langfuse events to the server.

        Safe to call in shutdown handlers even in no-op mode.
        """
        if self._is_noop or self._client is None:
            return
        try:
            self._client.flush()
            logger.debug("tracing.flush_ok")
        except Exception as exc:
            logger.warning("tracing.flush_failed", error=str(exc))
