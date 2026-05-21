"""Redaction-in-traces tests (graded).

CLAUDE.md rule: "Span attributes: model, prompt/completion tokens, latency,
tool I/O after redaction."

TracingClient must redact any sensitive values from span inputs, outputs, and
metadata before they reach the Langfuse client.  These tests verify the
guarantee with a mocked Langfuse instance so no real network calls are made.

Three assertion surfaces:
  1. start_span()  — input string/dict is redacted before trace.span() is called
  2. end_span()    — output string/dict is redacted before span.end() is called
  3. span() CM     — context manager auto-ends with redacted output on exception
  4. No-op mode    — a TracingClient(None) never calls any Langfuse method
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.infra.tracing import TracingClient, _NoOpTrace

# Canonical fake key used across all tests.
_FAKE_GEMINI = "AIzaTestFakeGeminiKey1234567890abcdefghij"
_FAKE_OPENAI = "sk-testFakeOpenAIKey1234567890abcdef"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_langfuse() -> MagicMock:
    """A fully-mocked Langfuse SDK object (not the real SDK)."""
    return MagicMock()


@pytest.fixture
def tracer(mock_langfuse: MagicMock) -> TracingClient:
    """TracingClient wired to the mock Langfuse (not in no-op mode)."""
    return TracingClient(langfuse_client=mock_langfuse)


@pytest.fixture
def mock_trace() -> MagicMock:
    """A trace mock that has a `span` attribute (so it isn't treated as no-op)."""
    trace = MagicMock()
    trace.span = MagicMock(return_value=MagicMock())
    return trace


# ---------------------------------------------------------------------------
# start_span — input redaction
# ---------------------------------------------------------------------------


class TestStartSpanRedaction:
    def test_string_input_with_api_key_is_redacted(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        # Arrange
        sensitive_input = f"user pasted: {_FAKE_GEMINI}"

        # Act
        tracer.start_span(mock_trace, name="classify", input=sensitive_input)

        # Assert — Langfuse trace.span() received the redacted version
        call_kwargs = mock_trace.span.call_args.kwargs
        assert _FAKE_GEMINI not in call_kwargs["input"]
        assert "[REDACTED" in call_kwargs["input"]

    def test_dict_input_with_api_key_is_redacted(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        sensitive_input = {"tool_input": f"auth={_FAKE_OPENAI}", "model": "gemini-flash"}

        tracer.start_span(mock_trace, name="llm_call", input=sensitive_input)

        call_kwargs = mock_trace.span.call_args.kwargs
        assert _FAKE_OPENAI not in call_kwargs["input"]["tool_input"]
        assert call_kwargs["input"]["model"] == "gemini-flash"  # safe value unchanged

    def test_metadata_with_api_key_is_redacted(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        sensitive_meta = {"raw_key": _FAKE_OPENAI, "latency_ms": 120}

        tracer.start_span(mock_trace, name="rag_search", metadata=sensitive_meta)

        call_kwargs = mock_trace.span.call_args.kwargs
        assert _FAKE_OPENAI not in call_kwargs["metadata"]["raw_key"]
        assert call_kwargs["metadata"]["latency_ms"] == 120  # non-string unchanged

    def test_safe_input_is_passed_through_unchanged(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        safe_input = "classify: bug report about NaN in transform pipeline"

        tracer.start_span(mock_trace, name="classify", input=safe_input)

        call_kwargs = mock_trace.span.call_args.kwargs
        assert call_kwargs["input"] == safe_input

    def test_none_input_is_passed_through(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        tracer.start_span(mock_trace, name="classify", input=None)

        call_kwargs = mock_trace.span.call_args.kwargs
        assert call_kwargs["input"] is None


# ---------------------------------------------------------------------------
# end_span — output redaction
# ---------------------------------------------------------------------------


class TestEndSpanRedaction:
    def test_string_output_with_api_key_is_redacted(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        # Arrange
        mock_span = MagicMock()
        sensitive_output = f"LLM echoed the key: {_FAKE_GEMINI}"

        # Act
        tracer.end_span(mock_span, output=sensitive_output)

        # Assert
        call_kwargs = mock_span.end.call_args.kwargs
        assert _FAKE_GEMINI not in call_kwargs["output"]
        assert "[REDACTED" in call_kwargs["output"]

    def test_dict_output_with_api_key_is_redacted(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        mock_span = MagicMock()
        sensitive_output = {"summary": f"classified as bug; token={_FAKE_OPENAI}"}

        tracer.end_span(mock_span, output=sensitive_output)

        call_kwargs = mock_span.end.call_args.kwargs
        assert _FAKE_OPENAI not in call_kwargs["output"]["summary"]

    def test_safe_output_is_passed_through_unchanged(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        mock_span = MagicMock()
        safe_output = "label=bug confidence=0.92"

        tracer.end_span(mock_span, output=safe_output)

        call_kwargs = mock_span.end.call_args.kwargs
        assert call_kwargs["output"] == safe_output


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------


class TestSpanContextManager:
    def test_span_context_manager_calls_end_on_normal_exit(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        with tracer.span(mock_trace, name="test_span"):
            pass  # normal exit

        mock_span.end.assert_called_once()

    def test_span_context_manager_calls_end_with_error_on_exception(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        with pytest.raises(ValueError, match="boom"):
            with tracer.span(mock_trace, name="test_span"):
                raise ValueError("boom")

        # end() is called even on exception
        mock_span.end.assert_called_once()
        call_kwargs = mock_span.end.call_args.kwargs
        assert call_kwargs["output"] == {"error": "exception raised"}

    def test_span_context_manager_yields_the_span(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        with tracer.span(mock_trace, name="check_yield") as yielded:
            assert yielded is mock_span


# ---------------------------------------------------------------------------
# No-op mode
# ---------------------------------------------------------------------------


class TestNoOpMode:
    def test_noop_client_does_not_call_langfuse(self) -> None:
        noop_tracer = TracingClient(langfuse_client=None)
        mock_trace = MagicMock()

        noop_tracer.start_span(mock_trace, name="test", input=_FAKE_GEMINI)

        # No span() call on the trace — we're in no-op mode
        mock_trace.span.assert_not_called()

    def test_noop_start_trace_returns_noop_trace(self) -> None:
        noop_tracer = TracingClient(langfuse_client=None)

        trace = noop_tracer.start_trace("chat_turn")

        assert isinstance(trace, _NoOpTrace)
        assert trace.id == "noop-trace"

    def test_noop_end_span_does_not_raise(self) -> None:
        noop_tracer = TracingClient(langfuse_client=None)
        mock_span = MagicMock()

        # Must not raise even with sensitive data
        noop_tracer.end_span(mock_span, output=_FAKE_GEMINI)

    def test_placeholder_keys_result_in_noop_mode(self) -> None:
        client = TracingClient.from_keys(
            public_key="dev-langfuse-public-key",
            secret_key="dev-langfuse-secret-key",
            host="http://langfuse:3000",
        )
        # Placeholder keys → TracingClient should be in no-op mode
        assert client._is_noop is True


# ---------------------------------------------------------------------------
# Full end-to-end: sensitive data in tool I/O never reaches Langfuse
# ---------------------------------------------------------------------------


class TestEndToEndRedactionGuarantee:
    def test_fake_api_key_in_tool_input_never_reaches_langfuse(
        self, tracer: TracingClient, mock_trace: MagicMock
    ) -> None:
        """CRITICAL PATH: simulate a tool receiving sensitive user input."""
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        # Simulate chatbot passing a message that contains a key to a tool span
        contaminated_input = {"text": f"user said their key is {_FAKE_GEMINI}"}

        with tracer.span(mock_trace, name="tool_classify", input=contaminated_input):
            pass  # tool ran successfully

        # Check what was passed to trace.span()
        span_kwargs = mock_trace.span.call_args.kwargs
        assert _FAKE_GEMINI not in str(span_kwargs["input"])
