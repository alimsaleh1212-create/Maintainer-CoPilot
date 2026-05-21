"""Mandatory redaction test (graded).

The brief requires: "A test asserts that a message containing a fake API key
never appears unredacted in logs, traces, or memory."

Three paths under test:
  1. Logger  — structlog_redaction_processor scrubs the event_dict.
  2. Traces  — redact_dict scrubs Langfuse span attributes before submission.
  3. Memory  — redact_text scrubs the summary before the pgvector write.
"""

from app.infra.redaction import redact_dict, redact_text, structlog_redaction_processor

# ---------------------------------------------------------------------------
# Canonical fake keys used across all three path tests.
# ---------------------------------------------------------------------------
FAKE_OPENAI = "sk-testFakeOpenAIKey1234567890abcdef"
FAKE_ANTHROPIC = "sk-ant-testFakeAnthropicKey1234567890abcdef"
FAKE_GEMINI = "AIzaTestFakeGeminiKey1234567890abcdefghij"
FAKE_GITHUB = "ghp_testFakeGitHubPATAAAABBBBCCCCDDDDEEEE"  # 40 chars after ghp_
FAKE_LANGFUSE_SK = "sk-lf-testFakeLangfuseKey1234567890abcdef"
FAKE_AWS = "AKIA1234567890ABCDEF"  # exactly AKIA + 16 uppercase alphanumerics
FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyMSJ9.testFakeJWTSignatureABCDEFGH"


# ---------------------------------------------------------------------------
# Core redact_text — pattern coverage
# ---------------------------------------------------------------------------
class TestRedactTextPatterns:
    def test_openai_key_redacted(self) -> None:
        result = redact_text(f"Authorization: Bearer {FAKE_OPENAI}")
        assert FAKE_OPENAI not in result
        assert "[REDACTED" in result

    def test_anthropic_key_redacted(self) -> None:
        result = redact_text(f"x-api-key: {FAKE_ANTHROPIC}")
        assert FAKE_ANTHROPIC not in result

    def test_gemini_key_redacted(self) -> None:
        result = redact_text(f"?key={FAKE_GEMINI}")
        assert FAKE_GEMINI not in result

    def test_github_pat_redacted(self) -> None:
        result = redact_text(f"token={FAKE_GITHUB}")
        assert FAKE_GITHUB not in result

    def test_langfuse_sk_redacted(self) -> None:
        result = redact_text(f"secret={FAKE_LANGFUSE_SK}")
        assert FAKE_LANGFUSE_SK not in result

    def test_aws_key_redacted(self) -> None:
        result = redact_text(f"AWS_ACCESS_KEY_ID={FAKE_AWS}")
        assert FAKE_AWS not in result

    def test_jwt_redacted(self) -> None:
        result = redact_text(f"token: {FAKE_JWT}")
        assert FAKE_JWT not in result

    def test_normal_text_unchanged(self) -> None:
        text = "User asked about MONAI model training. No secrets here."
        assert redact_text(text) == text

    def test_multiple_keys_in_one_string_both_redacted(self) -> None:
        text = f"key1={FAKE_OPENAI} key2={FAKE_GITHUB}"
        result = redact_text(text)
        assert FAKE_OPENAI not in result
        assert FAKE_GITHUB not in result


# ---------------------------------------------------------------------------
# Path 1 — Logger (structlog processor)
# ---------------------------------------------------------------------------
class TestLoggerPath:
    """structlog_redaction_processor must scrub secrets before log emission."""

    def test_event_field_redacted(self) -> None:
        event_dict = {"event": f"user pasted key: {FAKE_OPENAI}", "level": "info"}
        result = structlog_redaction_processor(None, "info", event_dict)
        assert FAKE_OPENAI not in result["event"]
        assert "[REDACTED" in result["event"]

    def test_arbitrary_field_redacted(self) -> None:
        event_dict = {
            "event": "tool_call",
            "tool_input": f"classify this issue {FAKE_ANTHROPIC}",
        }
        result = structlog_redaction_processor(None, "info", event_dict)
        assert FAKE_ANTHROPIC not in result["tool_input"]

    def test_non_string_fields_pass_through(self) -> None:
        event_dict = {"event": "latency", "duration_ms": 123, "tokens": 42}
        result = structlog_redaction_processor(None, "info", event_dict)
        assert result["duration_ms"] == 123
        assert result["tokens"] == 42


# ---------------------------------------------------------------------------
# Path 2 — Traces (Langfuse span attributes)
# ---------------------------------------------------------------------------
class TestTracesPath:
    """redact_dict must scrub span attribute values before they reach Langfuse."""

    def test_string_attribute_redacted(self) -> None:
        attrs = {
            "tool_input": f"analyze issue with key {FAKE_OPENAI}",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 512,
        }
        result = redact_dict(attrs)
        assert FAKE_OPENAI not in result["tool_input"]
        assert result["model"] == "gemini-2.5-flash"  # untouched safe string
        assert result["prompt_tokens"] == 512  # non-string pass-through

    def test_gemini_key_in_span_redacted(self) -> None:
        attrs = {"llm_request": f"api_key={FAKE_GEMINI}&q=hello"}
        result = redact_dict(attrs)
        assert FAKE_GEMINI not in result["llm_request"]

    def test_empty_dict_handled(self) -> None:
        assert redact_dict({}) == {}


# ---------------------------------------------------------------------------
# Path 3 — Memory writes (pgvector episodic storage)
# ---------------------------------------------------------------------------
class TestMemoryPath:
    """redact_text must strip secrets from summaries before the pgvector write."""

    def test_memory_summary_with_key_redacted(self) -> None:
        raw_summary = (
            f"User discussed issue #42. They pasted their token: {FAKE_OPENAI}. "
            "Resolved by pointing to the MONAI docs."
        )
        stored = redact_text(raw_summary)
        assert FAKE_OPENAI not in stored

    def test_memory_summary_without_key_preserved(self) -> None:
        summary = "User asked about data augmentation strategies for CT scans."
        assert redact_text(summary) == summary

    def test_jwt_in_memory_redacted(self) -> None:
        raw = f"User sent auth header: {FAKE_JWT}"
        assert FAKE_JWT not in redact_text(raw)
