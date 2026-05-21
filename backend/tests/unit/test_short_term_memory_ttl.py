"""Short-term memory (Redis) TTL tests.

MemoryService stores conversation history under the key
``conv:{conversation_id}:history`` with a 24-hour TTL.

These tests verify:
- Empty list is returned when the key is absent (new conversation or TTL expired).
- Messages are appended in order and the TTL is reset on every write.
- Clearing history deletes the Redis key.
- The history key format matches the expected pattern.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.memory import _HISTORY_TTL_SECONDS, MemoryService

_CONV_ID = "test-conv-001"
_HISTORY_KEY = f"conv:{_CONV_ID}:history"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.get.return_value = None
    return redis


@pytest.fixture
def memory_service(mock_redis: AsyncMock) -> MemoryService:
    return MemoryService(
        redis=mock_redis,
        session_factory=MagicMock(),
        embedder=MagicMock(),
    )


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    async def test_get_history_returns_empty_list_when_key_missing(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        # Arrange
        mock_redis.get.return_value = None

        # Act
        result = await memory_service.get_history(_CONV_ID)

        # Assert
        assert result == []
        mock_redis.get.assert_awaited_once_with(_HISTORY_KEY)

    async def test_get_history_returns_empty_list_after_ttl_expiry(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        # Simulate TTL expiry: Redis returns None (key gone)
        mock_redis.get.return_value = None

        result = await memory_service.get_history(_CONV_ID)

        assert result == []

    async def test_get_history_returns_stored_messages(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        stored = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_redis.get.return_value = json.dumps(stored)

        result = await memory_service.get_history(_CONV_ID)

        assert result == stored

    async def test_get_history_handles_corrupt_json_gracefully(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = "not-valid-json{{{"

        result = await memory_service.get_history(_CONV_ID)

        assert result == []


# ---------------------------------------------------------------------------
# append_message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    async def test_append_message_stores_message_in_redis(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        # Arrange — empty history
        mock_redis.get.return_value = None

        # Act
        await memory_service.append_message(_CONV_ID, "user", "Hello, bot!")

        # Assert — redis.set was called with valid JSON
        mock_redis.set.assert_awaited_once()
        key_arg, json_arg = mock_redis.set.call_args.args
        assert key_arg == _HISTORY_KEY
        stored = json.loads(json_arg)
        assert stored == [{"role": "user", "content": "Hello, bot!"}]

    async def test_append_message_sets_ttl_on_every_write(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        # Every write must reset the TTL so active conversations don't expire mid-flow.
        mock_redis.get.return_value = None

        await memory_service.append_message(_CONV_ID, "user", "first")

        mock_redis.set.assert_awaited_once()
        kwargs = mock_redis.set.call_args.kwargs
        assert kwargs["ex"] == _HISTORY_TTL_SECONDS

    async def test_append_message_preserves_existing_history(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        existing = [{"role": "user", "content": "Hello"}]
        mock_redis.get.return_value = json.dumps(existing)

        await memory_service.append_message(_CONV_ID, "assistant", "Hi!")

        _, json_arg = mock_redis.set.call_args.args
        stored = json.loads(json_arg)
        assert len(stored) == 2
        assert stored[0]["content"] == "Hello"
        assert stored[1]["role"] == "assistant"
        assert stored[1]["content"] == "Hi!"

    async def test_append_message_uses_correct_key_format(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        mock_redis.get.return_value = None
        custom_conv_id = "my-unique-conversation"

        await memory_service.append_message(custom_conv_id, "user", "msg")

        set_key = mock_redis.set.call_args.args[0]
        assert set_key == f"conv:{custom_conv_id}:history"


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------


class TestClearHistory:
    async def test_clear_history_deletes_redis_key(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        await memory_service.clear_history(_CONV_ID)

        mock_redis.delete.assert_awaited_once_with(_HISTORY_KEY)

    async def test_clear_history_is_idempotent_when_key_missing(
        self, memory_service: MemoryService, mock_redis: AsyncMock
    ) -> None:
        # Redis DELETE on a missing key is a no-op — service must not raise.
        mock_redis.delete.return_value = 0  # Redis returns 0 when key did not exist

        await memory_service.clear_history(_CONV_ID)  # must not raise

        mock_redis.delete.assert_awaited_once()


# ---------------------------------------------------------------------------
# TTL constant
# ---------------------------------------------------------------------------


class TestTTLConstant:
    def test_history_ttl_is_24_hours(self) -> None:
        assert _HISTORY_TTL_SECONDS == 86_400
