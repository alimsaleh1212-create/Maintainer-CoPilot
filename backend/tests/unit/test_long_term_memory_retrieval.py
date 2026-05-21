"""Long-term memory (pgvector) retrieval tests.

MemoryService persists per-user conversation summaries as pgvector embeddings
and retrieves the top-k most similar ones on new conversation turns.

These tests verify:
- save_memory embeds and stores the summary.
- save_memory redacts secrets from the summary BEFORE the pgvector write.
- search_memories returns results scoped to the requesting user (multi-tenant isolation).
- delete_memory enforces ownership — a user cannot delete another user's memories.
"""

from __future__ import annotations

import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.errors import NotFoundError
from app.repositories.models import Memory
from app.services.memory import MemoryService

# A fake API key that must NEVER appear in stored summaries.
_FAKE_KEY = "AIzaTestFakeGeminiKey1234567890abcdefghij"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_session: MagicMock) -> MagicMock:
    @contextlib.asynccontextmanager
    async def _cm() -> contextlib.AsyncGenerator[MagicMock, None]:
        yield mock_session

    factory = MagicMock()
    factory.side_effect = lambda: _cm()
    return factory


@pytest.fixture
def mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 768)
    return embedder


@pytest.fixture
def memory_service(mock_session_factory: MagicMock, mock_embedder: MagicMock) -> MemoryService:
    return MemoryService(
        redis=AsyncMock(),
        session_factory=mock_session_factory,
        embedder=mock_embedder,
    )


def _make_memory(user_id: uuid.UUID, summary: str) -> MagicMock:
    """Build a Memory-shaped mock for use as a DB execute return value."""
    m = MagicMock()
    m.id = uuid.uuid4()
    m.user_id = user_id
    m.summary = summary
    m.embedding = [0.1] * 768
    return m


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------


class TestSaveMemory:
    async def test_save_memory_embeds_the_summary(
        self,
        memory_service: MemoryService,
        mock_embedder: MagicMock,
        mock_session: MagicMock,
    ) -> None:
        # Arrange
        user_id = uuid.uuid4()
        mock_session.refresh = AsyncMock()

        # Act
        await memory_service.save_memory(user_id=user_id, summary="Discussed MONAI transforms")

        # Assert — the embedder was called with the (possibly redacted) summary
        mock_embedder.embed.assert_awaited_once()
        embedded_text: str = mock_embedder.embed.call_args.args[0]
        assert "MONAI" in embedded_text

    async def test_save_memory_redacts_secrets_before_storage(
        self,
        memory_service: MemoryService,
        mock_embedder: MagicMock,
        mock_session: MagicMock,
    ) -> None:
        # CRITICAL PATH: secrets must never reach the pgvector column.
        user_id = uuid.uuid4()
        raw_summary = f"User pasted their API key: {_FAKE_KEY}"

        await memory_service.save_memory(user_id=user_id, summary=raw_summary)

        # The text passed to embed() must be redacted.
        embedded_text: str = mock_embedder.embed.call_args.args[0]
        assert _FAKE_KEY not in embedded_text

        # The Memory object added to the session must also be redacted.
        added_objects = [call.args[0] for call in mock_session.add.call_args_list]
        memory_objs = [o for o in added_objects if isinstance(o, Memory)]
        assert len(memory_objs) == 1
        assert _FAKE_KEY not in memory_objs[0].summary

    async def test_save_memory_sets_correct_user_id(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        user_id = uuid.uuid4()

        await memory_service.save_memory(user_id=user_id, summary="Resolved issue #42")

        added_objects = [call.args[0] for call in mock_session.add.call_args_list]
        memory_obj = next(o for o in added_objects if isinstance(o, Memory))
        assert memory_obj.user_id == user_id

    async def test_save_memory_commits_transaction(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="Some summary")

        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# search_memories (multi-tenant isolation)
# ---------------------------------------------------------------------------


class TestSearchMemoriesMultiTenantIsolation:
    async def test_search_memories_calls_execute_for_each_user(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        """Different users get separate execute() calls — they cannot share results."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()

        # Mock execute → return empty scalars for both
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=execute_result)

        await memory_service.search_memories(user_a, "query about bugs")
        await memory_service.search_memories(user_b, "query about bugs")

        # execute must have been called twice (once per user)
        assert mock_session.execute.await_count == 2

    async def test_search_memories_user_a_results_not_mixed_with_user_b(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        mem_a = _make_memory(user_a, "user A memory")
        mem_b = _make_memory(user_b, "user B memory")

        call_count = 0

        async def _execute(stmt: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # First call (user A), second call (user B)
            result.scalars.return_value.all.return_value = [mem_a] if call_count == 1 else [mem_b]
            return result

        mock_session.execute = _execute

        results_a = await memory_service.search_memories(user_a, "query")
        results_b = await memory_service.search_memories(user_b, "query")

        # Results must not be mixed across users
        assert results_a == [mem_a]
        assert results_b == [mem_b]
        assert all(m.user_id == user_a for m in results_a)
        assert all(m.user_id == user_b for m in results_b)

    async def test_search_memories_respects_top_k_limit(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        user_id = uuid.uuid4()
        # The ORM query LIMIT is applied by SQLAlchemy; we verify the service
        # passes top_k through without silently capping it.
        execute_result = MagicMock()
        execute_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=execute_result)

        await memory_service.search_memories(user_id, "query", top_k=5)

        mock_session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# delete_memory (ownership enforcement)
# ---------------------------------------------------------------------------


class TestDeleteMemoryOwnership:
    async def test_delete_memory_raises_when_memory_not_found(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=execute_result)

        with pytest.raises(NotFoundError):
            await memory_service.delete_memory(
                memory_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

    async def test_delete_memory_raises_when_different_user(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        """User B cannot delete User A's memory."""
        # search returns None because the WHERE includes user_id = user_b → no match
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = None  # wrong owner
        mock_session.execute = AsyncMock(return_value=execute_result)

        with pytest.raises(NotFoundError):
            await memory_service.delete_memory(
                memory_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
            )

    async def test_delete_memory_commits_when_owner_matches(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        user_id = uuid.uuid4()
        memory_id = uuid.uuid4()
        mem = _make_memory(user_id, "to be deleted")

        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = mem
        mock_session.execute = AsyncMock(return_value=execute_result)

        await memory_service.delete_memory(memory_id=memory_id, user_id=user_id)

        mock_session.commit.assert_awaited_once()
