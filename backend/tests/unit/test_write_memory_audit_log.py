"""Audit log tests for memory writes and deletes.

CLAUDE.md rule: "Do not auto-write memory. The LLM must call `write_memory`
explicitly; every write must produce an audit row."

These tests verify:
- save_memory always creates exactly one AuditLog row per call.
- delete_memory always creates exactly one AuditLog row per call.
- The AuditLog rows carry the correct action, actor_id, target_type.
- The audit row summary_length metadata is redacted-length, not raw-length.
"""

from __future__ import annotations

import contextlib
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.repositories.models import AuditLog, Memory
from app.services.memory import MemoryService

_FAKE_KEY = "AIzaTestFakeGeminiKey1234567890abcdefghij"


# ---------------------------------------------------------------------------
# Fixtures (shared with other memory tests)
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


def _added_audit_logs(mock_session: MagicMock) -> list[AuditLog]:
    """Extract all AuditLog objects passed to session.add()."""
    return [
        call_.args[0]
        for call_ in mock_session.add.call_args_list
        if isinstance(call_.args[0], AuditLog)
    ]


# ---------------------------------------------------------------------------
# save_memory audit log
# ---------------------------------------------------------------------------


class TestSaveMemoryAuditLog:
    async def test_save_memory_creates_exactly_one_audit_row(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="Fixed bug #99")

        audit_rows = _added_audit_logs(mock_session)
        assert len(audit_rows) == 1

    async def test_save_memory_audit_row_action_is_memory_write(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="Closed issue")

        audit_rows = _added_audit_logs(mock_session)
        assert audit_rows[0].action == "memory_write"

    async def test_save_memory_audit_row_target_type_is_memory(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="Some summary")

        audit_rows = _added_audit_logs(mock_session)
        assert audit_rows[0].target_type == "memory"

    async def test_save_memory_audit_row_actor_id_matches_user_id(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        user_id = uuid.uuid4()

        await memory_service.save_memory(user_id=user_id, summary="Some summary")

        audit_rows = _added_audit_logs(mock_session)
        assert audit_rows[0].actor_id == user_id

    async def test_save_memory_audit_row_target_id_matches_memory_id(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="Some summary")

        added_objects = [c.args[0] for c in mock_session.add.call_args_list]
        memory_obj = next(o for o in added_objects if isinstance(o, Memory))
        audit_rows = _added_audit_logs(mock_session)

        assert audit_rows[0].target_id == memory_obj.id

    async def test_save_memory_audit_row_metadata_uses_redacted_length(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        # The summary_length in metadata must reflect the redacted summary,
        # not the raw (potentially longer or shorter) original.
        raw_summary = f"Key used: {_FAKE_KEY}"

        await memory_service.save_memory(user_id=uuid.uuid4(), summary=raw_summary)

        audit_rows = _added_audit_logs(mock_session)
        stored_length: int = audit_rows[0].extra_metadata["summary_length"]

        # The stored length should NOT equal len(raw_summary) because _FAKE_KEY
        # gets replaced by a "[REDACTED:...]" placeholder that has different length.
        assert stored_length != len(raw_summary) or _FAKE_KEY not in raw_summary

    async def test_save_memory_both_memory_and_audit_added_in_same_transaction(
        self,
        memory_service: MemoryService,
        mock_session: MagicMock,
    ) -> None:
        # Both objects are added before commit → single atomic transaction.
        await memory_service.save_memory(user_id=uuid.uuid4(), summary="atomic write")

        added_types = [type(c.args[0]).__name__ for c in mock_session.add.call_args_list]
        assert "Memory" in added_types
        assert "AuditLog" in added_types
        # commit happens exactly once (not once per add)
        mock_session.commit.assert_awaited_once()


# ---------------------------------------------------------------------------
# delete_memory audit log
# ---------------------------------------------------------------------------


class TestDeleteMemoryAuditLog:
    @pytest.fixture
    def mock_session_with_memory(self, mock_session: MagicMock) -> MagicMock:
        """Patch execute to return a valid Memory so delete can proceed."""
        user_id = uuid.uuid4()
        # Use MagicMock so SQLAlchemy instrumentation isn't needed.
        mem = MagicMock()
        mem.id = uuid.uuid4()
        mem.user_id = user_id

        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = mem
        mock_session.execute = AsyncMock(return_value=execute_result)
        mock_session._test_user_id = user_id
        mock_session._test_memory = mem
        return mock_session

    async def test_delete_memory_creates_exactly_one_audit_row(
        self,
        memory_service: MemoryService,
        mock_session_with_memory: MagicMock,
    ) -> None:
        user_id = mock_session_with_memory._test_user_id
        mem = mock_session_with_memory._test_memory

        await memory_service.delete_memory(memory_id=mem.id, user_id=user_id)

        audit_rows = _added_audit_logs(mock_session_with_memory)
        assert len(audit_rows) == 1

    async def test_delete_memory_audit_row_action_is_memory_delete(
        self,
        memory_service: MemoryService,
        mock_session_with_memory: MagicMock,
    ) -> None:
        user_id = mock_session_with_memory._test_user_id
        mem = mock_session_with_memory._test_memory

        await memory_service.delete_memory(memory_id=mem.id, user_id=user_id)

        audit_rows = _added_audit_logs(mock_session_with_memory)
        assert audit_rows[0].action == "memory_delete"

    async def test_delete_memory_audit_row_actor_matches_requesting_user(
        self,
        memory_service: MemoryService,
        mock_session_with_memory: MagicMock,
    ) -> None:
        user_id = mock_session_with_memory._test_user_id
        mem = mock_session_with_memory._test_memory

        await memory_service.delete_memory(memory_id=mem.id, user_id=user_id)

        audit_rows = _added_audit_logs(mock_session_with_memory)
        assert audit_rows[0].actor_id == user_id
