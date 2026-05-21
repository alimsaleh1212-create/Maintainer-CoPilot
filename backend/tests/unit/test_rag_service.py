"""Unit tests for RAG service: query expansion, retrieval orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.errors import ToolFailure
from app.services.rag import RAGService, SearchResult, SearchResults


@pytest.fixture
def mock_embedder():
    """Mock embedding model."""
    mock = AsyncMock()
    mock.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return mock


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return AsyncMock()


@pytest.fixture
def mock_retriever():
    """Mock hybrid retriever."""
    mock = AsyncMock()
    mock.retrieve = AsyncMock(
        return_value=[
            MagicMock(
                chunk_id="chunk_1",
                text="GPU memory handling",
                source="docs",
                score=0.95,
                rerank_score=None,
            ),
            MagicMock(
                chunk_id="chunk_2",
                text="Batch size tuning",
                source="issue",
                score=0.87,
                rerank_score=None,
            ),
        ]
    )
    return mock


@pytest.mark.asyncio
async def test_search_happy_path(mock_db_session, mock_embedder, mock_retriever):
    """RAG search with valid query returns ranked chunks."""
    service = RAGService()
    service.retriever = mock_retriever

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        results = await service.search(
            query="How do I handle GPU memory errors?",
            db_session=mock_db_session,
            top_k=5,
        )

    assert isinstance(results, SearchResults)
    assert results.query == "How do I handle GPU memory errors?"
    assert len(results.chunks) == 2
    assert results.chunks[0].chunk_id == "chunk_1"
    assert results.chunks[0].score == 0.95
    assert results.total_retrieved == 2


@pytest.mark.asyncio
async def test_search_empty_results(mock_db_session, mock_embedder):
    """RAG search returns empty list when no chunks found."""
    service = RAGService()
    service.retriever.retrieve = AsyncMock(return_value=[])

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        results = await service.search(
            query="Extremely obscure query that matches nothing",
            db_session=mock_db_session,
        )

    assert results.chunks == []
    assert results.total_retrieved == 0


@pytest.mark.asyncio
async def test_search_respects_top_k(mock_db_session, mock_embedder, mock_retriever):
    """RAG search passes top_k to retriever."""
    service = RAGService()
    service.retriever = mock_retriever

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        await service.search(
            query="test",
            db_session=mock_db_session,
            top_k=10,
        )

    # Verify retriever was called with top_k
    service.retriever.retrieve.assert_called_once()
    call_kwargs = service.retriever.retrieve.call_args[1]
    assert call_kwargs["top_k"] == 10


@pytest.mark.asyncio
async def test_search_query_expansion(mock_db_session, mock_embedder, mock_retriever):
    """RAG search calls query expander."""
    service = RAGService()
    service.retriever = mock_retriever

    with patch(
        "app.services.rag.MultiQueryExpander.expand",
        return_value=["GPU memory", "CUDA OOM", "device memory"],
    ) as mock_expand:
        with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
            results = await service.search(
                query="GPU memory error",
                db_session=mock_db_session,
            )

    mock_expand.assert_called_once_with("GPU memory error", None)
    assert results.query_variations == ["GPU memory", "CUDA OOM", "device memory"]


@pytest.mark.asyncio
async def test_search_expansion_failure_raises_tool_failure(mock_db_session):
    """RAG search raises ToolFailure if expansion fails."""
    service = RAGService()

    with patch(
        "app.services.rag.MultiQueryExpander.expand",
        side_effect=ValueError("Expansion failed"),
    ):
        with pytest.raises(ToolFailure) as exc_info:
            await service.search(
                query="test",
                db_session=mock_db_session,
            )

        assert exc_info.value.retryable is True
        assert "RAG search failed" in exc_info.value.message


@pytest.mark.asyncio
async def test_search_retrieval_failure_raises_tool_failure(
    mock_db_session, mock_embedder, mock_retriever
):
    """RAG search raises ToolFailure if retrieval fails."""
    service = RAGService()
    service.retriever.retrieve = AsyncMock(side_effect=Exception("DB error"))

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        with pytest.raises(ToolFailure) as exc_info:
            await service.search(
                query="test",
                db_session=mock_db_session,
            )

        assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_search_result_conversion(mock_db_session, mock_embedder, mock_retriever):
    """RAG search converts RetrievedChunk to SearchResult correctly."""
    service = RAGService()
    service.retriever = mock_retriever

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        results = await service.search(
            query="test",
            db_session=mock_db_session,
        )

    # Verify conversion
    result = results.chunks[0]
    assert isinstance(result, SearchResult)
    assert result.metadata == {"source_type": "docs"}
    assert result.score == 0.95  # Uses original score (no rerank_score)


@pytest.mark.asyncio
async def test_search_uses_rerank_score_when_available(
    mock_db_session, mock_embedder, mock_retriever
):
    """RAG search uses rerank_score if available."""
    mock_retriever.retrieve = AsyncMock(
        return_value=[
            MagicMock(
                chunk_id="chunk_1",
                text="Test",
                source="docs",
                score=0.8,
                rerank_score=0.95,
            ),
        ]
    )
    service = RAGService()
    service.retriever = mock_retriever

    with patch("app.services.rag.get_embedding_model", return_value=mock_embedder):
        results = await service.search(
            query="test",
            db_session=mock_db_session,
        )

    # Should use rerank_score (0.95) not original score (0.8)
    assert results.chunks[0].score == 0.95
