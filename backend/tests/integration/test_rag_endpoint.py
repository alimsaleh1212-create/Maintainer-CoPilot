"""Integration tests for RAG /search endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_db_session
from app.api.main import create_app


@pytest.fixture
def client():
    """FastAPI test client with mocked dependencies."""
    app = create_app()

    # Mock the get_db_session dependency
    async def mock_get_db_session():
        yield AsyncMock()

    app.dependency_overrides[get_db_session] = mock_get_db_session
    return TestClient(app)


@pytest.fixture
def mock_rag_service():
    """Mock RAG service."""
    mock = MagicMock()
    mock.search = AsyncMock()
    return mock


def test_search_endpoint_valid_request(client, mock_rag_service):
    """POST /rag/search with valid query returns 200 + results."""
    from app.services.rag import SearchResult, SearchResults

    # Mock the RAG service
    mock_results = SearchResults(
        query="GPU memory error",
        query_variations=["GPU memory", "CUDA OOM"],
        chunks=[
            SearchResult(
                chunk_id="chunk_1",
                text="Use smaller batch sizes",
                source="docs",
                score=0.95,
                metadata={"source_type": "docs"},
            ),
        ],
        total_retrieved=1,
    )

    with patch(
        "app.api.routes.rag.RAGService.search",
        return_value=mock_results,
    ):
        response = client.post(
            "/rag/search",
            json={"query": "GPU memory error", "top_k": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "GPU memory error"
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "chunk_1"
    assert data["total_retrieved"] == 1


def test_search_endpoint_missing_query(client):
    """POST /rag/search without query returns 422."""
    response = client.post("/rag/search", json={})
    assert response.status_code == 422


def test_search_endpoint_invalid_top_k(client):
    """POST /rag/search with invalid top_k returns 422."""
    response = client.post(
        "/rag/search",
        json={"query": "test", "top_k": 0},  # min 1
    )
    assert response.status_code == 422

    response = client.post(
        "/rag/search",
        json={"query": "test", "top_k": 25},  # max 20
    )
    assert response.status_code == 422


def test_search_endpoint_query_too_long(client):
    """POST /rag/search with very long query still works (no explicit limit)."""
    long_query = "x" * 5000
    with patch(
        "app.api.routes.rag.RAGService.search",
        return_value=MagicMock(
            query=long_query,
            query_variations=[long_query],
            chunks=[],
            total_retrieved=0,
        ),
    ):
        response = client.post(
            "/rag/search",
            json={"query": long_query, "top_k": 5},
        )

    assert response.status_code == 200


def test_search_endpoint_response_schema(client):
    """POST /rag/search response matches SearchResponse schema."""
    from app.services.rag import SearchResult, SearchResults

    mock_results = SearchResults(
        query="test",
        query_variations=["test", "test query"],
        chunks=[
            SearchResult(
                chunk_id="c1",
                text="content",
                source="docs",
                score=0.9,
                metadata={"source_type": "docs"},
            ),
        ],
        total_retrieved=1,
    )

    with patch(
        "app.api.routes.rag.RAGService.search",
        return_value=mock_results,
    ):
        response = client.post(
            "/rag/search",
            json={"query": "test"},
        )

    assert response.status_code == 200
    data = response.json()

    # Verify schema
    assert "query" in data
    assert "query_variations" in data
    assert "chunks" in data
    assert "total_retrieved" in data

    # Verify chunk schema
    chunk = data["chunks"][0]
    assert "chunk_id" in chunk
    assert "text" in chunk
    assert "source" in chunk
    assert "score" in chunk
    assert isinstance(chunk["score"], float)


def test_search_endpoint_default_top_k(client):
    """POST /rag/search without top_k uses default of 5."""
    from app.services.rag import SearchResults

    mock_results = SearchResults(
        query="test",
        query_variations=["test"],
        chunks=[],
        total_retrieved=0,
    )

    with patch(
        "app.api.routes.rag.RAGService.search",
        return_value=mock_results,
    ) as mock_search:
        client.post(
            "/rag/search",
            json={"query": "test"},
        )

        # Verify default top_k was passed
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["top_k"] == 5


def test_search_endpoint_empty_results(client):
    """POST /rag/search can return empty chunk list."""
    from app.services.rag import SearchResults

    mock_results = SearchResults(
        query="obscure query",
        query_variations=["obscure query"],
        chunks=[],
        total_retrieved=0,
    )

    with patch(
        "app.api.routes.rag.RAGService.search",
        return_value=mock_results,
    ):
        response = client.post(
            "/rag/search",
            json={"query": "obscure query"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["chunks"] == []
    assert data["total_retrieved"] == 0
