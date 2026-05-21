"""CSP frame-ancestors tests for the /embed route.

The /embed endpoint must:
- Return ``Content-Security-Policy: frame-ancestors <origins>`` derived from the
  widget's ``allowed_origins`` DB field (not a hardcoded env var).
- Return 404 when the widget does not exist or is disabled.
- Return ``frame-ancestors 'none'`` when allowed_origins is empty.

These are pure-unit tests: the database is never touched.  WidgetService is
injected via FastAPI dependency override.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_widget_service
from app.api.exceptions import add_exception_handlers
from app.api.routes.embed import _build_frame_ancestors
from app.api.routes.embed import router as embed_router
from app.repositories.models import Widget

# ---------------------------------------------------------------------------
# Pure-function tests for _build_frame_ancestors
# ---------------------------------------------------------------------------


class TestBuildFrameAncestors:
    def test_single_origin_returned_as_is(self) -> None:
        result = _build_frame_ancestors(["https://allowed.example.com"])
        assert result == "https://allowed.example.com"

    def test_multiple_origins_joined_with_space(self) -> None:
        result = _build_frame_ancestors(["https://a.com", "https://b.com"])
        assert result == "https://a.com https://b.com"

    def test_empty_list_returns_none_directive(self) -> None:
        result = _build_frame_ancestors([])
        assert result == "'none'"

    def test_order_preserved(self) -> None:
        origins = ["https://first.com", "https://second.com", "https://third.com"]
        result = _build_frame_ancestors(origins)
        parts = result.split()
        assert parts == origins


# ---------------------------------------------------------------------------
# Helpers for building fake Widget objects
# ---------------------------------------------------------------------------


def _make_widget(
    allowed_origins: list[str],
    enabled: bool = True,
    public_widget_id: str = "wgt_test1234",
) -> Widget:
    """Build a Widget ORM-shaped MagicMock without hitting the database."""
    widget = MagicMock(spec=Widget)
    widget.id = uuid.uuid4()
    widget.public_widget_id = public_widget_id
    widget.allowed_origins = allowed_origins
    widget.enabled = enabled
    return widget


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_widget_service() -> AsyncMock:
    """Async mock of WidgetService.get_widget."""
    svc = AsyncMock()
    svc.get_widget = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def embed_client(fake_widget_service: AsyncMock) -> TestClient:
    """TestClient for a minimal FastAPI app that only mounts the embed router."""
    app = FastAPI()
    add_exception_handlers(app)
    app.include_router(embed_router)
    app.dependency_overrides[get_widget_service] = lambda: fake_widget_service
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# /embed route — CSP header derived from widget.allowed_origins
# ---------------------------------------------------------------------------


class TestEmbedCSPHeader:
    def test_embed_returns_csp_header_with_allowed_origins(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        # Arrange
        widget = _make_widget(["https://allowed.example.com"])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        # Act
        response = embed_client.get("/embed?widget_id=wgt_test1234")

        # Assert
        assert response.status_code == 200
        csp = response.headers["content-security-policy"]
        assert "frame-ancestors https://allowed.example.com" in csp

    def test_embed_csp_contains_all_allowed_origins(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget(["https://a.com", "https://b.com"])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_test1234")

        csp = response.headers["content-security-policy"]
        assert "https://a.com" in csp
        assert "https://b.com" in csp

    def test_embed_csp_is_none_when_allowed_origins_empty(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget([])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_test1234")

        assert response.status_code == 200
        csp = response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" in csp

    def test_embed_cors_header_set_from_allowed_origins(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget(["https://trusted.com"])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_test1234")

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://trusted.com"

    def test_embed_no_cors_header_when_no_origins(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget([])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_test1234")

        # No allowed origins → no CORS header
        assert "access-control-allow-origin" not in response.headers


# ---------------------------------------------------------------------------
# /embed route — 404 paths
# ---------------------------------------------------------------------------


class TestEmbedNotFound:
    def test_embed_returns_404_when_widget_not_found(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        fake_widget_service.get_widget = AsyncMock(return_value=None)

        response = embed_client.get("/embed?widget_id=wgt_nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert data["code"] == "not_found"

    def test_embed_returns_404_when_widget_disabled(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget(["https://allowed.com"], enabled=False)
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_disabled")

        assert response.status_code == 404

    def test_embed_404_response_has_no_stack_trace(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        fake_widget_service.get_widget = AsyncMock(return_value=None)

        response = embed_client.get("/embed?widget_id=wgt_missing")

        assert response.status_code == 404
        assert "Traceback" not in response.text
        assert "traceback" not in response.text


# ---------------------------------------------------------------------------
# /embed route — response body
# ---------------------------------------------------------------------------


class TestEmbedResponseBody:
    def test_embed_returns_html_content_type(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget(["https://ok.com"])
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_test1234")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_embed_html_contains_widget_id(
        self, embed_client: TestClient, fake_widget_service: AsyncMock
    ) -> None:
        widget = _make_widget(["https://ok.com"], public_widget_id="wgt_mypublic")
        fake_widget_service.get_widget = AsyncMock(return_value=widget)

        response = embed_client.get("/embed?widget_id=wgt_mypublic")

        assert response.status_code == 200
        assert "wgt_mypublic" in response.text
