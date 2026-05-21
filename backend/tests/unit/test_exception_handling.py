"""Unit tests for exception handling layer.

Verifies domain errors map to correct HTTP status, stack traces are never exposed,
and tool failures are recoverable (not 500).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exceptions import add_exception_handlers
from app.domain.errors import (
    NotFoundError,
    PermissionDenied,
    ToolFailure,
    ValidationError,
)


@pytest.fixture
def app_with_handlers():
    """FastAPI app with exception handlers wired."""
    # Disable exception handling in Starlette so our handlers catch everything
    app = FastAPI(debug=False)
    add_exception_handlers(app)

    # Add test routes that raise various exceptions
    @app.get("/not-found")
    async def not_found() -> None:
        raise NotFoundError("User 123 not found")

    @app.get("/permission-denied")
    async def permission_denied() -> None:
        raise PermissionDenied("Admin role required")

    @app.get("/tool-failure")
    async def tool_failure() -> None:
        raise ToolFailure("Model server timeout", retryable=True)

    @app.get("/validation-error")
    async def validation_error() -> None:
        raise ValidationError("Invalid input: expected integer")

    return app


@pytest.fixture
def client(app_with_handlers):
    """TestClient for the app with handlers."""
    return TestClient(app_with_handlers)


def test_not_found_error_returns_404(client):
    """NotFoundError → 404 with code + message, no stack trace."""
    response = client.get("/not-found")

    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "not_found"
    assert data["message"] == "User 123 not found"
    assert "request_id" in data
    assert "Traceback" not in response.text
    assert "ValueError" not in response.text


def test_permission_denied_error_returns_403(client):
    """PermissionDenied → 403 with code + message, no stack trace."""
    response = client.get("/permission-denied")

    assert response.status_code == 403
    data = response.json()
    assert data["code"] == "permission_denied"
    assert data["message"] == "Admin role required"
    assert "request_id" in data


def test_tool_failure_returns_422(client):
    """ToolFailure → 422 (not 500, tool is recoverable)."""
    response = client.get("/tool-failure")

    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "tool_failure"
    assert data["message"] == "Model server timeout"
    # ToolFailure retryable flag is stored but not exposed in HTTP response


def test_validation_error_returns_422(client):
    """ValidationError → 422 with code + message."""
    response = client.get("/validation-error")

    assert response.status_code == 422
    data = response.json()
    assert data["code"] == "validation_error"
    assert data["message"] == "Invalid input: expected integer"


def test_all_error_responses_have_request_id(client):
    """All error responses include request_id for tracing."""
    endpoints = [
        ("/not-found", 404),
        ("/permission-denied", 403),
        ("/tool-failure", 422),
        ("/validation-error", 422),
    ]

    request_ids = []
    for endpoint, expected_status in endpoints:
        response = client.get(endpoint)
        assert response.status_code == expected_status
        data = response.json()
        assert "request_id" in data
        request_ids.append(data["request_id"])

    # request_ids should be unique (different for each request)
    assert len(request_ids) == len(set(request_ids))


def test_error_response_schema(client):
    """All error responses follow {code, message, request_id} schema."""
    response = client.get("/not-found")

    data = response.json()
    # Exactly these three fields
    assert set(data.keys()) == {"code", "message", "request_id"}
    assert isinstance(data["code"], str)
    assert isinstance(data["message"], str)
    assert isinstance(data["request_id"], str)


def test_tool_failure_vs_validation_error(client):
    """Tool failures (422) and validation errors both map to 422."""
    # Tool failure is recoverable
    response = client.get("/tool-failure")
    assert response.status_code == 422
    assert response.json()["code"] == "tool_failure"

    # Validation error also maps to 422
    response = client.get("/validation-error")
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_permission_denied_not_confused_with_not_found(client):
    """PermissionDenied (403) is distinct from NotFoundError (404)."""
    response = client.get("/permission-denied")
    assert response.status_code == 403
    assert response.json()["code"] == "permission_denied"

    response = client.get("/not-found")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_custom_app_error_message(client):
    """AppError subclass messages are preserved."""
    # NotFoundError with custom message
    response = client.get("/not-found")
    data = response.json()
    assert data["message"] == "User 123 not found"

    # PermissionDenied with custom message
    response = client.get("/permission-denied")
    data = response.json()
    assert data["message"] == "Admin role required"


def test_tool_failure_with_retryable_flag():
    """ToolFailure captures retryable flag for use in chatbot loop."""
    exc = ToolFailure("Model server timeout", retryable=True)
    assert exc.retryable is True
    assert exc.code == "tool_failure"

    exc = ToolFailure("Invalid configuration", retryable=False)
    assert exc.retryable is False


def test_app_error_code_mapping():
    """Domain errors map to correct codes."""
    codes = [
        (NotFoundError("test"), "not_found"),
        (PermissionDenied("test"), "permission_denied"),
        (ToolFailure("test"), "tool_failure"),
        (ValidationError("test"), "validation_error"),
    ]

    for exc, expected_code in codes:
        assert exc.code == expected_code
