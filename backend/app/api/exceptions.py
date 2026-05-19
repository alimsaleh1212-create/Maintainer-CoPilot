"""API boundary exception handler.

Maps domain errors to HTTP responses. Users never see a stack trace;
they see {code, request_id, message}. Wired in app.api.main.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.domain.errors import (
    AppError,
    NotFoundError,
    PermissionDenied,
    ToolFailure,
    ValidationError,
)

logger = structlog.get_logger(__name__)

_STATUS_MAP: dict[type[AppError], int] = {
    NotFoundError: 404,
    PermissionDenied: 403,
    ToolFailure: 422,
    ValidationError: 422,
}


def _error_response(request: Request, exc: AppError, status: int) -> JSONResponse:
    request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
    logger.warning(
        "domain_error",
        code=exc.code,
        message=exc.message,
        status=status,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status,
        content={"code": exc.code, "message": exc.message, "request_id": request_id},
    )


def add_exception_handlers(app: FastAPI) -> None:
    """Register all domain → HTTP mappings on the FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status = _STATUS_MAP.get(type(exc), 500)
        return _error_response(request, exc, status)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = str(getattr(request.state, "request_id", uuid.uuid4()))
        logger.exception(
            "unhandled_exception",
            exc_type=type(exc).__name__,
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
            },
        )
