"""Domain exception hierarchy.

Services raise these; the API boundary handler in app.api.exceptions maps
them to HTTP responses. Never raise HTTPException directly inside services.
"""

from __future__ import annotations


class AppError(RuntimeError):
    """Base class for all domain errors."""

    def __init__(self, message: str, code: str = "internal_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class NotFoundError(AppError):
    """The requested resource does not exist."""

    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, code="not_found")


class PermissionDenied(AppError):
    """The caller does not have permission to perform this action."""

    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__(message, code="permission_denied")


class ToolFailure(AppError):
    """An LLM tool call failed.

    Unlike most errors, ToolFailure is recoverable — the chatbot loop catches
    it and returns a structured ToolError to the LLM so it can decide what
    to do next, rather than propagating a 500.
    """

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message, code="tool_failure")
        self.retryable = retryable


class ValidationError(AppError):
    """Input failed domain-level validation (distinct from Pydantic's)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")
