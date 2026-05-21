"""Widget admin routes.

CRUD for embeddable chat widget configurations.  Creating and updating
widgets requires authentication.  The ``GET /widgets/{widget_id}/config``
endpoint is public — it is called by the React widget at load time.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import WidgetServiceDep
from app.api.routes.auth import get_current_user
from app.domain.errors import NotFoundError
from app.repositories.models import User, Widget

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/widgets", tags=["widgets"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WidgetCreateRequest(BaseModel):
    """Payload to create a new widget."""

    allowed_origins: list[str] = Field(
        ..., min_length=1, description="Origins allowed to embed this widget"
    )
    greeting: str = Field(
        default="Hi! How can I help?",
        max_length=500,
        description="Initial greeting message",
    )
    theme: dict[str, Any] = Field(default_factory=dict, description="Visual theme overrides")
    enabled_tools: list[str] = Field(
        default_factory=lambda: ["classify", "ner", "summarize", "rag_search"],
        description="Tools available in the embed",
    )


class WidgetUpdateRequest(BaseModel):
    """Payload to update widget fields (all optional)."""

    allowed_origins: list[str] | None = Field(default=None)
    greeting: str | None = Field(default=None, max_length=500)
    theme: dict[str, Any] | None = Field(default=None)
    enabled_tools: list[str] | None = Field(default=None)
    enabled: bool | None = Field(default=None)


class WidgetResponse(BaseModel):
    """Full widget representation (owner view)."""

    id: uuid.UUID
    public_widget_id: str
    owner_id: uuid.UUID
    allowed_origins: list[str]
    greeting: str
    theme: dict[str, Any]
    enabled_tools: list[str]
    enabled: bool

    model_config = {"from_attributes": True}


class WidgetConfigResponse(BaseModel):
    """Public widget config returned to the React widget at load time."""

    public_widget_id: str
    greeting: str
    theme: dict[str, Any]
    enabled_tools: list[str]
    enabled: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=WidgetResponse, status_code=201)
async def create_widget(
    body: WidgetCreateRequest,
    widget_service: WidgetServiceDep,
    current_user: User = Depends(get_current_user),
) -> WidgetResponse:
    """Create a new embeddable widget (authenticated).

    Args:
        body: Widget configuration.
        widget_service: Widget CRUD service.
        current_user: Authenticated owner.

    Returns:
        Full WidgetResponse for the newly created widget.
    """
    widget: Widget = await widget_service.create_widget(
        owner_id=current_user.id,
        allowed_origins=body.allowed_origins,
        greeting=body.greeting,
        theme=body.theme,
        enabled_tools=body.enabled_tools,
    )
    logger.info("widgets.created", widget_id=str(widget.id), user_id=str(current_user.id))
    return WidgetResponse.model_validate(widget)


@router.get("", response_model=list[WidgetResponse])
async def list_widgets(
    widget_service: WidgetServiceDep,
    current_user: User = Depends(get_current_user),
) -> list[WidgetResponse]:
    """List all widgets owned by the authenticated user.

    Args:
        widget_service: Widget CRUD service.
        current_user: Authenticated user.

    Returns:
        List of WidgetResponse objects, newest first.
    """
    widgets = await widget_service.list_widgets(owner_id=current_user.id)
    return [WidgetResponse.model_validate(w) for w in widgets]


@router.get("/{widget_id}/config", response_model=WidgetConfigResponse)
async def get_widget_config(
    widget_id: str,
    widget_service: WidgetServiceDep,
) -> WidgetConfigResponse:
    """Return public widget configuration (no auth required).

    Called by the React widget at load time to fetch greeting, theme, and
    enabled tools.

    Args:
        widget_id: Public widget ID (``wgt_*`` format).
        widget_service: Widget CRUD service.

    Returns:
        WidgetConfigResponse with greeting, theme, and enabled tools.

    Raises:
        NotFoundError: Widget not found.
    """
    widget: Widget | None = await widget_service.get_widget(widget_id)
    if widget is None:
        raise NotFoundError(f"Widget {widget_id} not found")
    return WidgetConfigResponse(
        public_widget_id=widget.public_widget_id,
        greeting=widget.greeting,
        theme=widget.theme,
        enabled_tools=widget.enabled_tools,
        enabled=widget.enabled,
    )


@router.put("/{widget_id}", response_model=WidgetResponse)
async def update_widget(
    widget_id: uuid.UUID,
    body: WidgetUpdateRequest,
    widget_service: WidgetServiceDep,
    current_user: User = Depends(get_current_user),
) -> WidgetResponse:
    """Update a widget (owner only).

    Args:
        widget_id: Internal widget UUID.
        body: Fields to update (partial update — all fields optional).
        widget_service: Widget CRUD service.
        current_user: Authenticated user (must be owner).

    Returns:
        Updated WidgetResponse.

    Raises:
        NotFoundError: Widget not found.
        PermissionDenied: Caller is not the owner.
    """
    updates = body.model_dump(exclude_none=True)
    widget = await widget_service.update_widget(
        widget_id=widget_id,
        owner_id=current_user.id,
        **updates,
    )
    return WidgetResponse.model_validate(widget)


@router.delete("/{widget_id}", status_code=204)
async def delete_widget(
    widget_id: uuid.UUID,
    widget_service: WidgetServiceDep,
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a widget (owner only).

    Args:
        widget_id: Internal widget UUID.
        widget_service: Widget CRUD service.
        current_user: Authenticated user (must be owner).

    Raises:
        NotFoundError: Widget not found.
        PermissionDenied: Caller is not the owner.
    """
    await widget_service.delete_widget(
        widget_id=widget_id,
        owner_id=current_user.id,
    )
    logger.info("widgets.deleted", widget_id=str(widget_id), user_id=str(current_user.id))
    return None
