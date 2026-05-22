"""Widget CRUD service.

Widgets are per-maintainer chatbot embeds.  Each widget has a public ID
(``wgt_<8-char-random>``), a set of allowed origins enforced by CSP, a
greeting, theme config, and the list of tools the embedded chat may invoke.

Every mutating operation writes an AuditLog row.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.errors import NotFoundError, PermissionDenied
from app.repositories.models import AuditLog, Widget

logger = structlog.get_logger(__name__)


def _generate_public_widget_id() -> str:
    """Return a unique, URL-safe public widget identifier.

    Format: ``wgt_<8 random hex chars>`` (e.g. ``wgt_3f8a1c7b``).
    """
    return f"wgt_{secrets.token_hex(4)}"


class WidgetService:
    """CRUD operations for embeddable chat widgets.

    Args:
        session_factory: SQLAlchemy async session factory from app.state.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_widget(
        self,
        owner_id: uuid.UUID,
        allowed_origins: list[str],
        greeting: str = "Hi! How can I help?",
        theme: dict[str, Any] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> Widget:
        """Create a new embeddable widget.

        Args:
            owner_id: User who owns this widget.
            allowed_origins: Hostnames/origins allowed to embed the widget
                (enforced via CSP ``frame-ancestors`` header).
            greeting: Initial bot message shown to widget visitors.
            theme: Optional JSON blob of visual theme overrides.
            enabled_tools: Tools available in this embed.  Defaults to the
                four standard tools if omitted.

        Returns:
            The newly created Widget ORM instance.
        """
        if theme is None:
            theme = {}
        if enabled_tools is None:
            enabled_tools = ["classify", "ner", "summarize", "rag_search"]
        # Normalise origins: strip trailing slashes so CSP frame-ancestors matches correctly.
        allowed_origins = [o.rstrip("/") for o in allowed_origins]

        async with self._session_factory() as session:
            widget = Widget(
                id=uuid.uuid4(),
                public_widget_id=_generate_public_widget_id(),
                owner_id=owner_id,
                allowed_origins=allowed_origins,
                greeting=greeting,
                theme=theme,
                enabled_tools=enabled_tools,
                enabled=True,
            )
            session.add(widget)

            audit = AuditLog(
                id=uuid.uuid4(),
                actor_id=owner_id,
                action="widget_create",
                target_type="widget",
                target_id=widget.id,
                extra_metadata={
                    "public_widget_id": widget.public_widget_id,
                    "allowed_origins": allowed_origins,
                },
            )
            session.add(audit)

            await session.commit()
            await session.refresh(widget)

        logger.info(
            "widget.created",
            widget_id=str(widget.id),
            public_widget_id=widget.public_widget_id,
            owner_id=str(owner_id),
        )
        return widget

    async def get_widget(self, public_widget_id: str) -> Widget | None:
        """Return a widget by public ID, or None if not found.

        Args:
            public_widget_id: The ``wgt_*`` identifier embedded in the iframe src.

        Returns:
            Widget ORM instance or None.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Widget).where(Widget.public_widget_id == public_widget_id)
            )
            widget: Widget | None = result.scalar_one_or_none()
        return widget

    async def get_widget_by_id(self, widget_id: uuid.UUID) -> Widget | None:
        """Return a widget by internal UUID, or None if not found.

        Args:
            widget_id: Internal primary key.

        Returns:
            Widget ORM instance or None.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(Widget).where(Widget.id == widget_id))
            widget: Widget | None = result.scalar_one_or_none()
        return widget

    async def update_widget(
        self,
        widget_id: uuid.UUID,
        owner_id: uuid.UUID,
        **kwargs: Any,
    ) -> Widget:
        """Update mutable widget fields, verifying ownership first.

        Args:
            widget_id: Internal primary key of the widget to update.
            owner_id: Must match the widget's ``owner_id`` field.
            **kwargs: Fields to update (allowed_origins, greeting, theme,
                enabled_tools, enabled).

        Returns:
            The updated Widget ORM instance.

        Raises:
            NotFoundError: Widget does not exist.
            PermissionDenied: Caller does not own the widget.
        """
        allowed_fields = {"allowed_origins", "greeting", "theme", "enabled_tools", "enabled"}
        update_data = {k: v for k, v in kwargs.items() if k in allowed_fields}

        async with self._session_factory() as session:
            result = await session.execute(select(Widget).where(Widget.id == widget_id))
            widget: Widget | None = result.scalar_one_or_none()

            if widget is None:
                raise NotFoundError(f"Widget {widget_id} not found")

            if widget.owner_id != owner_id:
                raise PermissionDenied("You do not own this widget")

            for field, value in update_data.items():
                setattr(widget, field, value)

            audit = AuditLog(
                id=uuid.uuid4(),
                actor_id=owner_id,
                action="widget_update",
                target_type="widget",
                target_id=widget_id,
                extra_metadata={"updated_fields": list(update_data.keys())},
            )
            session.add(audit)

            await session.commit()
            await session.refresh(widget)

        logger.info(
            "widget.updated",
            widget_id=str(widget_id),
            owner_id=str(owner_id),
            fields=list(update_data.keys()),
        )
        return widget

    async def first_enabled_widget(self, origin: str | None = None) -> Widget | None:
        """Return the most recently created enabled widget for ``origin``.

        Used by the public ``GET /widgets/discover`` demo endpoint so each
        demo host page picks the widget whose ``allowed_origins`` lists it
        (rather than any random enabled widget — that produced CSP blocks
        when multiple host pages existed).

        When ``origin`` is None, falls back to the most recent enabled widget
        across all origins so single-widget setups still work.

        Returns:
            Most recent enabled Widget matching ``origin``, or None.
        """
        async with self._session_factory() as session:
            stmt = (
                select(Widget)
                .where(Widget.enabled == True)  # noqa: E712
                .order_by(Widget.created_at.desc())
            )
            result = await session.execute(stmt)
            widgets = list(result.scalars())
            if not widgets:
                return None
            if origin is None:
                return widgets[0]
            normalized = origin.rstrip("/")
            for w in widgets:
                if normalized in (w.allowed_origins or []):
                    return w
            return None

    async def list_widgets(self, owner_id: uuid.UUID) -> list[Widget]:
        """Return all widgets owned by a user, newest first.

        Args:
            owner_id: Filter by this owner.

        Returns:
            List of Widget ORM instances.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Widget).where(Widget.owner_id == owner_id).order_by(Widget.created_at.desc())
            )
            widgets: list[Widget] = list(result.scalars().all())
        return widgets

    async def delete_widget(
        self,
        widget_id: uuid.UUID,
        owner_id: uuid.UUID,
    ) -> None:
        """Delete a widget, verifying ownership first.

        Args:
            widget_id: Internal primary key of the widget.
            owner_id: Must match the widget's ``owner_id`` field.

        Raises:
            NotFoundError: Widget does not exist.
            PermissionDenied: Caller does not own the widget.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(Widget).where(Widget.id == widget_id))
            widget: Widget | None = result.scalar_one_or_none()

            if widget is None:
                raise NotFoundError(f"Widget {widget_id} not found")

            if widget.owner_id != owner_id:
                raise PermissionDenied("You do not own this widget")

            audit = AuditLog(
                id=uuid.uuid4(),
                actor_id=owner_id,
                action="widget_delete",
                target_type="widget",
                target_id=widget_id,
                extra_metadata={"public_widget_id": widget.public_widget_id},
            )
            session.add(audit)
            await session.delete(widget)
            await session.commit()

        logger.info("widget.deleted", widget_id=str(widget_id), owner_id=str(owner_id))
