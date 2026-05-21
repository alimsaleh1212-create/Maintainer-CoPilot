"""Embed route for the React widget iframe.

``GET /embed?widget_id=wgt_*`` looks up the widget's allowed origins and
serves the widget HTML shell with the correct CSP ``frame-ancestors`` header.

If the widget is not found or disabled, returns 404.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_widget_service
from app.domain.errors import NotFoundError
from app.repositories.models import Widget
from app.services.widget import WidgetService

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["embed"])


def _build_frame_ancestors(allowed_origins: list[str]) -> str:
    """Build the CSP frame-ancestors value from a list of origins.

    Args:
        allowed_origins: Hostnames/origins configured on the widget.

    Returns:
        Space-joined string suitable for the ``frame-ancestors`` directive.
    """
    if not allowed_origins:
        return "'none'"
    return " ".join(allowed_origins)


def _widget_html_shell(public_widget_id: str, api_origin: str = "") -> str:
    """Return the minimal HTML shell that bootstraps the React widget bundle.

    The bundle URL is intentionally relative so it works behind any reverse
    proxy.  In production the bundle is served from the static demo-host; in
    local dev it is served by the Vite dev server.

    Args:
        public_widget_id: Widget identifier injected into the bootstrap script.
        api_origin: Base URL of the backend API (used by the widget JS client).

    Returns:
        Complete HTML string for the iframe page.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Maintainer's Copilot Widget</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ height: 100vh; overflow: hidden; font-family: system-ui, sans-serif; }}
    #root {{ height: 100%; }}
  </style>
</head>
<body>
  <div id="root"></div>
  <script>
    window.__WIDGET_CONFIG__ = {{
      widgetId: {public_widget_id!r},
      apiOrigin: {api_origin!r}
    }};
  </script>
  <script type="module" src="/static/widget/widget.js"></script>
</body>
</html>"""


@router.get("/embed", response_class=HTMLResponse)
async def embed(
    widget_id: str = Query(..., description="Public widget ID (wgt_* format)"),
    api_host: str = Query(default="", description="Backend API base URL forwarded by the loader"),
    widget_service: WidgetService = Depends(get_widget_service),
) -> HTMLResponse:
    """Serve the widget HTML shell with appropriate CSP headers.

    The ``frame-ancestors`` CSP directive is derived from the widget's
    ``allowed_origins`` field, preventing the widget from being embedded on
    unauthorised sites.

    Args:
        widget_id: Public widget identifier (``wgt_*`` format).
        api_host: API base URL forwarded by the loader script (``data-api-host``).
        widget_service: Widget CRUD service (injected by FastAPI).

    Returns:
        HTMLResponse with the widget shell and CSP headers.

    Raises:
        NotFoundError: Widget not found or disabled.
    """
    widget: Widget | None = await widget_service.get_widget(widget_id)

    if widget is None or not widget.enabled:
        raise NotFoundError(f"Widget {widget_id} not found or is disabled")

    frame_ancestors = _build_frame_ancestors(widget.allowed_origins)
    csp = (
        f"frame-ancestors {frame_ancestors}; "
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline';"
    )

    # CORS: allow XHR from all allowed origins so the widget JS can call the API.
    cors_origin = ", ".join(widget.allowed_origins) if widget.allowed_origins else ""

    headers: dict[str, str] = {
        "Content-Security-Policy": csp,
        "X-Frame-Options": "ALLOWALL",  # CSP frame-ancestors takes precedence in modern browsers.
        "X-Content-Type-Options": "nosniff",
    }
    if cors_origin:
        headers["Access-Control-Allow-Origin"] = cors_origin

    html = _widget_html_shell(
        public_widget_id=widget.public_widget_id,
        api_origin=api_host,
    )

    logger.info(
        "embed.served",
        widget_id=widget_id,
        allowed_origins=widget.allowed_origins,
        frame_ancestors=frame_ancestors,
    )

    return HTMLResponse(content=html, headers=headers)
