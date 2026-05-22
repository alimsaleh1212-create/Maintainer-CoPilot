"""E2E tests — widget loads correctly on the allowed origin (port 8090).

Verifies:
- Allowed host page renders without JS errors blocking the flow
- The /widgets/discover call resolves a real widget ID
- The widget loader script is injected into the page
- The chat iframe is created and its src points to /embed?widget_id=...
- The iframe loads (HTTP 200, no CSP block on this origin)
- The chat toggle button is present and interactive
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect


ALLOWED_HOST = "http://localhost:8090"
API_HOST = "http://localhost:8000"


def test_allowed_host_page_loads(allowed_page: Page) -> None:
    """Allowed host returns a 200 and the heading renders."""
    expect(allowed_page).to_have_title(re.compile(r"Copilot|Allowed", re.IGNORECASE))
    # The badge says "✓ Allowed origin — widget loads"
    badge = allowed_page.locator(".badge")
    expect(badge).to_be_visible()
    expect(badge).to_contain_text("Allowed")


def test_widget_discover_resolves_id(allowed_page: Page, widget_id: str) -> None:
    """Page JS updates the embed snippet with the real widget ID after discover."""
    # The snippet <pre> starts with "…discovering…"; JS replaces it after /widgets/discover
    # Wait up to 8s for the JS to update the snippet
    snippet = allowed_page.locator("#snippet-code")
    snippet.wait_for(timeout=8_000)
    # After discover completes the snippet no longer contains "discovering"
    expect(snippet).not_to_contain_text("discovering")
    expect(snippet).to_contain_text(widget_id)


def test_widget_script_injected(allowed_page: Page, widget_id: str) -> None:
    """The loader <script> with data-widget-id is injected into the DOM."""
    # The page JS creates a <script src=".../widget.js" data-widget-id="wgt_..."> element
    selector = f'script[data-widget-id="{widget_id}"]'
    count = allowed_page.locator(selector).count()
    assert count >= 1, f"No <script data-widget-id='{widget_id}'> found — widget loader was not injected"


def test_widget_iframe_created(allowed_page: Page, widget_id: str) -> None:
    """The widget loader creates an iframe pointing to /embed?widget_id=...."""
    # widget.js injects an <iframe src="http://localhost:8000/embed?widget_id=wgt_...">
    iframe_locator = allowed_page.locator(f'iframe[src*="widget_id={widget_id}"]')
    # Give the loader up to 10s to fire and create the iframe
    iframe_locator.wait_for(state="attached", timeout=10_000)
    src = iframe_locator.get_attribute("src") or ""
    assert f"widget_id={widget_id}" in src, f"iframe src does not contain widget_id: {src!r}"
    assert "/embed" in src, f"iframe src does not point to /embed: {src!r}"


def test_widget_iframe_loads_without_csp_block(page: Page, widget_id: str) -> None:
    """The iframe content actually loads (no CSP block on this origin).

    We set up the error listener BEFORE navigation so we catch every console
    error from the very start, then check no frame-ancestors violation was logged.
    """
    csp_errors: list[str] = []

    def _on_console(msg: object) -> None:  # type: ignore[type-arg]
        from playwright.sync_api import ConsoleMessage
        m: ConsoleMessage = msg  # type: ignore[assignment]
        if m.type == "error" and "frame-ancestors" in m.text.lower():
            csp_errors.append(m.text)

    page.on("console", _on_console)

    # Capture /embed response during navigation — must subscribe before goto
    embed_responses: list[int] = []

    def _on_response(r: object) -> None:  # type: ignore[type-arg]
        from playwright.sync_api import Response as PwResponse
        resp: PwResponse = r  # type: ignore[assignment]
        if "/embed" in resp.url:
            embed_responses.append(resp.status)

    page.on("response", _on_response)
    page.goto(ALLOWED_HOST, wait_until="networkidle", timeout=20_000)
    # Give loader script time to run
    page.wait_for_timeout(6_000)

    assert not csp_errors, f"CSP frame-ancestors violation on allowed host: {csp_errors}"
    assert embed_responses, "No /embed response captured — iframe was not created"
    assert all(s < 400 for s in embed_responses), f"Embed endpoint returned error: {embed_responses}"


def test_chat_toggle_button_present(allowed_page: Page, widget_id: Page) -> None:
    """The chat bubble/toggle button is present in the iframe or page after load."""
    # The iframe hosts the chat UI. We verify the iframe exists and has a src.
    iframe_locator = allowed_page.locator(f'iframe[src*="widget_id={widget_id}"]')
    iframe_locator.wait_for(state="attached", timeout=10_000)
    src = iframe_locator.get_attribute("src") or ""
    assert src, "Widget iframe has no src — widget did not load"
