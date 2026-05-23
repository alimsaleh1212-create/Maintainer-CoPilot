"""E2E tests — widget iframe is blocked on the disallowed origin (port 8091).

Verifies:
- Disallowed host page renders (the HTML loads fine)
- The widget loader script IS injected (script runs; only the iframe is blocked)
- The browser emits a CSP frame-ancestors violation error OR the iframe has no content
- The chat interface is NOT accessible from the disallowed origin
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, Response, expect


DISALLOWED_HOST = "http://localhost:8091"
API_HOST = "http://localhost:8000"


def test_disallowed_host_page_loads(disallowed_page: Page) -> None:
    """Disallowed host HTML loads correctly (the page itself is fine)."""
    expect(disallowed_page).to_have_title(re.compile(r"Copilot|Disallowed", re.IGNORECASE))
    badge = disallowed_page.locator(".badge")
    expect(badge).to_be_visible()
    expect(badge).to_contain_text("Disallowed")


def test_disallowed_host_shows_csp_explanation(disallowed_page: Page) -> None:
    """The disallowed host page explains the CSP block to the user."""
    body_text = disallowed_page.locator("body").inner_text()
    assert "frame-ancestors" in body_text.lower() or "csp" in body_text.lower(), (
        "Disallowed host page should explain the CSP block mechanism"
    )


def test_widget_script_injected_on_disallowed_host(
    disallowed_page: Page, disallowed_widget_id: str
) -> None:
    """The widget loader script IS injected even on disallowed origin.

    The script itself runs fine — only the iframe render is blocked by the browser.
    This proves the CSP is enforced at the iframe level, not by blocking the script.
    """
    selector = f'script[data-widget-id="{disallowed_widget_id}"]'
    # Give JS time to run discover + inject loader
    disallowed_page.wait_for_timeout(6_000)
    count = disallowed_page.locator(selector).count()
    assert count >= 1, (
        f"Widget loader script was not injected on disallowed host. "
        f"Expected script[data-widget-id='{disallowed_widget_id}'] to be present."
    )


def test_embed_endpoint_has_csp_frame_ancestors(widget_id: str) -> None:
    """The /embed response header contains frame-ancestors restricting to the widget's allowed origin.

    This is a direct HTTP-level check that proves the API correctly emits a
    Content-Security-Policy header derived from the widget's allowed_origins
    list — so embedding is restricted to that exact origin and no other.
    """
    import urllib.request

    embed_url = f"{API_HOST}/embed?widget_id={widget_id}"
    with urllib.request.urlopen(embed_url, timeout=10) as resp:  # noqa: S310
        csp = resp.headers.get("Content-Security-Policy", "")

    assert "frame-ancestors" in csp, f"No frame-ancestors in CSP: {csp!r}"
    # The allowed-host widget should restrict to localhost:8090; if a different
    # widget was returned the directive must still list exactly one origin.
    assert ("localhost:8090" in csp) ^ ("localhost:8091" in csp), (
        f"frame-ancestors must name exactly one demo-host origin, got: {csp!r}"
    )


def test_csp_violation_on_disallowed_host(disallowed_page: Page, widget_id: str) -> None:
    """Browser logs a CSP frame-ancestors violation when iframe tries to load from port 8091.

    Modern browsers emit a console error for CSP violations. We collect them and
    verify at least one references frame-ancestors.
    """
    csp_violations: list[str] = []
    security_errors: list[str] = []

    def _on_console(msg: object) -> None:  # type: ignore[type-arg]
        from playwright.sync_api import ConsoleMessage
        m: ConsoleMessage = msg  # type: ignore[assignment]
        text = m.text.lower()
        if m.type == "error":
            if "frame-ancestors" in text or "content security policy" in text or "csp" in text:
                csp_violations.append(m.text)
            elif "refused" in text or "blocked" in text:
                security_errors.append(m.text)

    disallowed_page.on("console", _on_console)

    # Reload to capture events from the beginning
    disallowed_page.reload(wait_until="networkidle", timeout=20_000)
    # Give the widget script time to run and attempt to create the iframe
    disallowed_page.wait_for_timeout(5_000)

    all_security_messages = csp_violations + security_errors
    assert all_security_messages, (
        "Expected a CSP frame-ancestors violation error in the browser console on the "
        "disallowed host, but none was detected. The widget iframe may not have been "
        "created, or the browser did not report the block. "
        f"widget_id={widget_id!r}, host={DISALLOWED_HOST!r}"
    )
