"""Pytest configuration for E2E browser tests using playwright."""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, BrowserContext, Page


ALLOWED_HOST_URL = "http://localhost:8090"
DISALLOWED_HOST_URL = "http://localhost:8091"
API_URL = "http://localhost:8000"
DISCOVER_URL = f"{API_URL}/widgets/discover"


@pytest.fixture(scope="session")
def widget_id() -> str:
    """Fetch the public widget ID from the discover endpoint once per session."""
    import urllib.request
    import json

    with urllib.request.urlopen(DISCOVER_URL, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read())
    wid = data.get("public_widget_id", "")
    assert wid, f"No public_widget_id returned by {DISCOVER_URL}. Create a widget in Streamlit admin first."
    return wid


@pytest.fixture
def allowed_page(page: Page) -> Page:
    """Navigate to the allowed host and wait for page to stabilise."""
    page.goto(ALLOWED_HOST_URL, wait_until="networkidle", timeout=15_000)
    return page


@pytest.fixture
def disallowed_page(page: Page) -> Page:
    """Navigate to the disallowed host and wait for page to stabilise."""
    page.goto(DISALLOWED_HOST_URL, wait_until="networkidle", timeout=15_000)
    return page
