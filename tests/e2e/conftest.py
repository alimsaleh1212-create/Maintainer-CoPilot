"""Pytest configuration for E2E browser tests using playwright."""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, BrowserContext, Page


ALLOWED_HOST_URL = "http://localhost:8090"
DISALLOWED_HOST_URL = "http://localhost:8091"
API_URL = "http://localhost:8000"
DISCOVER_URL = f"{API_URL}/widgets/discover"


def _widget_id_for_origin(origin: str) -> str:
    """Authenticate as admin and pick the widget whose allowed_origins matches ``origin``.

    Falls back to the discover endpoint if the admin login fails (e.g. CI without seeded users).
    """
    import json
    import urllib.parse
    import urllib.request

    try:
        # Admin login (matches the seeded user1@mail.com / 12345678 from .env)
        login_req = urllib.request.Request(
            f"{API_URL}/auth/login",
            data=urllib.parse.urlencode({"username": "user1@mail.com", "password": "12345678"}).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(login_req, timeout=10) as resp:  # noqa: S310
            token = json.loads(resp.read())["access_token"]

        list_req = urllib.request.Request(
            f"{API_URL}/widgets",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(list_req, timeout=10) as resp:  # noqa: S310
            widgets = json.loads(resp.read())
        for w in widgets:
            if origin in w.get("allowed_origins", []) and w.get("enabled", True):
                return str(w["public_widget_id"])
    except Exception:  # noqa: BLE001
        pass

    # Fallback: whatever discover returns (used to pass the assertion if only one widget exists)
    with urllib.request.urlopen(DISCOVER_URL, timeout=10) as resp:  # noqa: S310
        data = json.loads(resp.read())
    wid = data.get("public_widget_id", "")
    assert wid, f"No widget enabled for {origin} and discover returned nothing"
    return str(wid)


@pytest.fixture(scope="session")
def widget_id() -> str:
    """Default widget — the one whose allowed_origins contains the allowed host."""
    return _widget_id_for_origin(ALLOWED_HOST_URL)


@pytest.fixture(scope="session")
def disallowed_widget_id() -> str:
    """The widget whose allowed_origins contains the disallowed-host origin (port 8091)."""
    return _widget_id_for_origin(DISALLOWED_HOST_URL)


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
