"""HTTP client for the backend API — pure httpx, no backend imports."""

from __future__ import annotations

from typing import Any

import httpx


class APIClient:
    """Thin httpx wrapper around the backend REST API.

    All calls are synchronous (Streamlit's execution model is sync).
    The base URL is injected at construction — no os.getenv here.
    """

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._token: str | None = None

    # ------------------------------------------------------------------ auth

    def set_token(self, token: str) -> None:
        self._token = token

    def clear_token(self) -> None:
        self._token = None

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _get(self, path: str, **params: Any) -> Any:
        resp = httpx.get(
            f"{self._base}{path}",
            headers=self._headers(),
            params=params or None,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: Any = None, data: Any = None) -> Any:
        resp = httpx.post(
            f"{self._base}{path}",
            headers=self._headers() if json is not None else {},
            json=json,
            data=data,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:
        resp = httpx.delete(
            f"{self._base}{path}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ auth endpoints

    def login(self, email: str, password: str) -> dict[str, Any]:
        """POST /auth/jwt/login — returns {access_token, token_type}."""
        resp = httpx.post(
            f"{self._base}/auth/jwt/login",
            data={"username": email, "password": password},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def register(self, email: str, password: str) -> dict[str, Any]:
        return dict(self._post("/auth/register", {"email": email, "password": password}))

    def me(self) -> dict[str, Any]:
        return dict(self._get("/auth/me"))

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        widget_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if widget_id:
            payload["widget_id"] = widget_id
        return dict(self._post("/chat", payload))

    # ------------------------------------------------------------------ widgets (admin)

    def list_widgets(self) -> list[dict[str, Any]]:
        return list(self._get("/widgets"))

    def create_widget(self, data: dict[str, Any]) -> dict[str, Any]:
        return dict(self._post("/widgets", data))

    def delete_widget(self, widget_id: str) -> dict[str, Any]:
        return dict(self._delete(f"/widgets/{widget_id}"))

    # ------------------------------------------------------------------ memory

    def list_memories(self) -> list[dict[str, Any]]:
        return list(self._get("/memory"))

    def delete_memory(self, memory_id: str) -> dict[str, Any]:
        return dict(self._delete(f"/memory/{memory_id}"))

    # ------------------------------------------------------------------ health

    def health(self) -> dict[str, Any]:
        return dict(self._get("/health"))
