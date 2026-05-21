"""JWT token management in Streamlit session_state."""

from __future__ import annotations

import streamlit as st

from lib.api_client import APIClient

_TOKEN_KEY = "jwt_token"
_USER_KEY = "current_user"


def get_api_client() -> APIClient:
    """Return an APIClient, restoring token from session_state if present."""
    import os

    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    client = APIClient(base_url)
    token = st.session_state.get(_TOKEN_KEY)
    if token:
        client.set_token(token)
    return client


def do_login(email: str, password: str) -> bool:
    """Attempt login; store token in session_state. Returns True on success."""
    client = APIClient(_api_base_url())
    try:
        data = client.login(email, password)
        st.session_state[_TOKEN_KEY] = data["access_token"]
        client.set_token(data["access_token"])
        me = client.me()
        st.session_state[_USER_KEY] = me
        return True
    except Exception:  # noqa: BLE001
        return False


def do_logout() -> None:
    st.session_state.pop(_TOKEN_KEY, None)
    st.session_state.pop(_USER_KEY, None)


def current_user() -> dict[str, object] | None:
    return st.session_state.get(_USER_KEY)  # type: ignore[return-value]


def is_admin() -> bool:
    user = current_user()
    if not user:
        return False
    return bool(user.get("is_superuser", False))


def require_auth() -> bool:
    """Render login form if not authenticated. Returns True if authenticated."""
    if st.session_state.get(_TOKEN_KEY):
        return True

    st.title("Maintainer's Copilot")
    st.subheader("Sign in")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if do_login(email, password):
            st.rerun()
        else:
            st.error("Invalid email or password")

    return False


def _api_base_url() -> str:
    import os

    return os.environ.get("API_BASE_URL", "http://localhost:8000")
