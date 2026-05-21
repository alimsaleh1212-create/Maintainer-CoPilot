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


def do_register(email: str, password: str) -> tuple[bool, str]:
    """Attempt registration. Returns (success, error_message)."""
    client = APIClient(_api_base_url())
    try:
        client.register(email, password)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "400" in msg or "422" in msg:
            return False, "Email already registered or password too short (min 8 chars)."
        return False, "Registration failed — check the API is reachable."


def do_logout() -> None:
    """Clear session state to log out."""
    st.session_state.pop(_TOKEN_KEY, None)
    st.session_state.pop(_USER_KEY, None)


def current_user() -> dict[str, object] | None:
    """Return the currently authenticated user dict, or None."""
    return st.session_state.get(_USER_KEY)  # type: ignore[return-value]


def is_admin() -> bool:
    """Return True if the current user has superuser (admin) role."""
    user = current_user()
    if not user:
        return False
    return bool(user.get("is_superuser", False))


def require_auth() -> bool:
    """Render sign-in / sign-up form if not authenticated.

    Returns:
        True if the user is already authenticated (caller can proceed).
        False if showing the auth form (caller should st.stop()).
    """
    if st.session_state.get(_TOKEN_KEY):
        return True

    # ── Branded header ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:2.5rem 0 1rem;">
            <div style="
                display:inline-flex;align-items:center;justify-content:center;
                width:52px;height:52px;border-radius:14px;
                background:linear-gradient(135deg,#22c55e,#16a34a);
                margin-bottom:1rem;
            ">
                <svg width="26" height="26" viewBox="0 0 24 24" fill="none"
                     stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="16 18 22 12 16 6"/>
                    <polyline points="8 6 2 12 8 18"/>
                </svg>
            </div>
            <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;
                        letter-spacing:-0.02em;margin-bottom:0.3rem;">
                Maintainer's Copilot
            </div>
            <div style="font-size:0.875rem;color:#64748b;">
                AI-powered triage for open-source maintainers
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs: Sign in | Sign up ───────────────────────────────────────────────
    tab_login, tab_register = st.tabs(["Sign in", "Create account"])

    # ─── Sign in ──────────────────────────────────────────────────────────────
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter your email and password.")
            elif do_login(email, password):
                st.rerun()
            else:
                st.error("Invalid email or password.")

    # ─── Sign up ──────────────────────────────────────────────────────────────
    with tab_register:
        with st.form("register_form"):
            reg_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
            reg_password = st.text_input(
                "Password",
                type="password",
                placeholder="Min 8 characters",
                key="reg_password",
            )
            reg_confirm = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Repeat password",
                key="reg_confirm",
            )
            reg_submitted = st.form_submit_button("Create account", use_container_width=True)

        if reg_submitted:
            if not reg_email or not reg_password:
                st.error("Email and password are required.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            elif len(reg_password) < 8:
                st.error("Password must be at least 8 characters.")
            else:
                ok, err = do_register(reg_email, reg_password)
                if ok:
                    # Auto-login after successful registration
                    if do_login(reg_email, reg_password):
                        st.success("Account created! Redirecting…")
                        st.rerun()
                    else:
                        st.success("Account created. Please sign in.")
                else:
                    st.error(err)

    return False


def _api_base_url() -> str:
    import os

    return os.environ.get("API_BASE_URL", "http://localhost:8000")
