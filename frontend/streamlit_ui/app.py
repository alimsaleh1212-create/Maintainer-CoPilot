"""Maintainer's Copilot — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Maintainer's Copilot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.auth import current_user, do_logout, require_auth  # noqa: E402
from lib.styles import inject_styles, page_header  # noqa: E402

inject_styles()

if not require_auth():
    st.stop()

user = current_user() or {}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1.5rem;border-bottom:1px solid #1e293b;margin-bottom:1.5rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
                <div style="
                    width:36px;height:36px;border-radius:10px;
                    background:linear-gradient(135deg,#22c55e,#16a34a);
                    display:flex;align-items:center;justify-content:center;
                ">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                         stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="16 18 22 12 16 6"/>
                        <polyline points="8 6 2 12 8 18"/>
                    </svg>
                </div>
                <div>
                    <div style="font-weight:700;font-size:0.9rem;color:#f1f5f9;letter-spacing:-0.01em;">
                        Maintainer's Copilot
                    </div>
                    <div style="font-size:0.7rem;color:#22c55e;">v0.1.0</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Nav links
    st.page_link("pages/1_chat.py", label="Chat", icon="💬")
    st.page_link("pages/2_admin_widgets.py", label="Widget Admin", icon="⚙️")
    st.page_link("pages/3_memory_inspector.py", label="Memory Inspector", icon="🧠")

    st.markdown("<div style='flex:1'/>", unsafe_allow_html=True)

    # User info footer
    st.markdown(
        f"""
        <div style="
            margin-top:auto;padding-top:1rem;
            border-top:1px solid #1e293b;margin-bottom:0.5rem;
        ">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="
                    width:32px;height:32px;border-radius:50%;
                    background:linear-gradient(135deg,#334155,#1e293b);
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;color:#94a3b8;font-weight:600;flex-shrink:0;
                ">{str(user.get("email","?"))[0].upper()}</div>
                <div style="overflow:hidden;">
                    <div style="font-size:0.8rem;font-weight:500;color:#e2e8f0;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                        {user.get("email","unknown")}
                    </div>
                    <div style="font-size:0.7rem;color:#64748b;">
                        {"Admin" if user.get("is_superuser") else "User"}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="signout"):
        do_logout()
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────
page_header(
    "Maintainer's Copilot",
    "AI-powered triage, RAG search, and memory for open-source maintainers",
    icon_svg='<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:1.25rem;">
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;
                        color:#64748b;margin-bottom:0.5rem;">Feature</div>
            <div style="font-size:1.5rem;font-weight:700;color:#22c55e;margin-bottom:0.25rem;">Chat</div>
            <div style="font-size:0.8rem;color:#94a3b8;">Tool-calling LLM with RAG &amp; classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:1.25rem;">
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;
                        color:#64748b;margin-bottom:0.5rem;">Feature</div>
            <div style="font-size:1.5rem;font-weight:700;color:#3b82f6;margin-bottom:0.25rem;">RAG</div>
            <div style="font-size:0.8rem;color:#94a3b8;">Hybrid retrieval + reranking from MONAI docs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;padding:1.25rem;">
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;
                        color:#64748b;margin-bottom:0.5rem;">Feature</div>
            <div style="font-size:1.5rem;font-weight:700;color:#a855f7;margin-bottom:0.25rem;">Memory</div>
            <div style="font-size:0.8rem;color:#94a3b8;">Episodic long-term memory via pgvector</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="margin-top:2rem;padding:1.25rem;background:#0a1628;
                border:1px solid #1e293b;border-radius:14px;">
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:0.06em;">Quick start</div>
        <p style="color:#94a3b8;font-size:0.875rem;margin:0;">
            Head to <strong style="color:#f1f5f9;">Chat</strong> to triage an issue,
            <strong style="color:#f1f5f9;">Widget Admin</strong> to embed the copilot in your app,
            or <strong style="color:#f1f5f9;">Memory Inspector</strong> to review stored memories.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
