"""Memory inspector — view and delete long-term episodic memories."""

from __future__ import annotations

import streamlit as st

from lib.auth import get_api_client, require_auth
from lib.styles import inject_styles, page_header

st.set_page_config(page_title="Memory Inspector — Copilot", page_icon="🧠", layout="wide")
inject_styles()

if not require_auth():
    st.stop()

page_header(
    "Memory Inspector",
    "Long-term episodic memories stored across conversations",
    icon_svg='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>',
)

client = get_api_client()

col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button("Refresh", key="refresh_mem"):
        st.rerun()

try:
    memories = client.list_memories()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load memories: {exc}")
    memories = []

if not memories:
    st.markdown(
        """
        <div style="background:#1e293b;border:1px dashed #334155;border-radius:14px;
                    padding:3rem 2rem;text-align:center;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#475569"
                 stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
                 style="margin:0 auto 1rem;display:block">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8
                         a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
            </svg>
            <div style="color:#64748b;font-size:0.875rem;">No memories stored yet.</div>
            <div style="color:#475569;font-size:0.8rem;margin-top:0.5rem;">
                Ask the copilot to remember something in a chat session.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div style="font-size:0.75rem;color:#64748b;margin-bottom:1rem;">'
        f"{len(memories)} memor{'y' if len(memories)==1 else 'ies'} stored</div>",
        unsafe_allow_html=True,
    )

    for mem in memories:
        mid = str(mem.get("id", "?"))
        created = str(mem.get("created_at", ""))[:10]
        content = str(mem.get("content", ""))

        col_content, col_del = st.columns([5, 1])
        with col_content:
            st.markdown(
                f"""
                <div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
                            padding:1rem 1.25rem;margin-bottom:0.5rem;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.625rem;">
                        <code style="font-size:10px;color:#64748b;background:#0a1628;
                                     border:1px solid #1e293b;border-radius:5px;
                                     padding:2px 8px;">{mid[:12]}…</code>
                        <span style="font-size:11px;color:#475569;">{created}</span>
                    </div>
                    <div style="font-size:0.875rem;color:#e2e8f0;line-height:1.55;">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_del:
            st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)
            if st.button("Delete", key=f"del_mem_{mid}", type="secondary"):
                try:
                    client.delete_memory(mid)
                    st.success("Memory deleted")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Delete failed: {exc}")
