"""Chat page — tool-calling LLM backed by the backend /chat endpoint.

Source filter and confidence threshold flow through to the backend's
rag_search tool so the LLM only sees chunks from the user's chosen
sources. Citations are surfaced as chips under each assistant message.
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from lib.auth import get_api_client, require_auth
from lib.styles import inject_styles, page_header, render_citations

st.set_page_config(page_title="Chat — Copilot", page_icon="💬", layout="wide")
inject_styles()

if not require_auth():
    st.stop()

page_header(
    "Chat",
    "Triage issues, search docs, and extract entities — all in one thread",
    icon_svg=(
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
        'stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
    ),
)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "source_filter" not in st.session_state:
    st.session_state.source_filter = "Both"
if "min_confidence" not in st.session_state:
    st.session_state.min_confidence = 0.30

# ── Top control row: conv id · source filter · settings · new conv ────────
col_info, col_filter, col_settings, col_btn = st.columns([3, 3, 1, 2])
with col_info:
    conv_id = st.session_state.conversation_id
    st.markdown(
        f'<div style="font-size:0.75rem;color:#475569;'
        f"font-family:'JetBrains Mono',monospace;padding-top:0.4rem;\">"
        f"conv: {conv_id[:18]}…</div>",
        unsafe_allow_html=True,
    )
with col_filter:
    st.session_state.source_filter = st.radio(
        "Source filter",
        options=["Both", "Wiki only", "Issues only"],
        index=["Both", "Wiki only", "Issues only"].index(st.session_state.source_filter),
        horizontal=True,
        label_visibility="collapsed",
        help="Restrict RAG retrieval to a subset of the corpus",
    )
with col_settings:
    with st.popover("⚙", use_container_width=True):
        st.markdown("**Citation threshold**")
        st.session_state.min_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.min_confidence,
            step=0.05,
            label_visibility="collapsed",
            help="Drop citations whose normalized score is below this.",
        )
        st.caption(
            f"Currently dropping citations below {st.session_state.min_confidence:.2f}"
        )
with col_btn:
    if st.button("New conversation", key="new_conv", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────


def _source_types_for_api() -> list[str] | None:
    """Translate UI radio label → backend source_types list (None = all)."""
    label = st.session_state.source_filter
    if label == "Wiki only":
        return ["wiki"]
    if label == "Issues only":
        return ["issue"]
    return None  # Both


def _render_tools_expander(tool_calls: list[dict[str, Any]]) -> None:
    if not tool_calls:
        return
    summary = ", ".join(str(tc.get("tool", "?")) for tc in tool_calls)
    with st.expander(f"Tools used: {summary}", expanded=False):
        for tc in tool_calls:
            st.markdown(
                f"**`{tc.get('tool','?')}`**  \n```\n{tc.get('result','')}\n```"
            )


# ── Message history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))
        _render_tools_expander(msg.get("tool_calls", []))

# ── Input ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Paste an issue title/body or ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            citations: list[dict[str, Any]] = []
            tool_calls_made: list[dict[str, Any]] = []
            try:
                client = get_api_client()
                data = client.chat(
                    message=prompt,
                    conversation_id=st.session_state.conversation_id,
                    rag_source_types=_source_types_for_api(),
                    rag_min_confidence=st.session_state.min_confidence,
                )
                reply: str = data.get("response", "(no response)")
                conversation_id = data.get("conversation_id")
                if conversation_id:
                    st.session_state.conversation_id = conversation_id
                tool_calls_made = data.get("tool_calls_made") or data.get("tools_used") or []
                # Normalize "tools_used" (list[str]) into list[dict] for the expander
                if tool_calls_made and isinstance(tool_calls_made[0], str):
                    tool_calls_made = [{"tool": t} for t in tool_calls_made]
                citations = list(data.get("citations") or [])
            except Exception as exc:  # noqa: BLE001
                reply = f"Error contacting backend: {exc}"

        st.markdown(reply)
        render_citations(citations)
        _render_tools_expander(tool_calls_made)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "tool_calls": tool_calls_made,
            "citations": citations,
        }
    )
