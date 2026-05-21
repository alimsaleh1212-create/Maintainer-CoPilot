"""Chat page — tool-calling LLM backed by the backend /chat endpoint."""

from __future__ import annotations

import uuid

import streamlit as st

from lib.auth import get_api_client, require_auth
from lib.styles import inject_styles, page_header

st.set_page_config(page_title="Chat — Copilot", page_icon="💬", layout="wide")
inject_styles()

if not require_auth():
    st.stop()

page_header(
    "Chat",
    "Triage issues, search docs, and extract entities — all in one thread",
    icon_svg='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>',
)

# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# ── Controls ───────────────────────────────────────────────────────────────
col_info, col_btn = st.columns([4, 1])
with col_info:
    conv_id = st.session_state.conversation_id
    st.markdown(
        f'<div style="font-size:0.75rem;color:#475569;font-family:\'JetBrains Mono\',monospace;">'
        f"conv: {conv_id[:18]}…</div>",
        unsafe_allow_html=True,
    )
with col_btn:
    if st.button("New conversation", key="new_conv"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)

# ── Message history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show tool calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            with st.expander(f"Tools used: {', '.join(tc.get('tool','?') for tc in tool_calls)}", expanded=False):
                for tc in tool_calls:
                    st.markdown(
                        f"**`{tc.get('tool','?')}`**  \n```json\n{tc.get('result','')}\n```"
                    )

# ── Input ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Paste an issue title/body or ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                client = get_api_client()
                data = client.chat(
                    message=prompt,
                    conversation_id=st.session_state.conversation_id,
                )
                reply: str = data.get("response", "(no response)")
                conversation_id: str | None = data.get("conversation_id")
                if conversation_id:
                    st.session_state.conversation_id = conversation_id

                tool_calls_made: list[dict[str, object]] = data.get("tool_calls_made", [])
            except Exception as exc:  # noqa: BLE001
                reply = f"Error contacting backend: {exc}"
                tool_calls_made = []

        st.markdown(reply)
        if tool_calls_made:
            with st.expander(f"Tools used: {', '.join(str(tc.get('tool','?')) for tc in tool_calls_made)}", expanded=False):
                for tc in tool_calls_made:
                    st.markdown(
                        f"**`{tc.get('tool','?')}`**  \n```\n{tc.get('result','')}\n```"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "tool_calls": tool_calls_made}
    )
