"""Chat page — tool-calling LLM backed by the backend /chat endpoint.

Conversation history is kept in session_state so users can switch between
past conversations using the sidebar list. Source filter applies immediately
on any new message — no new conversation needed.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from lib.auth import current_user, get_api_client, require_auth
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

# ── CSS additions for conversation history sidebar items ───────────────────
st.markdown(
    """
    <style>
    .conv-item {
        display:flex;align-items:flex-start;gap:10px;
        background:#1e293b;border:1px solid #334155;border-radius:10px;
        padding:0.75rem 1rem;margin-bottom:0.5rem;cursor:pointer;
        transition:border-color 0.15s,background 0.15s;
    }
    .conv-item:hover { border-color:#22c55e60;background:#1e293b; }
    .conv-item.active { border-color:#22c55e;background:#22c55e12; }
    .conv-title { font-size:0.8rem;color:#e2e8f0;font-weight:500;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                  max-width:160px; }
    .conv-time  { font-size:0.68rem;color:#475569;margin-top:2px; }
    .conv-count { font-size:0.65rem;color:#64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Per-user session state bootstrap ───────────────────────────────────────
# Conversations are scoped by user_id so logging out and back in as a
# different user shows that user's conversations only — never the previous
# session's. The server enforces ownership too (verify_conversation_owner)
# so the client-side scoping is defence-in-depth, not the only barrier.
_user = current_user() or {}
_user_id = str(_user.get("id", "anon"))

if "conversations_by_user" not in st.session_state:
    st.session_state.conversations_by_user = {}
if "active_conv_by_user" not in st.session_state:
    st.session_state.active_conv_by_user = {}
if "convs_loaded_for_user" not in st.session_state:
    st.session_state.convs_loaded_for_user = ""
if "source_filter" not in st.session_state:
    st.session_state.source_filter = "Both"
if "min_confidence" not in st.session_state:
    st.session_state.min_confidence = 0.30


def _user_conversations() -> dict[str, dict[str, Any]]:
    bucket = st.session_state.conversations_by_user.setdefault(_user_id, {})
    return bucket  # type: ignore[no-any-return]


def _hydrate_conversations_from_server() -> None:
    """Pull this user's conversations from /chat/conversations once per login.

    Server is the source of truth for ownership and history; the session_state
    cache lets the sidebar render without an extra API call per rerun.
    """
    if st.session_state.convs_loaded_for_user == _user_id:
        return
    try:
        client = get_api_client()
        rows = client.list_conversations()
    except Exception:  # noqa: BLE001
        rows = []
    bucket = _user_conversations()
    for row in rows:
        cid = row.get("conversation_id")
        if not cid or cid in bucket:
            continue
        # Lazy-load messages: leave empty until the user clicks the conv.
        ts = row.get("updated_at") or row.get("created_at") or ""
        try:
            ts_h = datetime.fromtimestamp(float(ts)).strftime("%H:%M") if ts else ""
        except (TypeError, ValueError):
            ts_h = ""
        bucket[cid] = {
            "title": row.get("title", "Conversation"),
            "ts": ts_h,
            "messages": [],
            "_loaded": False,
        }
    st.session_state.convs_loaded_for_user = _user_id


_hydrate_conversations_from_server()


def _ensure_active_conversation() -> str:
    """Return the active conversation ID, creating a new one if needed."""
    bucket = _user_conversations()
    cid = st.session_state.active_conv_by_user.get(_user_id)
    if cid is None or cid not in bucket:
        cid = str(uuid.uuid4())
        st.session_state.active_conv_by_user[_user_id] = cid
        bucket[cid] = {
            "title": "New conversation",
            "ts": datetime.now().strftime("%H:%M"),
            "messages": [],
            "_loaded": True,
        }
    return cid


def _switch_conversation(cid: str) -> None:
    st.session_state.active_conv_by_user[_user_id] = cid
    bucket = _user_conversations()
    meta = bucket.get(cid)
    # Lazy-load messages from the server the first time the user opens this conv.
    if meta is not None and not meta.get("_loaded"):
        try:
            data = get_api_client().get_conversation(cid)
            meta["title"] = data.get("title", meta["title"])
            meta["messages"] = [
                {
                    "role": m.get("role", "user"),
                    "content": m.get("content", ""),
                    "tool_calls": [],
                    "citations": [],
                }
                for m in data.get("messages", [])
            ]
            meta["_loaded"] = True
        except Exception:  # noqa: BLE001
            meta["_loaded"] = True  # avoid retry loops; user can refresh


def _new_conversation() -> None:
    cid = str(uuid.uuid4())
    st.session_state.active_conv_by_user[_user_id] = cid
    _user_conversations()[cid] = {
        "title": "New conversation",
        "ts": datetime.now().strftime("%H:%M"),
        "messages": [],
        "_loaded": True,
    }


def _source_types_for_api() -> list[str] | None:
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


# ── Sidebar: conversation history ──────────────────────────────────────────
with st.sidebar:
    # Section header
    st.markdown(
        '<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;'
        'color:#475569;padding:0.25rem 0 0.75rem;border-bottom:1px solid #1e293b;'
        'margin-bottom:0.75rem;">Conversations</div>',
        unsafe_allow_html=True,
    )

    # New conversation button
    if st.button(
        "New conversation",
        key="sidebar_new_conv",
        use_container_width=True,
    ):
        _new_conversation()
        st.rerun()

    st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)

    # Conversation list — most recent first (by insertion order reversed)
    all_convs = list(_user_conversations().items())
    active_id = st.session_state.active_conv_by_user.get(_user_id) or ""

    for cid, meta in reversed(all_convs):
        msgs = meta.get("messages", [])
        n_msgs = len(msgs)
        is_active = cid == active_id
        title = meta.get("title", "Conversation")
        ts = meta.get("ts", "")

        # Truncate title
        display_title = title if len(title) <= 28 else title[:26] + "…"
        active_cls = "active" if is_active else ""

        # Render as a clickable Streamlit button styled to look like a card
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(
                f"""<div class="conv-item {active_cls}">
                    <div>
                        <div class="conv-title">{display_title}</div>
                        <div class="conv-time">{ts} · <span class="conv-count">{n_msgs} msg{'s' if n_msgs != 1 else ''}</span></div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown("<div style='height:0.35rem'/>", unsafe_allow_html=True)
            if not is_active:
                if st.button("›", key=f"switch_{cid}", help=f"Open: {title}"):
                    _switch_conversation(cid)
                    st.rerun()


# ── Ensure we have a valid active conversation ─────────────────────────────
conv_id = _ensure_active_conversation()
conv_meta = _user_conversations()[conv_id]
messages: list[dict[str, Any]] = conv_meta["messages"]

# ── Top control row: conv id · source filter · settings ───────────────────
col_info, col_filter, col_settings = st.columns([3, 4, 1])

with col_info:
    st.markdown(
        f'<div style="font-size:0.75rem;color:#475569;'
        f"font-family:'JetBrains Mono',monospace;padding-top:0.4rem;\">"
        f"conv: {conv_id[:18]}…</div>",
        unsafe_allow_html=True,
    )

with col_filter:
    # Use key= so Streamlit persists the value reliably across re-runs.
    # This fixes the bug where changing the filter mid-conversation had no effect
    # because manual index= recalculation could silently revert the selection.
    st.radio(
        "Source filter",
        options=["Both", "Wiki only", "Issues only"],
        key="source_filter",
        horizontal=True,
        label_visibility="collapsed",
        help="Restrict RAG retrieval — applies to the NEXT message immediately, no new conversation needed",
    )

    # Live indicator when filter is not "Both"
    if st.session_state.source_filter != "Both":
        filter_label = st.session_state.source_filter
        st.markdown(
            f'<div style="font-size:0.68rem;color:#22c55e;margin-top:2px;">'
            f'&#x2713; Filter active: <strong>{filter_label}</strong> — next message will use this</div>',
            unsafe_allow_html=True,
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

st.markdown("<div style='height:0.5rem'/>", unsafe_allow_html=True)

# ── Message history for the active conversation ────────────────────────────
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))
        _render_tools_expander(msg.get("tool_calls", []))

# ── Input ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Paste an issue title/body or ask a question…"):
    messages.append({"role": "user", "content": prompt})

    # Update the conversation title from the first user message
    if conv_meta["title"] == "New conversation":
        conv_meta["title"] = prompt[:50]
        conv_meta["ts"] = datetime.now().strftime("%H:%M")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            citations: list[dict[str, Any]] = []
            tool_calls_made: list[dict[str, Any]] = []
            try:
                client = get_api_client()
                # source_filter is read NOW from session_state — always reflects the
                # current radio selection, even if changed mid-conversation.
                data = client.chat(
                    message=prompt,
                    conversation_id=conv_id,
                    rag_source_types=_source_types_for_api(),
                    rag_min_confidence=st.session_state.min_confidence,
                )
                reply: str = data.get("response", "(no response)")
                returned_cid = data.get("conversation_id")
                if returned_cid and returned_cid != conv_id:
                    # Backend assigned a different ID — sync it
                    _user_conversations()[returned_cid] = conv_meta
                    del _user_conversations()[conv_id]
                    st.session_state.active_conv_by_user[_user_id] = returned_cid
                tool_calls_made = data.get("tool_calls_made") or data.get("tools_used") or []
                if tool_calls_made and isinstance(tool_calls_made[0], str):
                    tool_calls_made = [{"tool": t} for t in tool_calls_made]
                citations = list(data.get("citations") or [])
            except Exception as exc:  # noqa: BLE001
                reply = f"Error contacting backend: {exc}"

        st.markdown(reply)
        render_citations(citations)
        _render_tools_expander(tool_calls_made)

    messages.append(
        {
            "role": "assistant",
            "content": reply,
            "tool_calls": tool_calls_made,
            "citations": citations,
        }
    )
