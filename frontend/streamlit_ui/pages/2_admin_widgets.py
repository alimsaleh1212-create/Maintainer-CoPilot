"""Admin widget management page."""

from __future__ import annotations

import streamlit as st

from lib.auth import get_api_client, is_admin, require_auth
from lib.styles import inject_styles, page_header

st.set_page_config(page_title="Widget Admin — Copilot", page_icon="⚙️", layout="wide")
inject_styles()

if not require_auth():
    st.stop()
if not is_admin():
    st.error("Admin access required.")
    st.stop()

page_header(
    "Widget Admin",
    "Create and manage embeddable chat widgets",
    icon_svg='<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="9"/><line x1="9" y1="12" x2="15" y2="12"/><line x1="9" y1="15" x2="12" y2="15"/></svg>',
)

client = get_api_client()

# ── Existing widgets ───────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;'
    'color:#64748b;margin-bottom:0.75rem;">Your widgets</div>',
    unsafe_allow_html=True,
)

try:
    widgets = client.list_widgets()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load widgets: {exc}")
    widgets = []

if not widgets:
    st.markdown(
        '<div style="background:#1e293b;border:1px dashed #334155;border-radius:14px;'
        'padding:2rem;text-align:center;color:#64748b;font-size:0.875rem;">'
        "No widgets yet. Create one below.</div>",
        unsafe_allow_html=True,
    )
else:
    for w in widgets:
        internal_id = str(w.get("id", ""))
        public_wid = str(w.get("public_widget_id", ""))
        enabled = bool(w.get("enabled", True))
        origins = w.get("allowed_origins", [])
        tools = w.get("enabled_tools", [])

        status_color = "#22c55e" if enabled else "#ef4444"
        status_label = "Enabled" if enabled else "Disabled"

        with st.expander(f"Widget  ·  {public_wid}  ·  {status_label}", expanded=False):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<div style="margin-bottom:0.75rem;">'
                    f'<span style="background:{status_color}20;color:{status_color};'
                    f'border:1px solid {status_color}50;border-radius:6px;'
                    f'padding:2px 10px;font-size:11px;font-weight:600;">{status_label}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("**Embed snippet:**")
                st.code(
                    f'<script src="http://localhost:8000/widget.js"\n'
                    f'        data-widget-id="{public_wid}"\n'
                    f'        data-api-host="http://localhost:8000" async></script>',
                    language="html",
                )
                if origins:
                    st.markdown(f"**Allowed origins:** `{'` · `'.join(origins)}`")
                if tools:
                    st.markdown(f"**Enabled tools:** `{'` · `'.join(tools)}`")
            with col_b:
                if st.button("Delete widget", key=f"del_{internal_id}", type="secondary"):
                    try:
                        client.delete_widget(internal_id)
                        st.success("Deleted")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Delete failed: {exc}")

st.markdown("<div style='height:1.5rem'/>", unsafe_allow_html=True)

# ── Create widget ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.06em;'
    'color:#64748b;margin-bottom:0.75rem;">Create a new widget</div>',
    unsafe_allow_html=True,
)

with st.form("create_widget", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        origins_raw = st.text_input(
            "Allowed origins",
            placeholder="http://localhost:8090, https://myapp.com",
            help="Comma-separated list of origins that can embed this widget",
        )
        greeting = st.text_input(
            "Greeting message",
            value="Hi! I'm the Maintainer's Copilot. Paste an issue and I'll help triage it.",
        )
    with col2:
        enabled_tools = st.multiselect(
            "Enabled tools",
            ["classify", "ner", "summarize", "rag_search", "write_memory"],
            default=["classify", "rag_search", "summarize"],
            help="Tools the LLM can call during a chat session",
        )
        primary_color = st.color_picker("Primary color", value="#22c55e")

    submitted = st.form_submit_button("Create widget", type="primary")

if submitted:
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    try:
        result = client.create_widget(
            {
                "allowed_origins": origins,
                "greeting": greeting,
                "enabled_tools": enabled_tools,
                "theme": {"primaryColor": primary_color},
            }
        )
        new_id = result.get("id", "?")
        st.success(f"Widget created: `{new_id}`")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Create failed: {exc}")
