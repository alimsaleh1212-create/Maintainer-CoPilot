"""Global CSS injection for polished dark-mode developer UI."""

import streamlit as st

_CSS = """
<style>
/* ── Import font ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root overrides ───────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Main background */
.stApp {
    background: #0f172a !important;
}

/* Hide default header / footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Sidebar nav links */
.stPageLink a {
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}
.stPageLink a:hover {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.25rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 2px 12px #22c55e30 !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 20px #22c55e50 !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

/* ── Text inputs & text areas ─────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 3px #22c55e20 !important;
}

/* ── Chat messages ────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #1e293b !important;
    border: 1px solid #2d3f55 !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
    padding: 0.875rem 1rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #162035 !important;
    border-color: #22c55e30 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #1e293b !important;
    border-top: 1px solid #1e293b !important;
}
[data-testid="stChatInputTextArea"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #22c55e !important;
    font-weight: 700 !important;
}

/* ── Expanders ────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
}

/* ── Info / success / warning / error boxes ───────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 3px !important;
}

/* ── Code blocks ──────────────────────────────────────────── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    background: #0a1628 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #22c55e !important;
}

/* ── Dividers ─────────────────────────────────────────────── */
hr {
    border-color: #1e293b !important;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* ── Spinner ──────────────────────────────────────────────── */
.stSpinner > div {
    border-color: #22c55e transparent transparent transparent !important;
}

/* ── Forms ────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 14px !important;
    padding: 1.5rem !important;
}

/* ── Multiselect ──────────────────────────────────────────── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #22c55e20 !important;
    border: 1px solid #22c55e50 !important;
    border-radius: 6px !important;
    color: #22c55e !important;
}
</style>
"""


def inject_styles() -> None:
    """Inject global CSS into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon_svg: str = "") -> None:
    """Render a consistent page header with optional icon and subtitle.

    Important: the rendered HTML is emitted as a single un-indented string —
    Streamlit's markdown parser treats any line indented 4+ spaces as a code
    block, even with ``unsafe_allow_html=True``. Keep this on one line.
    """
    icon_html = (
        '<div style="width:42px;height:42px;border-radius:12px;'
        "background:linear-gradient(135deg,#22c55e,#16a34a);"
        "display:flex;align-items:center;justify-content:center;"
        f'flex-shrink:0;">{icon_svg}</div>'
        if icon_svg
        else ""
    )
    subtitle_html = (
        f"<p style=\"margin:4px 0 0;font-size:0.875rem;color:#64748b;\">{subtitle}</p>"
        if subtitle
        else ""
    )
    html = (
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:1.5rem;">'
        f"{icon_html}"
        "<div>"
        '<h1 style="margin:0;font-size:1.5rem;font-weight:700;'
        'color:#f1f5f9;letter-spacing:-0.02em;line-height:1.2;">'
        f"{title}</h1>"
        f"{subtitle_html}"
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def card(content_html: str, padding: str = "1.25rem") -> None:
    """Render content inside a styled dark card."""
    st.markdown(
        f"""
        <div style="
            background:#1e293b;border:1px solid #334155;
            border-radius:14px;padding:{padding};
            margin-bottom:0.75rem;
        ">{content_html}</div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = "#22c55e") -> str:
    """Return HTML for an inline badge."""
    return (
        f'<span style="background:{color}20;color:{color};border:1px solid {color}50;'
        f'border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;">{text}</span>'
    )
