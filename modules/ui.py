"""Shared UI styles — ops-tool aesthetic, applied globally."""

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* ── monospace utility ────────────────────────────────────────────── */
.mono, .metric-label, .status-line {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
}

/* ── status colours ───────────────────────────────────────────────── */
.status-up       { color: #22C55E; }
.status-warn     { color: #F59E0B; }
.status-critical { color: #EF4444; }
.status-ok       { color: #94A3B8; }
.status-down     { color: #EF4444; }
.status-error    { color: #EF4444; }

/* ── alert feed rows ──────────────────────────────────────────────── */
.alert-feed {
    border-left: 3px solid #334155;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.45rem;
    background: #1E293B;
    border-radius: 0 2px 2px 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.alert-feed.critical { border-left-color: #EF4444; background: rgba(239,68,68,0.07); }
.alert-feed.warning  { border-left-color: #F59E0B; background: rgba(245,158,11,0.07); }
.alert-feed.ok       { border-left-color: #22C55E; background: rgba(34,197,94,0.05); }

/* ── page header ──────────────────────────────────────────────────── */
.page-header {
    border-bottom: 1px solid #334155;
    padding-bottom: 0.6rem;
    margin-bottom: 1.1rem;
}
.page-header h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #E2E8F0;
    margin: 0;
}

/* ── constraint / info box ────────────────────────────────────────── */
.constraint-box {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 1rem 1.1rem;
    border-radius: 3px;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── RTL Arabic text ──────────────────────────────────────────────── */
.rtl-block {
    direction: rtl;
    text-align: right;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #64748B;
    font-size: 0.82rem;
}

/* ── device / data rows ───────────────────────────────────────────── */
.device-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1E293B;
    line-height: 1.6;
}

/* ── section divider ──────────────────────────────────────────────── */
.section-rule {
    border: none;
    border-top: 1px solid #334155;
    margin: 1rem 0;
}

/* ── home page feature cards ──────────────────────────────────────── */
.feature-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 3px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.feature-card:hover {
    border-color: #3B82F6;
}
.feature-card .card-icon {
    font-size: 1.1rem;
    margin-bottom: 0.4rem;
    display: block;
}
.feature-card .card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    color: #E2E8F0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}
.feature-card .card-desc {
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.5;
}

/* ── status badge pill ────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-up       { background: rgba(34,197,94,0.15);  color: #22C55E; }
.badge-down     { background: rgba(239,68,68,0.15);  color: #EF4444; }
.badge-warn     { background: rgba(245,158,11,0.15); color: #F59E0B; }
.badge-neutral  { background: rgba(148,163,184,0.1); color: #94A3B8; }

/* ── tool trace steps ─────────────────────────────────────────────── */
.tool-step {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 2px;
    padding: 0.4rem 0.7rem;
    margin-bottom: 0.35rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
}
.tool-step .tool-name { color: #38BDF8; font-weight: 600; }
.tool-step .tool-iter { color: #64748B; }

/* ── log block ────────────────────────────────────────────────────── */
.log-block {
    background: #0F172A;
    border: 1px solid #334155;
    border-radius: 3px;
    padding: 0.75rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #94A3B8;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.6;
}

/* ── tighten Streamlit's own metric delta ─────────────────────────── */
[data-testid="stMetricDelta"] svg { display: none; }
</style>
"""


def inject_global_css() -> None:
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def stretch_kwargs() -> dict:
    """Version-safe width kwarg: returns width='stretch' on Streamlit >=1.42
    or use_container_width=True on older versions."""
    from importlib.metadata import version as _ver
    try:
        v = tuple(int(p) for p in _ver("streamlit").split(".")[:2])
        if v >= (1, 42):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}
