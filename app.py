import streamlit as st

from modules.settings import get_language, get_text, init_session_settings, set_language
from modules.ui import inject_global_css
from modules.data_sources import get_data_source, render_data_source_sidebar

st.set_page_config(
    page_title="Network AI Bot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_settings()
inject_global_css()

st.markdown(
    f'<div class="page-header"><h1>{get_text("app_title")}</h1></div>',
    unsafe_allow_html=True,
)

st.markdown(get_text("app_caption"))

with st.sidebar:
    st.markdown(f"**{get_text('language_section')}**")
    col_en, col_ar = st.columns(2)
    with col_en:
        if st.button(get_text("english"), key="lang_en"):
            set_language("en")
            st.rerun()
    with col_ar:
        if st.button(get_text("arabic"), key="lang_ar"):
            set_language("ar")
            st.rerun()
    
    current_lang = get_language()
    st.caption(f"{get_text('current_language')}: {get_text('english') if current_lang == 'en' else get_text('arabic')}")
    
    st.divider()
    st.markdown(f"**{get_text('data_source')}**")
    render_data_source_sidebar()
    st.caption(get_text("live_mode_caption"))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"**{get_text('chatbot_label')}**")
    st.caption("Agent-mode chatbot with tool-calling: checks devices, runs pings, reads alerts, scores anomalies, and answers with real data. RAG-only mode also available.")
    st.page_link("pages/1_Chatbot.py", label=f"Open {get_text('chatbot_label')} →")

with col2:
    st.markdown(f"**{get_text('dashboard_label')}**")
    st.caption("Device metrics, charts, alerts, and AI explanations for incidents.")
    st.page_link("pages/2_Dashboard.py", label=f"Open {get_text('dashboard_label')} →")

with col3:
    st.markdown(f"**{get_text('monitor_label')}**")
    st.caption("Run real ICMP checks against public hosts and review session history.")
    st.page_link("pages/3_Network_Monitor.py", label=f"Open {get_text('monitor_label')} →")

with col4:
    st.markdown("**AI Ops**")
    st.caption("Automatic fault diagnosis, Cisco/MikroTik command recommendations, log-based root-cause analysis, predictive ML warnings, and intelligent reporting.")
    st.page_link("pages/4_AI_Ops.py", label="Open AI Ops →")

st.markdown("### Core capabilities")
st.markdown(
    "- Automatic AI-powered fault diagnosis\n"
    "- AI-generated troubleshooting recommendations with ready-to-use Cisco and MikroTik commands\n"
    "- Log file analysis to identify the root cause of network issues\n"
    "- Predictive fault detection using machine learning to prevent failures before they occur\n"
    "- Automatic generation of intelligent reports and interactive visual dashboards",
    unsafe_allow_html=False,
)

with st.expander("How this works / limitations", expanded=True):
    st.markdown(
        """
        **Architecture**

        - **Chatbot (Agent mode):** Uses Groq function-calling to invoke real tools before answering — searches the knowledge base, checks device status, runs live ICMP pings, reads current alerts, and queries the ML anomaly detector. Tool trace is shown in the UI (collapsed by default).
        - **Chatbot (RAG-only mode):** TF-IDF retrieval over a local JSON knowledge base + hosted LLM (Groq/Gemini).
        - **Dashboard:** Simulated device inventory for demo metrics; threshold engine raises alerts; ML anomaly detection via Isolation Forest.
        - **Diagnostics bridge:** Dashboard alerts can be explained by the LLM with concrete next steps.

        **Honest constraints**

        1. **Streamlit Community Cloud is not on your LAN.** It can ping public hosts but not private
           `192.168.x.x` / `10.x.x.x` devices unless you deploy on-prem, use VPN, or add a local collector agent.
        2. **Hosted LLM APIs are rate-limited** on free tiers — provider is swappable via secrets.
        3. **Session state resets** on restart; SQLite stores optional chat/alert history locally.
        4. **No local LLM inference** — keeps the app lightweight for free-tier hosting.
        5. **Agent tool calls cost latency** — each tool invocation is one Groq API round-trip; expect 2-5s per tool.

        **SNMP stretch goal:** `snmp_poll()` is stubbed in `modules/network_monitor.py` for lab integration.
        """
    )

st.markdown(
    '<p class="rtl-block mono">مساعد شبكات ذكي — مراقبة، تنبيهات، وتشخيص بلغة طبيعية</p>',
    unsafe_allow_html=True,
)
