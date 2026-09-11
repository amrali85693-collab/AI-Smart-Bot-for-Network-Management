"""App-wide settings, configurable thresholds, and internationalization."""

from __future__ import annotations

from typing import Any, Dict
import streamlit as st

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cpu_warning": 85.0,
    "packet_loss_warning": 1.5,
    "latency_warning_ms": 100.0,
}

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        "language_section": "Language / اللغة",
        "english": "English",
        "arabic": "العربية",
        "current_language": "Current",
        "data_source": "Data Source",
        "live_mode_caption": "Real dataset mode trains models on a 30-day real-world CSV. Live mode pings hosts and reads this machine's metrics. Simulated uses generated demo data.",
        "chatbot_label": "Chatbot",
        "dashboard_label": "Dashboard",
        "monitor_label": "Network Monitor",
        "how_it_works": "How this works / limitations",
        "app_title": "AI Smart Bot for Network Management",
        "app_caption": "Network operations assistant combining live monitoring, threshold alerts, and an LLM-powered diagnostics chatbot with a local knowledge base.",
        "chatbot_title": "Network Chatbot",
        "chatbot_caption": "RAG-backed Q&A for networking topics and pasted diagnostics.",
        "suggested_questions": "Suggested questions",
        "clear_chat": "Clear chat",
        "ask_placeholder": "Ask a networking question or paste diagnostics...",
        "dashboard_title": "Network Dashboard",
        "dashboard_caption": "Simulated device telemetry for demo — alerts feed the diagnostics bridge.",
        "alert_thresholds": "Alert Thresholds",
        "cpu_warning": "CPU Warning (%)",
        "packet_loss_warning": "Packet Loss Warning (%)",
        "latency_warning": "Latency Warning (ms)",
        "adjust_thresholds": "Adjust thresholds to customize alert sensitivity.",
        "alert_feed": "Alert feed",
        "device_status": "Device status",
        "metrics": "Metrics",
        "refresh_telemetry": "Refresh simulated telemetry",
        "telemetry_note": "Telemetry is simulated for demo purposes. Deploy on-prem or add an agent for real devices.",
        "monitor_title": "Network Monitor",
        "monitor_caption": "Real ICMP checks and HTTP health checks against public hosts (Streamlit Cloud cannot reach private LAN IPs).",
        "host_or_ip": "Host or IP",
        "run_ping": "Run ping",
        "quick_targets": "Quick targets",
        "url": "URL",
        "check_http": "Check HTTP",
        "quick_urls": "Quick URLs",
        "clear_history": "Clear history",
        "alert_history": "Alert History",
        "evaluation_metrics": "Evaluation Metrics (for Graduation Report)",
        "chatbot_performance": "Chatbot Performance",
        "alert_performance": "Alert Explanation Performance",
    },
    "ar": {
        "language_section": "اللغة / Language",
        "english": "English",
        "arabic": "العربية",
        "current_language": "الحالي",
        "data_source": "مصدر البيانات",
        "live_mode_caption": "وضع البيانات الحقيقية يدرّب النماذج على ملف CSV حقيقي لمدة 30 يوماً. الوضع المباشر يرسل ping ويقرأ مقاييس هذا الجهاز. الوضع المحاكى يستخدم بيانات توضيحية.",
        "chatbot_label": "الدردشة",
        "dashboard_label": "لوحة البيانات",
        "monitor_label": "مراقبة الشبكة",
        "how_it_works": "كيف يعمل هذا / القيود",
        "app_title": "روبوت ذكي لإدارة الشبكة",
        "app_caption": "مساعد عمليات الشبكة يجمع بين المراقبة الحية والتنبيهات والتشخيص الذكي المدعوم بقاعدة معرفة محلية.",
        "chatbot_title": "روبوت الشبكات",
        "chatbot_caption": "أسئلة وأجوبة مدعومة بالبحث عن مواضيع الشبكات والتشخيصات الملصقة.",
        "suggested_questions": "أسئلة مقترحة",
        "clear_chat": "مسح المحادثة",
        "ask_placeholder": "اسأل عن الشبكات أو الصق مخرجات التشخيص...",
        "dashboard_title": "لوحة التحكم بالشبكة",
        "dashboard_caption": "بيانات محاكاة للأجهزة — التنبيهات تغذي جسر التشخيص.",
        "alert_thresholds": "عتبات التنبيه",
        "cpu_warning": "تحذير وحدة المعالجة المركزية (%)",
        "packet_loss_warning": "تحذير فقدان الحزم (%)",
        "latency_warning": "تحذير زمن الاستجابة (مللي ثانية)",
        "adjust_thresholds": "اضبط العتبات لتخصيص حساسية التنبيهات.",
        "alert_feed": "موجز التنبيهات",
        "device_status": "حالة الجهاز",
        "metrics": "المقاييس",
        "refresh_telemetry": "تحديث البيانات المحاكاة",
        "telemetry_note": "البيانات محاكاة للتوضيح. انشر محلياً أو أضف عامل للأجهزة الحقيقية.",
        "monitor_title": "مراقبة الشبكة",
        "monitor_caption": "فحوصات ICMP و HTTP حقيقية للمضيفين العامين (السحابة لا تصل للشبكات الخاصة).",
        "host_or_ip": "المضيف أو IP",
        "run_ping": "تشغيل ping",
        "quick_targets": "أهداف سريعة",
        "url": "الرابط",
        "check_http": "فحص HTTP",
        "quick_urls": "روابط سريعة",
        "clear_history": "مسح السجل",
        "alert_history": "سجل التنبيهات",
        "evaluation_metrics": "مقاييس التقييم (لتقرير التخرج)",
        "chatbot_performance": "أداء الروبوت",
        "alert_performance": "أداء شرح التنبيهات",
    },
}


def init_session_settings() -> None:
    """Initialize default settings in the Streamlit session state if not already set."""
    if "thresholds" not in st.session_state:
        st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()
    if "llm_provider" not in st.session_state:
        try:
            st.session_state.llm_provider = st.secrets.get("LLM_PROVIDER", "groq")
        except Exception:
            st.session_state.llm_provider = "groq"
    if "language" not in st.session_state:
        st.session_state.language = "en"


def get_thresholds() -> Dict[str, float]:
    """Retrieve the current alert thresholds."""
    init_session_settings()
    thresholds: Dict[str, float] = st.session_state.thresholds
    return thresholds


def get_llm_provider() -> str:
    """Retrieve the active LLM provider choice (e.g. 'groq' or 'gemini')."""
    init_session_settings()
    provider: str = st.session_state.llm_provider
    return provider


def get_language() -> str:
    """Retrieve the active language setting."""
    init_session_settings()
    lang: str = st.session_state.language
    return lang


def set_language(lang: str) -> None:
    """Update the active language setting."""
    st.session_state.language = lang


def get_text(key: str) -> str:
    """Get translated text for the current language, falling back to English."""
    lang = get_language()
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def llm_configured() -> bool:
    """Check if the active LLM provider API key is present in the configurations."""
    try:
        provider = get_llm_provider()
        if provider == "gemini":
            return bool(st.secrets.get("GEMINI_API_KEY"))
        return bool(st.secrets.get("GROQ_API_KEY"))
    except Exception:
        return False
