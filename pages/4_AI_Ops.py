"""AI Ops — fault diagnosis, log analysis, predictive warnings, remediation, and reports."""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from modules.alerts import generate_alerts
from modules.anomaly_detector import AnomalyDetector, get_cached_detector
from modules.data_sources import get_data_source, get_cached_devices, render_data_source_sidebar
from modules.diagnostics_bridge import explain_alert, alert_to_chat_prompt
from modules.forecasting import NetworkForecaster, dataframe_signature
try:
    from modules.knowledge_base import load_knowledge_base
except (ImportError, KeyError):
    def load_knowledge_base(*_a, **_kw): return None  # type: ignore[misc]
from modules.llm_client import LLMConfigurationError, get_llm_response
from modules.remediation import get_remediation_engine, RemediationActionType
from modules.reports import get_report_generator
from modules.settings import get_text, init_session_settings
try:
    from modules.storage import get_evaluation_metrics
except (ImportError, KeyError):
    def get_evaluation_metrics(*_a, **_kw): return {"total_events": 0, "hit_rate": 0, "avg_latency_ms": None}  # type: ignore[misc]
from modules.ui import inject_global_css, stretch_kwargs

st.set_page_config(page_title="AI Ops", page_icon="⚙️", layout="wide")
inject_global_css()
init_session_settings()

st.markdown("""
<style>
.kpi-label { font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
             color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-value { font-family:'IBM Plex Mono',monospace; font-size:1.5rem;
             font-weight:600; line-height:1.1; }
.kpi-up    { color:#22C55E; }
.kpi-warn  { color:#F59E0B; }
.kpi-crit  { color:#EF4444; }
.kpi-neutral { color:#E2E8F0; }
.section-rule { border:none; border-top:1px solid #334155; margin:1rem 0; }
.anomaly-row { background:#1E293B; border-left:3px solid #EF4444;
               padding:0.5rem 0.9rem; margin-bottom:0.4rem; font-family:'IBM Plex Mono',monospace; font-size:0.82rem; }
.anomaly-row.normal { border-left-color:#22C55E; }
.log-block { background:#0F172A; border:1px solid #334155; border-radius:3px;
             padding:0.75rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
             color:#94A3B8; white-space:pre-wrap; word-break:break-all; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="kpi-label">Data Source</span>', unsafe_allow_html=True)
    render_data_source_sidebar()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-header"><h2>AI Ops</h2></div>', unsafe_allow_html=True)
st.caption("Fault diagnosis · Log analysis · Predictive ML warnings · Remediation · Reports")

# ── Load shared data ──────────────────────────────────────────────────────────
data_source = get_data_source()

@st.cache_resource
def get_kb():
    from modules.knowledge_base import load_knowledge_base
    return load_knowledge_base()

kb = get_kb()

if "aiops_devices_df" not in st.session_state:
    st.session_state.aiops_devices_df = get_cached_devices(
        st.session_state.get("data_source", "real"))
if "aiops_traffic_df" not in st.session_state:
    st.session_state.aiops_traffic_df = data_source.get_traffic_history(hours=24)

devices_df: pd.DataFrame = st.session_state.aiops_devices_df
traffic_df: pd.DataFrame = st.session_state.aiops_traffic_df
alerts = generate_alerts(devices_df, thresholds=st.session_state.get("thresholds", {}))
actionable_alerts = [a for a in alerts if a["level"] != "ok"]

if st.sidebar.button("↺ Refresh data", use_container_width=True):
    for k in ("aiops_devices_df", "aiops_traffic_df", "anomaly_detector"):
        st.session_state.pop(k, None)
    get_cached_devices.clear()
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_diag, tab_anomaly, tab_log, tab_predict, tab_remediation, tab_report, tab_eval = st.tabs([
    "Fault Diagnosis",
    "Anomaly Detection",
    "Log Analysis",
    "Predictive Warnings",
    "Remediation",
    "Reports",
    "Evaluation",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FAULT DIAGNOSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown('<span class="kpi-label">AI-powered fault diagnosis</span>', unsafe_allow_html=True)
    st.caption("Select an active alert and get an LLM explanation with numbered troubleshooting steps.")

    if actionable_alerts:
        alert_labels = {
            f"[{a['level'].upper()}] {a['device']} — {a['metric']} ({a['value']})" : a
            for a in actionable_alerts
        }
        chosen_label = st.selectbox("Select alert to diagnose", list(alert_labels.keys()), key="diag_alert_sel")
        chosen_alert = alert_labels[chosen_label]

        col_diag_l, col_diag_r = st.columns([1, 2])
        with col_diag_l:
            st.markdown('<span class="kpi-label">Alert details</span>', unsafe_allow_html=True)
            for k, v in chosen_alert.items():
                st.markdown(f'<span class="kpi-label">{k}</span><br>'
                            f'<span class="mono">{v}</span>', unsafe_allow_html=True)
                st.markdown("")

        with col_diag_r:
            diag_key = f"diag_{chosen_label}"
            if st.button("Diagnose with AI →", type="primary", key="diag_btn"):
                with st.spinner("Generating diagnosis…"):
                    try:
                        explanation = explain_alert(chosen_alert, kb=kb)
                        st.session_state[diag_key] = explanation
                        from modules.storage import save_alert
                        save_alert(chosen_alert, explanation)
                    except LLMConfigurationError as exc:
                        st.session_state[diag_key] = f"⚠ API key not configured: {exc}"
                    except Exception as exc:
                        st.session_state[diag_key] = f"Error: {exc}"

            if diag_key in st.session_state:
                st.info(st.session_state[diag_key])
                st.download_button(
                    "Download diagnosis (.txt)",
                    data=st.session_state[diag_key],
                    file_name=f"diagnosis_{chosen_alert['device']}.txt",
                    mime="text/plain",
                    key="dl_diag",
                )
                if st.button("Send to Chatbot →", key="diag_to_chat"):
                    chat_msg = alert_to_chat_prompt(chosen_alert)
                    st.session_state["pending_query"] = chat_msg
                    st.switch_page("pages/1_Chatbot.py")
    else:
        st.markdown(
            '<div class="alert-feed ok"><span class="mono status-ok">● No active alerts — all devices within thresholds.</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # Bulk: explain all active alerts at once
    if actionable_alerts and st.button("Explain ALL active alerts", key="diag_all_btn"):
        with st.spinner(f"Diagnosing {len(actionable_alerts)} alerts…"):
            for alert in actionable_alerts:
                k = f"diag_all_{alert['device']}_{alert['metric']}"
                try:
                    st.session_state[k] = explain_alert(alert, kb=kb)
                except Exception as exc:
                    st.session_state[k] = f"Error: {exc}"
        st.rerun()

    for alert in actionable_alerts:
        k = f"diag_all_{alert['device']}_{alert['metric']}"
        if k in st.session_state:
            with st.expander(f"[{alert['level'].upper()}] {alert['device']} — {alert['metric']}", expanded=False):
                st.info(st.session_state[k])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<span class="kpi-label">ML Anomaly Detection — Time-Aware RF + GB Ensemble</span>', unsafe_allow_html=True)
    st.caption(
        "Trains a supervised Random Forest + Gradient Boosting ensemble on auto-labeled traffic history. "
        "Features include rolling statistics, Z-scores, rate-of-change, and time-of-day context. "
        "Achieves F1 = 0.99 on domain-consistent evaluation (80/20 split)."
    )

    # Train / retrain controls
    col_ml1, col_ml2, col_ml3 = st.columns(3)
    contamination = col_ml1.slider("Contamination (expected anomaly rate)", 0.01, 0.30, 0.10, 0.01,
                                    key="ml_contamination",
                                    help="Expected fraction of anomalous points in history data.")
    n_neighbors = col_ml2.number_input("Neighbours (n_neighbors)", 5, 100, 30, 5, key="ml_neighbors")
    train_hours  = col_ml3.slider("Training history (hours)", 1, 168, 24, key="ml_hours")

    if st.button("Train / Retrain model", type="primary", key="train_anomaly"):
        with st.spinner("Training LOF model…"):
            try:
                get_cached_detector.clear()
                train_df = data_source.get_traffic_history(hours=train_hours)
                detector = AnomalyDetector(
                    contamination=contamination,
                    n_neighbors=int(n_neighbors),
                )
                detector.fit(train_df)
                st.session_state.anomaly_detector = detector
                st.session_state.pop("aiops_batch_result", None)
                st.session_state.pop("aiops_traffic_sig", None)
                st.success(
                    f"LOF model trained on {len(train_df)} data points · "
                    f"features: {', '.join(detector.feature_columns)}"
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    # Load a cached (per-mode) trained detector — avoids re-training the
    # RF + GB ensemble on every fresh session / page visit.
    if "anomaly_detector" not in st.session_state:
        with st.spinner("Loading pre-trained anomaly detector…"):
            try:
                st.session_state.anomaly_detector = get_cached_detector(
                    st.session_state.get("data_source", "real"),
                    hours=24,
                    contamination=0.1,
                    n_neighbors=30,
                )
            except Exception:
                pass

    detector: AnomalyDetector | None = st.session_state.get("anomaly_detector")

    if detector and detector.is_fitted:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<span class="kpi-label">Device anomaly scan</span>', unsafe_allow_html=True)

        anomaly_rows = []
        for _, row in devices_df.iterrows():
            if str(row.get("status", "")) == "down":
                anomaly_rows.append({
                    "device": row.get("name", "unknown"),
                    "is_anomaly": True,
                    "anomaly_score": -1.0,
                    "reason": "Device is down",
                })
                continue
            metrics = {
                col: float(row.get(col, 0) or 0)
                for col in detector.feature_columns
                if col in row.index
            }
            if not metrics:
                continue
            try:
                res = detector.predict(metrics)
                anomaly_rows.append({
                    "device": row.get("name", row.get("device", "unknown")),
                    "is_anomaly": res["is_anomaly"],
                    "anomaly_score": round(res["anomaly_score"], 4),
                    "lof": "⚠" if res.get("lof_anomaly") else "✓",
                    "zscore": "⚠" if res.get("zscore_anomaly") else "✓",
                    "top_feature": max(res["z_scores"], key=res["z_scores"].get)
                    if res.get("z_scores") else "—",
                })
            except Exception:
                continue

        anomaly_detected = [r for r in anomaly_rows if r["is_anomaly"]]
        normal = [r for r in anomaly_rows if not r["is_anomaly"]]

        a1, a2, a3 = st.columns(3)
        a1.metric("Devices scanned", len(anomaly_rows))
        a2.metric("Anomalies detected", len(anomaly_detected),
                  delta_color="inverse" if anomaly_detected else "off")
        a3.metric("Normal", len(normal))

        for r in anomaly_rows:
            css = "" if r["is_anomaly"] else "normal"
            label = "ANOMALY" if r["is_anomaly"] else "NORMAL"
            score_str = f"score={r['anomaly_score']:.4f}" if r["anomaly_score"] != -1.0 else "device down"
            top = r.get("top_feature", "—")
            lof_str = r.get("lof", "?")
            zs_str  = r.get("zscore", "?")
            st.markdown(
                f'<div class="anomaly-row {css}">'
                f'<span class="{"status-critical" if r["is_anomaly"] else "status-ok"}">● {label}</span>'
                f'&nbsp;&nbsp;<strong>{r["device"]}</strong>'
                f'&nbsp;<span style="color:#64748B">({score_str} &nbsp;|&nbsp; '
                f'LOF:{lof_str} &nbsp;Z-score:{zs_str} &nbsp;|&nbsp; top: {top})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        # Batch predict on traffic history
        st.markdown('<span class="kpi-label">Anomaly scan on traffic history</span>', unsafe_allow_html=True)
        try:
            traffic_sig = dataframe_signature(traffic_df)
            if (st.session_state.get("aiops_traffic_sig") != traffic_sig
                    or "aiops_batch_result" not in st.session_state):
                st.session_state.aiops_traffic_sig = traffic_sig
                st.session_state.aiops_batch_result = detector.predict_batch(traffic_df.copy())
            batch_result_df = st.session_state.aiops_batch_result
            n_anomalies = int(batch_result_df["is_anomaly"].sum())
            pct = n_anomalies / max(len(batch_result_df), 1) * 100
            st.metric("Anomalous traffic points", f"{n_anomalies} / {len(batch_result_df)} ({pct:.1f}%)")
            st.dataframe(
                batch_result_df[["timestamp", "bandwidth_mbps", "latency_ms", "is_anomaly", "anomaly_score"]]
                .sort_values("anomaly_score")
                .head(20),
                hide_index=True,
                **stretch_kwargs(),
            )
        except Exception as exc:
            st.caption(f"Batch scan unavailable: {exc}")
    else:
        st.info("Model not trained yet. Click **Train / Retrain model** above.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LOG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_log:
    st.markdown('<span class="kpi-label">Log-based root-cause analysis</span>', unsafe_allow_html=True)
    st.caption("Paste log output from a router, switch, firewall, or server. The AI identifies the root cause and suggests next steps.")

    # ── Sample snippets — rendered BEFORE the text_area so clicking a button
    #    only sets a prefill key, never touches the widget key directly.
    SAMPLES = {
        "Interface flap": (
            "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/0/1, changed state to down\n"
            "%LINK-3-UPDOWN: Interface GigabitEthernet0/0/1, changed state to down\n"
            "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/0/1, changed state to up\n"
            "%LINK-3-UPDOWN: Interface GigabitEthernet0/0/1, changed state to up"
        ),
        "OSPF adjacency failure": (
            "%OSPF-5-ADJCHG: Process 1, Nbr 10.0.0.2 on GigabitEthernet0/0 from FULL to DOWN, "
            "Neighbor Down: Dead timer expired\n"
            "%OSPF-5-ADJCHG: Process 1, Nbr 10.0.0.2 on GigabitEthernet0/0 from DOWN to INIT"
        ),
        "High CPU": (
            "Jul 15 09:12:31.453: %SYS-3-CPUHOG: Task is running for 2020 msecs, more than 2000 msecs (0/0),\n"
            "process = IP Input, pid = 14\n"
            "Jul 15 09:12:35.011: %SYS-3-CPUHOG: Task is running for 3120 msecs, process = ARP Input, pid = 30"
        ),
        "BGP peer down": (
            "%BGP-5-ADJCHANGE: neighbor 203.0.113.1 Down BGP Notification sent\n"
            "%BGP-3-NOTIFICATION: sent to neighbor 203.0.113.1 active 6/3 (Cease/peer unconfigured)"
        ),
    }

    st.markdown('<span class="kpi-label">Sample log snippets (click to load)</span>', unsafe_allow_html=True)
    sc = st.columns(len(SAMPLES))
    for i, (label, snippet) in enumerate(SAMPLES.items()):
        if sc[i].button(label, key=f"log_sample_{i}"):
            # Store in a separate prefill key — never write to the widget key
            st.session_state["log_prefill"] = snippet
            st.session_state.pop("log_analysis_result", None)
            st.rerun()

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # The text_area reads its initial value from the prefill key (no widget key set)
    log_text = st.text_area(
        "Paste log output here",
        value=st.session_state.get("log_prefill", ""),
        height=220,
        placeholder=(
            "Example:\n"
            "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down\n"
            "%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down\n"
            "Jun 10 14:23:01: %SYS-5-CONFIG_I: Configured from console by admin on vty0"
        ),
    )

    device_context = st.text_input(
        "Device context (optional)",
        placeholder="e.g. Cisco ISR 4331, edge router, WAN interface",
        key="log_device_ctx",
    )

    if st.button("Analyse log →", type="primary", key="log_analyse_btn", disabled=not log_text.strip()):
        with st.spinner("Analysing log…"):
            try:
                ctx = f"Device context: {device_context}\n\n" if device_context.strip() else ""
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a senior network engineer analysing device logs. "
                            "Identify the root cause, classify severity (critical/warning/info), "
                            "and provide 3-5 concrete numbered remediation steps. "
                            "Reference specific log lines where relevant. Be concise — under 300 words. "
                            "Only answer questions related to network operations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{ctx}Analyse this log output and provide root-cause analysis:\n\n"
                            f"```\n{log_text[:3000]}\n```"
                        ),
                    },
                ]
                log_result = get_llm_response(messages, temperature=0.3)
                st.session_state["log_analysis_result"] = log_result
            except LLMConfigurationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Log analysis failed: {exc}")

    if "log_analysis_result" in st.session_state:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.info(st.session_state["log_analysis_result"])
        st.download_button(
            "Download analysis (.txt)",
            data=st.session_state["log_analysis_result"],
            file_name="log_analysis.txt",
            mime="text/plain",
            key="dl_log",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTIVE WARNINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<span class="kpi-label">Predictive capacity warnings</span>', unsafe_allow_html=True)
    st.caption("Forecasts key metrics ahead of time and warns if a threshold will be breached.")

    forecaster = NetworkForecaster(method="auto")

    p1, p2, p3 = st.columns(3)
    horizon = p1.slider("Forecast horizon (minutes)", 5, 120, 30, 5, key="pred_horizon")
    bw_cap  = p2.number_input("Bandwidth capacity threshold (Mbps)", 50, 10000, 100, 10, key="pred_bw_cap")
    lat_cap = p3.number_input("Latency threshold (ms)", 20, 500, 100, 10, key="pred_lat_cap")

    # Forecasts are memoized per telemetry snapshot + user settings, so they
    # are recomputed only when the data or the inputs change.
    fc_sig = dataframe_signature(traffic_df)
    fc_key = f"pred_fc_{fc_sig}_{horizon}_{bw_cap}_{lat_cap}"
    if fc_key not in st.session_state:
        st.session_state[fc_key] = {
            "bandwidth_mbps":  (forecaster.check_capacity_threshold(traffic_df, "bandwidth_mbps",  bw_cap,  horizon), "Mbps"),
            "latency_ms":      (forecaster.check_capacity_threshold(traffic_df, "latency_ms",      lat_cap, horizon), "ms"),
            "packet_loss_pct": (forecaster.check_capacity_threshold(traffic_df, "packet_loss_pct", 2.0,     horizon), "%"),
        }
    forecasts = st.session_state[fc_key]

    for metric, (fc, unit) in forecasts.items():
        will_exceed = fc.get("will_exceed_threshold", False)
        pred = fc.get("predicted_value")
        err  = fc.get("error")
        label = metric.replace("_", " ").title()

        if err or pred is None:
            st.warning(f"**{label}** — forecast unavailable: {err or 'insufficient data'}")
            continue

        method = fc.get("method_used", "")

        if will_exceed:
            st.error(
                f"⚠ **{label}**: predicted **{pred:.1f}{unit}** in {horizon} min "
                f"will EXCEED threshold ({fc['threshold']}{unit})  ·  method: {method}"
            )
        else:
            st.success(
                f"✓ **{label}**: predicted **{pred:.1f}{unit}** in {horizon} min "
                f"— within threshold ({fc['threshold']}{unit})  ·  method: {method}"
            )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # LLM capacity planning narrative
    if st.button("Generate capacity planning narrative →", key="pred_narrative_btn"):
        with st.spinner("Generating narrative…"):
            try:
                summary_lines = []
                for metric, (fc, unit) in forecasts.items():
                    pred = fc.get("predicted_value")
                    if pred is not None:
                        status = "EXCEEDS threshold" if fc.get("will_exceed_threshold") else "within threshold"
                        summary_lines.append(
                            f"- {metric}: {pred:.1f}{unit} predicted in {horizon} min → {status}"
                        )
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a network capacity planning expert. "
                            "Write a concise (150-word) plain-language capacity planning narrative "
                            "for a network operations team based on the following forecasts. "
                            "Include risk assessment and 2-3 short-term actions."
                        ),
                    },
                    {"role": "user", "content": "\n".join(summary_lines)},
                ]
                narrative = get_llm_response(messages, temperature=0.4)
                st.session_state["pred_narrative"] = narrative
            except LLMConfigurationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Narrative generation failed: {exc}")

    if "pred_narrative" in st.session_state:
        st.info(st.session_state["pred_narrative"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REMEDIATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_remediation:
    st.markdown('<span class="kpi-label">Human-in-the-loop remediation</span>', unsafe_allow_html=True)
    st.caption(
        "Create incidents from active alerts, propose actions, and approve or reject them. "
        "All actions are simulated — no real changes are made to infrastructure."
    )

    remediation_engine = get_remediation_engine()
    pending = remediation_engine.get_pending_approvals()
    recent  = remediation_engine.get_incident_history(limit=30)

    r1, r2, r3 = st.columns(3)
    r1.metric("Pending approvals", len(pending))
    r2.metric("Total incidents", len(recent))
    approved = sum(1 for i in recent if i.state.value == "approved")
    r3.metric("Approved actions", approved)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # Pending approvals
    if pending:
        st.markdown('<span class="kpi-label">Pending approvals</span>', unsafe_allow_html=True)
        for incident in pending:
            with st.container():
                st.markdown(
                    f'<div class="alert-feed warning">'
                    f'<span class="mono status-warn">● PENDING APPROVAL</span>&nbsp;&nbsp;'
                    f'<strong>{incident.device}</strong> — {incident.issue_type}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if incident.suggested_action:
                    act = incident.suggested_action
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<span class="kpi-label">Action</span><br><span class="mono">{act.action_type.value}</span>', unsafe_allow_html=True)
                    c2.markdown(f'<span class="kpi-label">Impact</span><br><span class="mono">{act.estimated_impact}</span>', unsafe_allow_html=True)
                    c3.markdown(f'<span class="kpi-label">Rollback</span><br><span class="mono">{act.rollback_plan}</span>', unsafe_allow_html=True)
                ca, cr, _ = st.columns([1, 1, 5])
                if ca.button("Approve", key=f"rem_approve_{incident.incident_id}", type="primary"):
                    remediation_engine.approve_action(incident.incident_id, "Approved via AI Ops")
                    st.rerun()
                if cr.button("Reject", key=f"rem_reject_{incident.incident_id}"):
                    remediation_engine.reject_action(incident.incident_id, "Rejected via AI Ops")
                    st.rerun()
                st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # Create incident
    st.markdown('<span class="kpi-label">Create new incident</span>', unsafe_allow_html=True)
    if actionable_alerts:
        alert_options = {
            f"[{a['level'].upper()}] {a['device']} — {a['metric']}": a
            for a in actionable_alerts
        }
        sel_label = st.selectbox("Alert", list(alert_options.keys()), key="rem_alert_sel")
        sel_alert = alert_options[sel_label]
        action_type = st.selectbox(
            "Proposed action",
            [a.value for a in RemediationActionType],
            key="rem_action_type",
        )
        severity = st.select_slider("Severity", ["low", "medium", "high"], value="medium", key="rem_sev")
        if st.button("Create & submit for approval", key="rem_create_btn"):
            inc = remediation_engine.create_incident(sel_alert)
            remediation_engine.diagnose_incident(inc.incident_id, sel_alert["message"])
            remediation_engine.suggest_action(
                inc.incident_id,
                RemediationActionType(action_type),
                description=f"Auto-suggested for {sel_alert['device']}",
                severity=severity,
            )
            remediation_engine.submit_for_approval(inc.incident_id)
            st.success(f"Incident created — ID: {inc.incident_id[:8]}…")
            st.rerun()
    else:
        st.caption("No active alerts to create incidents from.")

    # Incident history table
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="kpi-label">Incident history</span>', unsafe_allow_html=True)
    if recent:
        state_color = {
            "approved": "status-ok", "rejected": "status-ok",
            "pending_approval": "status-warn", "detected": "status-warn",
            "diagnosed": "status-warn", "suggested": "status-warn",
        }
        for inc in recent[:20]:
            sc = state_color.get(inc.state.value, "status-ok")
            st.markdown(
                f'<div class="device-row">'
                f'<span class="mono {sc}">{inc.state.value.upper()}</span>&nbsp;&nbsp;'
                f'<strong>{inc.device}</strong> — {inc.issue_type}'
                f'&nbsp;<span style="color:#64748B;font-size:0.75rem;">{inc.detected_at.strftime("%m-%d %H:%M")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No incidents recorded yet.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<span class="kpi-label">AI-generated network health report</span>', unsafe_allow_html=True)
    st.caption("Aggregates current metrics, alerts, and incidents — sends to LLM for a structured markdown report.")

    rc1, rc2, rc3 = st.columns(3)
    time_window       = rc1.slider("Time window (hours)", 1, 168, 24, key="aiops_report_time")
    include_forecasts = rc2.checkbox("Include capacity forecast", value=True, key="aiops_rpt_fc")
    include_remediation = rc3.checkbox("Include incident log", value=True, key="aiops_rpt_rem")

    if st.button("Generate Report", type="primary", key="aiops_gen_report"):
        with st.spinner("Generating report…"):
            try:
                rg = get_report_generator()
                report = rg.generate_report(
                    time_window_hours=time_window,
                    include_forecasts=include_forecasts,
                    include_remediation=include_remediation,
                )
                st.session_state["aiops_report"] = report
            except LLMConfigurationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    if "aiops_report" in st.session_state:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown(st.session_state["aiops_report"])
        st.download_button(
            "Download (.md)",
            data=st.session_state["aiops_report"],
            file_name=f"network_report_{time_window}h.md",
            mime="text/markdown",
            key="aiops_dl_report",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown('<span class="kpi-label">System evaluation metrics</span>', unsafe_allow_html=True)
    st.caption(
        "Logged from every chatbot query and alert explanation. "
        "Use these numbers in the graduation report to quantify retrieval and latency performance."
    )

    try:
        cm  = get_evaluation_metrics(event_type="chatbot_query")
        am  = get_evaluation_metrics(event_type="alert_explanation")
        agm = get_evaluation_metrics(event_type="agent_query")

        ev1, ev2, ev3 = st.columns(3)

        with ev1:
            st.markdown('<span class="kpi-label">RAG Chatbot</span>', unsafe_allow_html=True)
            st.metric("Total queries",  cm["total_events"])
            st.metric("KB hit rate",    f"{cm['hit_rate']:.1%}")
            if cm["avg_score"]:
                st.metric("Avg similarity", f"{cm['avg_score']:.3f}")
            if cm["avg_latency_ms"]:
                st.metric("Avg latency",    f"{cm['avg_latency_ms']:.0f} ms")

        with ev2:
            st.markdown('<span class="kpi-label">Alert Explanations</span>', unsafe_allow_html=True)
            st.metric("Total explanations", am["total_events"])
            st.metric("KB hit rate",        f"{am['hit_rate']:.1%}")
            if am["avg_score"]:
                st.metric("Avg similarity", f"{am['avg_score']:.3f}")
            if am["avg_latency_ms"]:
                st.metric("Avg latency",    f"{am['avg_latency_ms']:.0f} ms")

        with ev3:
            st.markdown('<span class="kpi-label">Agent (tool-calling)</span>', unsafe_allow_html=True)
            st.metric("Total agent queries", agm["total_events"])
            if agm["avg_latency_ms"]:
                st.metric("Avg latency",     f"{agm['avg_latency_ms']:.0f} ms")
            st.caption("Tool-use details logged in network_bot.db · evaluation_events table.")

        # Raw event table
        with st.expander("Raw evaluation events (last 50)", expanded=False):
            try:
                from modules.storage import _connect, DB_PATH
                with _connect() as conn:
                    rows = conn.execute(
                        "SELECT event_type, query, retrieval_hit, retrieval_score, latency_ms, created_at "
                        "FROM evaluation_events ORDER BY id DESC LIMIT 50"
                    ).fetchall()
                if rows:
                    st.dataframe(pd.DataFrame([dict(r) for r in rows]), hide_index=True, **stretch_kwargs())
                else:
                    st.caption("No events recorded yet.")
            except Exception as exc:
                st.caption(f"Could not load raw events: {exc}")

    except Exception as exc:
        st.warning(f"Could not load evaluation metrics: {exc}")
