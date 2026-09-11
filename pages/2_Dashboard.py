"""Network Dashboard — NOC-style ops view with tabs, forecasting, and incident management."""

from __future__ import annotations

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.alerts import generate_alerts, alert_summary
from modules.diagnostics_bridge import explain_alert
from modules.llm_client import LLMConfigurationError
from modules.data_sources import get_data_source, get_cached_host_metrics, get_cached_devices, render_data_source_sidebar
from modules.forecasting import NetworkForecaster, dataframe_signature
from modules.settings import get_text, init_session_settings
try:
    from modules.storage import save_alert, get_history, get_evaluation_metrics
except (ImportError, KeyError):
    def save_alert(*_a, **_kw): pass  # type: ignore[misc]
    def get_history(*_a, **_kw): return {"alerts": []}  # type: ignore[misc]
    def get_evaluation_metrics(*_a, **_kw): return {"total_events": 0, "hit_rate": 0, "avg_latency_ms": None}  # type: ignore[misc]
from modules.ui import inject_global_css, stretch_kwargs
from modules.remediation import get_remediation_engine
from modules.reports import get_report_generator

st.set_page_config(page_title="Dashboard", page_icon="📡", layout="wide")
inject_global_css()
init_session_settings()

# ── Extra CSS for dashboard-specific density ──────────────────────────────────
st.markdown("""
<style>
.kpi-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
             color: #94A3B8; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem;
             font-weight: 600; line-height: 1.1; }
.kpi-up    { color: #22C55E; }
.kpi-warn  { color: #F59E0B; }
.kpi-crit  { color: #EF4444; }
.kpi-neutral { color: #E2E8F0; }
.device-row { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
              padding: 0.3rem 0; border-bottom: 1px solid #1E293B; }
.section-rule { border: none; border-top: 1px solid #334155; margin: 1rem 0; }
.forecast-card { background: #1E293B; border: 1px solid #334155;
                 border-radius: 3px; padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; }
.incident-row { background: #1E293B; border-left: 3px solid #F59E0B;
                padding: 0.6rem 0.9rem; margin-bottom: 0.6rem; }
.incident-row.critical { border-left-color: #EF4444; }
.incident-row.approved  { border-left-color: #22C55E; }
.incident-row.rejected  { border-left-color: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — thresholds + refresh ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="kpi-label">Alert Thresholds</span>', unsafe_allow_html=True)
    cpu_threshold = st.slider("CPU Warning (%)", 50, 100,
        int(st.session_state.thresholds.get("cpu_warning", 85)), step=5)
    pkt_threshold = st.slider("Packet Loss (%)", 0.5, 10.0,
        float(st.session_state.thresholds.get("packet_loss_warning", 1.5)), step=0.5)
    lat_threshold = st.slider("Latency Warning (ms)", 50, 500,
        int(st.session_state.thresholds.get("latency_warning_ms", 100)), step=10)

    st.session_state.thresholds = {
        "cpu_warning": float(cpu_threshold),
        "packet_loss_warning": float(pkt_threshold),
        "latency_warning_ms": float(lat_threshold),
    }

    st.divider()
    if st.button("↺  Refresh telemetry", use_container_width=True):
        for key in ("devices_df", "traffic_df", "alert_explanations",
                    "forecast_cache", "forecast_sig"):
            st.session_state.pop(key, None)
        get_cached_devices.clear()
        st.rerun()

    st.divider()
    st.markdown('<span class="kpi-label">Data source</span>', unsafe_allow_html=True)
    render_data_source_sidebar()

# ── Data loading (cached in session for the page lifetime) ────────────────────
data_source = get_data_source()

if "devices_df" not in st.session_state:
    st.session_state.devices_df = get_cached_devices(
        st.session_state.get("data_source", "real"))
if "traffic_df" not in st.session_state:
    st.session_state.traffic_df = data_source.get_traffic_history(hours=24)
if "alert_explanations" not in st.session_state:
    st.session_state.alert_explanations = {}

devices_df: pd.DataFrame = st.session_state.devices_df
traffic_df: pd.DataFrame = st.session_state.traffic_df
alerts = generate_alerts(devices_df, thresholds=st.session_state.thresholds)
summary = alert_summary(alerts)

# ── Derived KPIs ──────────────────────────────────────────────────────────────
up_count   = int((devices_df["status"] == "up").sum())
down_count = int((devices_df["status"] == "down").sum())
total      = len(devices_df)
up_pct     = (up_count / total * 100) if total else 0.0
avg_cpu    = devices_df.loc[devices_df["status"] == "up", "cpu_usage"].mean()
avg_lat    = devices_df.loc[devices_df["status"] == "up", "latency_ms"].mean()
crit_count = summary["critical"]
warn_count = summary["warning"]

# Availability colour
avail_cls = "kpi-up" if up_pct >= 95 else ("kpi-warn" if up_pct >= 80 else "kpi-crit")
alert_cls = "kpi-crit" if crit_count else ("kpi-warn" if warn_count else "kpi-up")

# ── Page header + KPI strip ───────────────────────────────────────────────────
st.markdown(f'<div class="page-header"><h2>{get_text("dashboard_title")}</h2></div>', unsafe_allow_html=True)
st.caption(get_text("dashboard_caption"))

k1, k2, k3, k4, k5, k6 = st.columns(6)

def _kpi(col, label: str, value: str, css_class: str = "kpi-neutral") -> None:
    col.markdown(
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {css_class}">{value}</div>',
        unsafe_allow_html=True,
    )

_kpi(k1, "Availability",    f"{up_pct:.1f}%",                            avail_cls)
_kpi(k2, "Devices up",      f"{up_count}/{total}",                       "kpi-up" if down_count == 0 else "kpi-warn")
_kpi(k3, "Devices down",    str(down_count),                             "kpi-crit" if down_count else "kpi-neutral")
_kpi(k4, "Active alerts",   f"{crit_count}C / {warn_count}W",           alert_cls)
_kpi(k5, "Avg CPU",         f"{avg_cpu:.1f}%" if pd.notna(avg_cpu) else "—",
     "kpi-warn" if pd.notna(avg_cpu) and avg_cpu > cpu_threshold else "kpi-neutral")
_kpi(k6, "Avg latency",     f"{avg_lat:.0f} ms" if pd.notna(avg_lat) else "—",
     "kpi-warn" if pd.notna(avg_lat) and avg_lat > lat_threshold else "kpi-neutral")

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_devices, tab_traffic, tab_incidents, tab_report = st.tabs([
    "Overview",
    "Devices",
    "Traffic & Forecast",
    "Incident Management",
    "Report",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW: alerts + host metrics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<span class="kpi-label">Alert Feed</span>', unsafe_allow_html=True)
        actionable = [a for a in alerts if a["level"] != "ok"]
        display_alerts = actionable if actionable else alerts

        for idx, alert in enumerate(display_alerts):
            level = alert["level"]
            css  = level if level in ("critical", "warning") else "ok"
            dot_cls = f"status-{level}" if level in ("critical","warning") else "status-ok"

            st.markdown(
                f'<div class="alert-feed {css}">'
                f'<span class="mono {dot_cls}">● {level.upper()}</span>&nbsp;&nbsp;'
                f'{alert["message"]}'
                f"</div>",
                unsafe_allow_html=True,
            )

            btn_key = f"explain_{idx}_{alert['device']}_{alert['metric']}"
            if level != "ok":
                if st.button("Explain →", key=btn_key, help="Get AI diagnosis for this alert"):
                    with st.spinner("Generating diagnosis…"):
                        try:
                            explanation = explain_alert(alert)
                            st.session_state.alert_explanations[btn_key] = explanation
                            save_alert(alert, explanation)
                        except LLMConfigurationError as exc:
                            st.session_state.alert_explanations[btn_key] = f"⚠ {exc}"
                        except Exception as exc:
                            st.session_state.alert_explanations[btn_key] = f"Error: {exc}"

                if btn_key in st.session_state.alert_explanations:
                    st.info(st.session_state.alert_explanations[btn_key])

    with right:
        st.markdown('<span class="kpi-label">Host Metrics (this machine)</span>', unsafe_allow_html=True)
        try:
            host = get_cached_host_metrics(st.session_state.get("data_source", "real"))
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("CPU", f"{host.get('cpu_percent', 0):.1f}%")
            hc2.metric("Memory", f"{host.get('memory_percent', 0):.1f}%")
            hc3.metric("Net I/O", f"{host.get('network_throughput_mbps', 0):.1f} Mbps")
            if host.get("error"):
                st.caption(f"Note: {host['error']}")
        except Exception as exc:
            st.caption(f"Host metrics unavailable: {exc}")

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<span class="kpi-label">Alert History</span>', unsafe_allow_html=True)
        try:
            history_data = get_history(limit=20)
            past_alerts = history_data.get("alerts", [])
            if past_alerts:
                for ev in past_alerts[:8]:
                    try:
                        ad = json.loads(ev["alert_json"])
                        lvl = ad.get("level","warning")
                        st.markdown(
                            f'<div class="alert-feed {lvl}" style="margin-bottom:0.3rem;">'
                            f'<span class="mono" style="font-size:0.7rem;">{ev["created_at"][:16]}</span>'
                            f'&nbsp;<span class="mono status-{lvl}">● {lvl.upper()}</span>'
                            f'&nbsp;{ad.get("message","")[:60]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        continue
            else:
                st.caption("No alert history yet.")
        except Exception as exc:
            st.caption(f"Could not load history: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEVICES: full table + per-device bar charts
# ═══════════════════════════════════════════════════════════════════════════════
with tab_devices:
    st.markdown('<span class="kpi-label">Device Inventory</span>', unsafe_allow_html=True)

    # Render compact table — works for both Simulated (8 cols) and Live (9 cols with 'host')
    display_df = devices_df.copy()
    display_df["cpu_usage"]       = display_df["cpu_usage"].apply(lambda v: f"{v:.1f}%")
    display_df["latency_ms"]      = display_df["latency_ms"].apply(lambda v: f"{v:.1f} ms")
    display_df["packet_loss_pct"] = display_df["packet_loss_pct"].apply(lambda v: f"{v:.2f}%")
    display_df["bandwidth_mbps"]  = display_df["bandwidth_mbps"].apply(lambda v: f"{v:.0f} Mbps")
    display_df["uptime_pct"]      = display_df["uptime_pct"].apply(lambda v: f"{v:.1f}%")
    display_df = display_df.rename(columns={
        "name": "Name", "host": "Host", "type": "Type", "status": "Status",
        "cpu_usage": "CPU", "latency_ms": "Latency", "packet_loss_pct": "Pkt Loss",
        "bandwidth_mbps": "Bandwidth", "uptime_pct": "Uptime",
    })
    st.dataframe(display_df, hide_index=True, **stretch_kwargs())

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="kpi-label">Per-device metrics</span>', unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    raw = st.session_state.devices_df  # use raw numeric df for charts

    # ── Shared dark layout helper ──────────────────────────────────────────
    _dark = dict(plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
                 font_color="#94A3B8", margin=dict(t=44, b=0, l=0, r=0),
                 xaxis_tickangle=-30)

    with bc1:
        fig = go.Figure()
        cpu_colors = [
            "#EF4444" if v >= cpu_threshold else "#F59E0B" if v >= cpu_threshold * 0.75 else "#22C55E"
            for v in raw["cpu_usage"]
        ]
        fig.add_trace(go.Bar(
            x=raw["name"], y=raw["cpu_usage"],
            marker_color=cpu_colors, name="CPU %",
            text=[f"{v:.1f}%" for v in raw["cpu_usage"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>CPU: %{y:.1f}%%<extra></extra>",
        ))
        # Threshold reference lines
        fig.add_hline(y=cpu_threshold, line_dash="dash", line_color="#EF4444",
                      line_width=1.2,
                      annotation_text=f"Critical {cpu_threshold}%", annotation_font_color="#EF4444",
                      annotation_font_size=10, annotation_position="top right")
        fig.add_hline(y=cpu_threshold * 0.75, line_dash="dot", line_color="#F59E0B",
                      line_width=1,
                      annotation_text=f"Warning {cpu_threshold * 0.75:.0f}%", annotation_font_color="#F59E0B",
                      annotation_font_size=9, annotation_position="top right")
        # Peak annotation
        peak_idx = raw["cpu_usage"].idxmax()
        peak_name = raw.loc[peak_idx, "name"]
        peak_val  = raw.loc[peak_idx, "cpu_usage"]
        fig.add_annotation(x=peak_name, y=peak_val, text=f"Peak {peak_val:.0f}%",
                           showarrow=True, arrowhead=2, arrowcolor="#F59E0B",
                           font=dict(color="#F59E0B", size=10), ay=-30)
        fig.update_layout(title="CPU Usage (%)", yaxis_range=[0, 110],
                          xaxis=dict(gridcolor="#1E293B"),
                          yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"), **_dark)
        st.plotly_chart(fig, **stretch_kwargs())

    with bc2:
        lat_df = raw.dropna(subset=["latency_ms"])
        fig2 = go.Figure()
        lat_colors = [
            "#EF4444" if v >= lat_threshold else "#F59E0B" if v >= lat_threshold * 0.75 else "#22C55E"
            for v in lat_df["latency_ms"]
        ]
        fig2.add_trace(go.Bar(
            x=lat_df["name"], y=lat_df["latency_ms"],
            marker_color=lat_colors, name="Latency ms",
            text=[f"{v:.0f} ms" for v in lat_df["latency_ms"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Latency: %{y:.1f} ms<extra></extra>",
        ))
        fig2.add_hline(y=lat_threshold, line_dash="dash", line_color="#EF4444",
                       line_width=1.2,
                       annotation_text=f"Critical {lat_threshold} ms", annotation_font_color="#EF4444",
                       annotation_font_size=10, annotation_position="top right")
        fig2.add_hline(y=lat_threshold * 0.75, line_dash="dot", line_color="#F59E0B",
                       line_width=1,
                       annotation_text=f"Warning {lat_threshold * 0.75:.0f} ms", annotation_font_color="#F59E0B",
                       annotation_font_size=9, annotation_position="top right")
        if len(lat_df) > 0:
            peak_lat_idx = lat_df["latency_ms"].idxmax()
            fig2.add_annotation(
                x=lat_df.loc[peak_lat_idx, "name"], y=lat_df.loc[peak_lat_idx, "latency_ms"],
                text=f"Peak {lat_df.loc[peak_lat_idx, 'latency_ms']:.0f} ms",
                showarrow=True, arrowhead=2, arrowcolor="#F59E0B",
                font=dict(color="#F59E0B", size=10), ay=-30)
        fig2.update_layout(title="Latency (ms)",
                           xaxis=dict(gridcolor="#1E293B"),
                           yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"), **_dark)
        st.plotly_chart(fig2, **stretch_kwargs())

    # Bandwidth with threshold-aware coloring
    bw_max = raw["bandwidth_mbps"].max() if len(raw) > 0 else 1
    bw_threshold = bw_max * 0.85  # high utilization warning
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=raw["name"], y=raw["bandwidth_mbps"],
        marker_color=[
            "#F59E0B" if v >= bw_threshold else "#38BDF8"
            for v in raw["bandwidth_mbps"]
        ],
        name="Bandwidth Mbps",
        text=[f"{v:.0f}" for v in raw["bandwidth_mbps"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Bandwidth: %{y:.0f} Mbps<extra></extra>",
    ))
    fig3.add_hline(y=bw_threshold, line_dash="dot", line_color="#F59E0B",
                   line_width=1,
                   annotation_text=f"High utilization {bw_threshold:.0f} Mbps",
                   annotation_font_color="#F59E0B", annotation_font_size=9,
                   annotation_position="top right")
    fig3.update_layout(title="Bandwidth Snapshot (Mbps)",
                       xaxis=dict(gridcolor="#1E293B"),
                       yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"), **_dark)
    st.plotly_chart(fig3, **stretch_kwargs())

    # ── Device Health Radar Chart ──────────────────────────────────────────
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="kpi-label">Device Health Overview</span>', unsafe_allow_html=True)
    st.caption("Radar chart — normalized metrics per device (lower = healthier)")

    radar_df = raw.copy()
    # Normalize each metric to 0-1 range for radar comparison
    for col in ["cpu_usage", "latency_ms", "packet_loss_pct"]:
        col_max = radar_df[col].max() if radar_df[col].max() > 0 else 1
        radar_df[f"{col}_norm"] = radar_df[col] / col_max
    # Invert bandwidth (higher is better) → lower = healthier
    bw_max_r = radar_df["bandwidth_mbps"].max() if radar_df["bandwidth_mbps"].max() > 0 else 1
    radar_df["bandwidth_norm"] = 1 - (radar_df["bandwidth_mbps"] / bw_max_r)

    radar_metrics = ["cpu_usage_norm", "latency_ms_norm", "packet_loss_pct_norm", "bandwidth_norm"]
    radar_labels  = ["CPU Load", "Latency", "Packet Loss", "BW Under-util"]

    fig_radar = go.Figure()
    radar_colors = ["#38BDF8", "#A78BFA", "#22C55E", "#F59E0B", "#EF4444",
                    "#F87171", "#34D399", "#FB923C"]

    def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"

    for i, (_, row) in enumerate(radar_df.iterrows()):
        vals = [row[m] for m in radar_metrics]
        vals.append(vals[0])  # close the polygon
        cats = list(radar_labels) + [radar_labels[0]]
        color = radar_colors[i % len(radar_colors)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=str(row["name"]),
            line=dict(color=color, width=2),
            fillcolor=_hex_to_rgba(color, 0.08),
            opacity=0.85,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0F172A",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1E293B",
                            tickfont=dict(size=9, color="#64748B")),
            angularaxis=dict(gridcolor="#1E293B",
                             tickfont=dict(size=10, color="#94A3B8")),
        ),
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font_color="#94A3B8",
        legend=dict(orientation="h", y=-0.15, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9)),
        margin=dict(t=20, b=0, l=0, r=0),
        height=420,
    )
    st.plotly_chart(fig_radar, **stretch_kwargs())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAFFIC & FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_traffic:
    # Forecasting inline KPIs — computed once per telemetry snapshot, then
    # reused on every rerun (the data hasn't changed).
    forecast_sig = dataframe_signature(traffic_df)
    if st.session_state.get("forecast_sig") != forecast_sig:
        forecaster = NetworkForecaster(method="auto")
        st.session_state.forecast_sig = forecast_sig
        st.session_state.forecast_cache = {
            "bandwidth_mbps":  forecaster.forecast_metric(traffic_df, "bandwidth_mbps",  horizon_minutes=30),
            "latency_ms":      forecaster.forecast_metric(traffic_df, "latency_ms",      horizon_minutes=30),
            "packet_loss_pct": forecaster.forecast_metric(traffic_df, "packet_loss_pct", horizon_minutes=30),
        }
    fc_bw  = st.session_state.forecast_cache["bandwidth_mbps"]
    fc_lat = st.session_state.forecast_cache["latency_ms"]
    fc_pkt = st.session_state.forecast_cache["packet_loss_pct"]

    fc1, fc2, fc3 = st.columns(3)
    def _fc_metric(col, label, forecast_dict, unit="", invert=False):
        v = forecast_dict.get("predicted_value")
        err = forecast_dict.get("error")
        if err or v is None:
            col.metric(f"30-min forecast — {label}", "n/a", help=err or "insufficient data")
        else:
            col.metric(
                f"30-min forecast — {label}",
                f"{v:.1f}{unit}",
            )

    _fc_metric(fc1, "Bandwidth",   fc_bw,  " Mbps")
    _fc_metric(fc2, "Latency",     fc_lat, " ms")
    _fc_metric(fc3, "Packet Loss", fc_pkt, "%")

    method_used = fc_bw.get("method_used", "rolling")
    st.caption(f"Forecast method: {method_used} · {fc_bw.get('data_points_used', 0)} data points")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # 24-hour traffic chart — bandwidth + latency dual axis with moving average & forecast band
    st.markdown('<span class="kpi-label">24-hour traffic history</span>', unsafe_allow_html=True)

    tdf = traffic_df.copy()
    tdf["timestamp"] = pd.to_datetime(tdf["timestamp"])

    # Compute rolling averages for trend overlay
    window_size = max(3, len(tdf) // 20)
    tdf["bw_ma"]  = tdf["bandwidth_mbps"].rolling(window=window_size, min_periods=1).mean()
    tdf["lat_ma"] = tdf["latency_ms"].rolling(window=window_size, min_periods=1).mean()

    fig_traffic = go.Figure()

    # Bandwidth — area fill
    fig_traffic.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["bandwidth_mbps"],
        mode="lines", name="Bandwidth (Mbps)",
        line=dict(color="#38BDF8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.06)",
        hovertemplate="<b>%{x|%H:%M}</b><br>BW: %{y:.1f} Mbps<extra></extra>",
    ))
    # Bandwidth moving average
    fig_traffic.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["bw_ma"],
        mode="lines", name=f"BW moving avg ({window_size})",
        line=dict(color="#38BDF8", width=2, dash="dash"),
        opacity=0.7,
    ))

    # Forecast point marker
    if fc_bw.get("predicted_value") is not None:
        last_ts = tdf["timestamp"].max()
        fut_ts  = last_ts + pd.Timedelta(minutes=30)
        # Forecast diamond marker
        fig_traffic.add_trace(go.Scatter(
            x=[fut_ts], y=[fc_bw["predicted_value"]],
            mode="markers+text", name="BW forecast (+30 min)",
            marker=dict(color="#F59E0B", size=12, symbol="diamond",
                        line=dict(color="#0F172A", width=2)),
            text=[f"{fc_bw['predicted_value']:.1f}"],
            textposition="top center", textfont=dict(color="#F59E0B", size=10),
        ))

    # Latency — dotted line on y2
    fig_traffic.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["latency_ms"],
        mode="lines", name="Latency (ms)",
        line=dict(color="#A78BFA", width=1, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x|%H:%M}</b><br>Latency: %{y:.1f} ms<extra></extra>",
    ))
    # Latency moving average
    fig_traffic.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["lat_ma"],
        mode="lines", name=f"Latency MA ({window_size})",
        line=dict(color="#A78BFA", width=2, dash="dashdot"),
        yaxis="y2", opacity=0.6,
    ))

    # Latency forecast marker on y2
    if fc_lat.get("predicted_value") is not None:
        fig_traffic.add_trace(go.Scatter(
            x=[fut_ts], y=[fc_lat["predicted_value"]],
            mode="markers", name="Latency forecast",
            marker=dict(color="#A78BFA", size=10, symbol="diamond-open",
                        line=dict(width=2)),
            yaxis="y2",
        ))

    # Threshold warning band for bandwidth (shaded region above threshold)
    bw_thresh_val = raw["bandwidth_mbps"].max() * 0.9 if "raw" in dir() else None
    fig_traffic.add_hline(y=lat_threshold, yref="y2", line_dash="dot",
                          line_color="#EF4444", line_width=0.8, opacity=0.5,
                          annotation_text=f"Latency warn {lat_threshold} ms",
                          annotation_font_color="#EF4444", annotation_font_size=8,
                          annotation_position="bottom right", annotation_yref="y2")

    fig_traffic.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8",
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9)),
        xaxis=dict(gridcolor="#1E293B", showspikes=True, spikecolor="#475569",
                   spikethickness=1, spikedash="dot", spikemode="across"),
        yaxis=dict(title=dict(text="Bandwidth (Mbps)", font=dict(color="#38BDF8")),
                   gridcolor="#1E293B"),
        yaxis2=dict(title=dict(text="Latency (ms)", font=dict(color="#A78BFA")),
                    overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=0, l=0, r=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_traffic, **stretch_kwargs())

    # Packet loss chart — with threshold line, MA, and spike annotations
    st.markdown('<span class="kpi-label">Packet loss %</span>', unsafe_allow_html=True)
    tdf["pkt_ma"] = tdf["packet_loss_pct"].rolling(window=window_size, min_periods=1).mean()
    pkt_mean = tdf["packet_loss_pct"].mean()
    pkt_std  = tdf["packet_loss_pct"].std() if len(tdf) > 1 else 0

    fig_pkt = go.Figure()
    fig_pkt.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["packet_loss_pct"],
        mode="lines", name="Packet loss %",
        line=dict(color="#F87171", width=1),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.06)",
        hovertemplate="<b>%{x|%H:%M}</b><br>Loss: %{y:.2f}%%<extra></extra>",
    ))
    # Moving average overlay
    fig_pkt.add_trace(go.Scatter(
        x=tdf["timestamp"], y=tdf["pkt_ma"],
        mode="lines", name=f"MA ({window_size})",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        opacity=0.8,
    ))
    # Threshold reference line
    fig_pkt.add_hline(y=pkt_threshold, line_dash="dash", line_color="#EF4444",
                      line_width=1.2,
                      annotation_text=f"Warning {pkt_threshold}%", annotation_font_color="#EF4444",
                      annotation_font_size=10, annotation_position="top right")
    # Mean reference line
    fig_pkt.add_hline(y=pkt_mean, line_dash="dot", line_color="#64748B",
                      line_width=1,
                      annotation_text=f"Mean {pkt_mean:.2f}%", annotation_font_color="#64748B",
                      annotation_font_size=9, annotation_position="top left")
    # Annotate spikes above mean + 2*std
    if pkt_std > 0:
        spike_threshold = pkt_mean + 2 * pkt_std
        spikes = tdf[tdf["packet_loss_pct"] > spike_threshold]
        for _, spike_row in spikes.head(5).iterrows():
            fig_pkt.add_annotation(
                x=spike_row["timestamp"], y=spike_row["packet_loss_pct"],
                text=f"{spike_row['packet_loss_pct']:.2f}%",
                showarrow=True, arrowhead=2, arrowcolor="#F87171",
                font=dict(color="#F87171", size=9), ay=-25,
            )

    fig_pkt.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8",
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=9)),
        xaxis=dict(gridcolor="#1E293B", showspikes=True, spikecolor="#475569",
                   spikethickness=1, spikedash="dot"),
        yaxis=dict(title="Packet Loss (%)", gridcolor="#1E293B"),
        margin=dict(t=10, b=0, l=0, r=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_pkt, **stretch_kwargs())

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INCIDENT MANAGEMENT (Remediation approvals + create incident)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_incidents:
    remediation_engine = get_remediation_engine()
    pending = remediation_engine.get_pending_approvals()
    recent  = remediation_engine.get_incident_history(limit=20)

    # ── Pending approvals (surfaced prominently) ──────────────────────────────
    if pending:
        st.markdown(
            f'<div class="kpi-value kpi-warn" style="font-size:1rem;">'
            f'▲ {len(pending)} pending approval{"s" if len(pending) != 1 else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        for incident in pending:
            with st.container():
                st.markdown(
                    f'<div class="incident-row">'
                    f'<span class="mono status-warn">● PENDING APPROVAL</span>&nbsp;&nbsp;'
                    f'<strong>{incident.device}</strong> — {incident.issue_type}'
                    f'&nbsp;<span class="mono" style="color:#64748B; font-size:0.75rem;">'
                    f'detected {incident.detected_at.strftime("%Y-%m-%d %H:%M")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if incident.diagnosis:
                    st.caption(f"Diagnosis: {incident.diagnosis[:120]}")
                if incident.suggested_action:
                    act = incident.suggested_action
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.markdown(f'<span class="kpi-label">Proposed action</span><br>'
                                 f'<span class="mono">{act.action_type.value}</span>',
                                 unsafe_allow_html=True)
                    ic2.markdown(f'<span class="kpi-label">Estimated impact</span><br>'
                                 f'<span class="mono">{act.estimated_impact}</span>',
                                 unsafe_allow_html=True)
                    ic3.markdown(f'<span class="kpi-label">Rollback plan</span><br>'
                                 f'<span class="mono">{act.rollback_plan}</span>',
                                 unsafe_allow_html=True)
                    st.caption(f"Severity: {act.severity} · {act.description}")

                ca, cr, _ = st.columns([1, 1, 5])
                if ca.button("Approve", key=f"approve_{incident.incident_id}", type="primary"):
                    remediation_engine.approve_action(incident.incident_id, "Approved via Dashboard")
                    st.rerun()
                if cr.button("Reject", key=f"reject_{incident.incident_id}"):
                    remediation_engine.reject_action(incident.incident_id, "Rejected via Dashboard")
                    st.rerun()
                st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-feed ok"><span class="mono status-ok">● No pending approvals</span></div>',
            unsafe_allow_html=True,
        )

    # ── Create incident from current alert ────────────────────────────────────
    st.markdown('<span class="kpi-label">Create incident from alert</span>', unsafe_allow_html=True)
    actionable_alerts = [a for a in alerts if a["level"] != "ok"]
    if actionable_alerts:
        alert_options = {
            f"[{a['level'].upper()}] {a['device']} — {a['metric']}": a
            for a in actionable_alerts
        }
        selected_label = st.selectbox("Select alert", list(alert_options.keys()), key="incident_alert_sel")
        selected_alert = alert_options[selected_label]

        from modules.remediation import RemediationActionType
        action_type = st.selectbox(
            "Proposed action",
            [a.value for a in RemediationActionType],
            key="incident_action_sel",
        )
        severity = st.select_slider("Severity", ["low", "medium", "high"], value="medium", key="incident_sev")

        if st.button("Create & submit for approval", key="create_incident_btn"):
            inc = remediation_engine.create_incident(selected_alert)
            remediation_engine.diagnose_incident(inc.incident_id, selected_alert["message"])
            remediation_engine.suggest_action(
                inc.incident_id,
                RemediationActionType(action_type),
                description=f"Auto-suggested for {selected_alert['device']}",
                severity=severity,
            )
            remediation_engine.submit_for_approval(inc.incident_id)
            st.success(f"Incident created and submitted for approval (ID: {inc.incident_id[:8]}…)")
            st.rerun()
    else:
        st.caption("No active alerts to create incidents from.")

    # ── Incident history ──────────────────────────────────────────────────────
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="kpi-label">Recent incident history</span>', unsafe_allow_html=True)
    if recent:
        state_color = {
            "approved": "kpi-up", "rejected": "kpi-neutral",
            "pending_approval": "kpi-warn", "detected": "kpi-warn",
            "diagnosed": "kpi-warn", "suggested": "kpi-warn",
        }
        for inc in recent[:15]:
            sc = state_color.get(inc.state.value, "kpi-neutral")
            st.markdown(
                f'<div class="device-row">'
                f'<span class="mono {sc}">{inc.state.value.upper()}</span>&nbsp;&nbsp;'
                f'<strong>{inc.device}</strong> — {inc.issue_type}'
                f'&nbsp;<span style="color:#64748B; font-size:0.75rem;">'
                f'{inc.detected_at.strftime("%m-%d %H:%M")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No incidents recorded yet.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown('<span class="kpi-label">AI-generated network health report</span>', unsafe_allow_html=True)
    st.caption("Pulls current metrics, alerts, and incident log — sends to LLM for a structured markdown report.")

    rc1, rc2 = st.columns(2)
    with rc1:
        time_window = st.slider("Time window (hours)", 1, 168, 24, key="report_time")
    with rc2:
        include_forecasts   = st.checkbox("Include capacity forecast", value=True, key="rpt_fc")

    if st.button("Generate Report", type="primary", key="gen_report_btn"):
        with st.spinner("Generating report…"):
            try:
                rg = get_report_generator()
                report = rg.generate_report(
                    time_window_hours=time_window,
                    include_forecasts=include_forecasts,
                    include_remediation=True,
                )
                st.session_state.generated_report = report
            except LLMConfigurationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    if "generated_report" in st.session_state:
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown(st.session_state.generated_report)
        st.download_button(
            label="Download (.md)",
            data=st.session_state.generated_report,
            file_name=f"network_report_{time_window}h.md",
            mime="text/markdown",
        )

    # Evaluation metrics — collapsed, for graduation report reference
    with st.expander("Evaluation metrics (graduation report)", expanded=False):
        try:
            cm = get_evaluation_metrics(event_type="chatbot_query")
            am = get_evaluation_metrics(event_type="alert_explanation")
            agm = get_evaluation_metrics(event_type="agent_query")

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.markdown("**RAG Chatbot**")
                st.metric("Queries", cm["total_events"])
                st.metric("Hit rate", f"{cm['hit_rate']:.2%}")
                if cm["avg_latency_ms"]:
                    st.metric("Avg latency", f"{cm['avg_latency_ms']:.0f} ms")
            with ec2:
                st.markdown("**Alert Explanations**")
                st.metric("Explanations", am["total_events"])
                st.metric("Hit rate", f"{am['hit_rate']:.2%}")
                if am["avg_latency_ms"]:
                    st.metric("Avg latency", f"{am['avg_latency_ms']:.0f} ms")
            with ec3:
                st.markdown("**Agent (tool-calling)**")
                st.metric("Queries", agm["total_events"])
                if agm["avg_latency_ms"]:
                    st.metric("Avg latency", f"{agm['avg_latency_ms']:.0f} ms")
                st.caption("Detailed tool-use metrics are logged in the database.")
        except Exception as exc:
            st.warning(f"Could not load evaluation metrics: {exc}")
