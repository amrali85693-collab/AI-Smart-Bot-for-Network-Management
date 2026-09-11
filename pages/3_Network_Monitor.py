import pandas as pd
import streamlit as st
import json
import time

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from modules.network_monitor import (
    real_ping, http_health_check, batch_ping, calculate_latency_stats,
    dns_lookup, check_port, batch_port_check, traceroute,
    create_latency_chart, create_multi_host_chart, ping_host_telemetry,
    check_alert_thresholds, continuous_monitor, PORT_PRESETS,
    get_service_info, scan_port_range, assess_port_security,
    analyze_ping_history, analyze_http_history, analyze_dns_history,
    calculate_network_health_score, compare_hosts
)
from modules.settings import get_text, init_session_settings
from modules.ui import inject_global_css, stretch_kwargs

st.set_page_config(page_title="Network Monitor", page_icon="🔍", layout="wide")
inject_global_css()
init_session_settings()

st.markdown(f'<div class="page-header"><h2>{get_text("monitor_title")}</h2></div>', unsafe_allow_html=True)
st.caption(get_text("monitor_caption"))

# Initialize session state
if "ping_history" not in st.session_state:
    st.session_state.ping_history = []
if "http_history" not in st.session_state:
    st.session_state.http_history = []
if "dns_history" not in st.session_state:
    st.session_state.dns_history = []
if "port_history" not in st.session_state:
    st.session_state.port_history = []
if "traceroute_history" not in st.session_state:
    st.session_state.traceroute_history = []
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "alert_history" not in st.session_state:
    st.session_state.alert_history = []
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

# Quick targets
QUICK_HOSTS = ["8.8.8.8", "1.1.1.1", "google.com", "cloudflare.com"]
QUICK_URLS = ["https://www.google.com", "https://www.cloudflare.com", "https://httpbin.org/status/200", "https://api.github.com"]

# Tabs for different monitoring features
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["ICMP Ping", "HTTP Health", "DNS Lookup", "Port Scanner", "Batch Monitor", "Traceroute", "Analysis Dashboard"]
)

with tab1:
    st.markdown("### Single Host Ping")
    col_input, col_btn, col_count = st.columns([3, 1, 1])
    with col_input:
        host = st.text_input("Host or IP", placeholder="e.g. 8.8.8.8 or example.com", key="ping_host")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_ping = st.button("Run ping", type="primary", key="run_ping")
    with col_count:
        st.markdown("<br>", unsafe_allow_html=True)
        ping_count = st.number_input("Count", min_value=1, max_value=10, value=3, key="ping_count")

    st.markdown("**Quick targets**")
    quick_cols = st.columns(len(QUICK_HOSTS))
    for i, quick_host in enumerate(QUICK_HOSTS):
        if quick_cols[i].button(quick_host, key=f"quick_{quick_host}"):
            host = quick_host
            run_ping = True

    if run_ping and host:
        with st.spinner(f"Pinging {host} {ping_count} times..."):
            if ping_count > 1:
                result = ping_host_telemetry(host.strip(), count=ping_count)
                result["host"] = host.strip()
                result["timestamp"] = pd.Timestamp.now().isoformat()
                st.session_state.ping_history.insert(0, result)
            else:
                result = real_ping(host.strip())
                st.session_state.ping_history.insert(0, result)

    if st.session_state.ping_history:
        df = pd.DataFrame(st.session_state.ping_history)
        st.dataframe(df, hide_index=True, **stretch_kwargs())
        
        # Statistics
        latencies = [r.get("latency_ms") for r in st.session_state.ping_history if r.get("latency_ms") is not None]
        if latencies:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Avg Latency", f"{sum(latencies)/len(latencies):.2f} ms")
            col_stat2.metric("Min Latency", f"{min(latencies):.2f} ms")
            col_stat3.metric("Max Latency", f"{max(latencies):.2f} ms")
        
        # Detailed Analysis
        with st.expander("Detailed Ping Analysis", expanded=False):
            ping_analysis = analyze_ping_history(st.session_state.ping_history)
            if "error" not in ping_analysis:
                st.markdown(f"**Status:** {ping_analysis['status'].upper()}")
                st.markdown(f"**Success Rate:** {ping_analysis['success_rate']:.1f}%")
                
                if ping_analysis.get('latency'):
                    st.markdown("**Advanced Latency Metrics:**")
                    col_adv1, col_adv2, col_adv3 = st.columns(3)
                    col_adv1.metric("Median", f"{ping_analysis['latency']['median_ms']}ms")
                    col_adv2.metric("Std Dev", f"{ping_analysis['latency']['std_dev_ms']}ms")
                    col_adv3.metric("Jitter", f"{ping_analysis['latency']['jitter_ms']}ms")
                
                if ping_analysis['recommendations']:
                    st.markdown("**Analysis Recommendations:**")
                    for rec in ping_analysis['recommendations']:
                        st.info(rec)
            else:
                st.warning(ping_analysis['error'])

        # Show chart
        if len(st.session_state.ping_history) > 1:
            chart_fig = create_latency_chart(st.session_state.ping_history, host)
            if chart_fig is not None:
                st.plotly_chart(chart_fig, **stretch_kwargs())

        latest = st.session_state.ping_history[0]
        status = latest.get("status", "unknown")
        status_class = "up" if status == "up" else "critical" if status == "down" else "warn"
        st.markdown(
            f'<p class="mono status-{status_class}">Latest: {latest.get("host")} — '
            f'{status.upper()} '
            f'({latest.get("latency_ms", "—")} ms)</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No ping results yet. Enter a host or pick a quick target.")

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("Clear ping history", key="clear_ping"):
            st.session_state.ping_history = []
            st.rerun()
    with col_export:
        if st.button("Export ping history (JSON)", key="export_ping"):
            json_str = json.dumps(st.session_state.ping_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="ping_history.json",
                mime="application/json"
            )

    st.markdown(
        """
        **Note:** ICMP may require elevated privileges on some hosts. Private RFC1918 addresses
        will only respond when the app runs inside your network or via VPN/agent.
        """
    )

with tab2:
    col_url_input, col_url_btn = st.columns([3, 1])
    with col_url_input:
        url = st.text_input("URL", placeholder="e.g. https://example.com", key="http_url")
    with col_url_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_http = st.button("Check HTTP", type="primary", key="run_http")

    st.markdown("**Quick URLs**")
    quick_url_cols = st.columns(len(QUICK_URLS))
    for i, quick_url in enumerate(QUICK_URLS):
        if quick_url_cols[i].button(quick_url, key=f"quick_url_{i}"):
            url = quick_url
            run_http = True

    if run_http and url:
        with st.spinner(f"Checking {url}..."):
            result = http_health_check(url.strip())
        st.session_state.http_history.insert(0, result)

    if st.session_state.http_history:
        df = pd.DataFrame(st.session_state.http_history)
        st.dataframe(df, hide_index=True, **stretch_kwargs())

        latest = st.session_state.http_history[0]
        status = latest.get("status", "unknown")
        status_class = "up" if status == "up" else "critical" if status == "down" else "warn"
        st.markdown(
            f'<p class="mono status-{status_class}">Latest: {latest.get("url")} — '
            f'{status.upper()} '
            f'({latest.get("http_status", "—")}) '
            f'({latest.get("latency_ms", "—")} ms)</p>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No HTTP check results yet. Enter a URL or pick a quick target.")

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("Clear HTTP history", key="clear_http"):
            st.session_state.http_history = []
            st.rerun()
    with col_export:
        if st.button("Export HTTP history (JSON)", key="export_http"):
            json_str = json.dumps(st.session_state.http_history, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="http_history.json",
                mime="application/json"
            )

with tab3:
    st.markdown("### DNS Lookup")
    col_dns_input, col_dns_btn = st.columns([3, 1])
    with col_dns_input:
        dns_host = st.text_input("Hostname", placeholder="e.g. google.com", key="dns_host")
    with col_dns_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_dns = st.button("Lookup DNS", type="primary", key="run_dns")

    if run_dns and dns_host:
        with st.spinner(f"Looking up {dns_host}..."):
            result = dns_lookup(dns_host.strip())
        st.session_state.dns_history.insert(0, result)

    if st.session_state.dns_history:
        for i, result in enumerate(st.session_state.dns_history):
            status = result.get("status", "unknown")
            status_class = "up" if status == "success" else "critical" if status == "error" else "warn"
            
            with st.expander(f"{result.get('hostname')} — {status.upper()}", expanded=i == 0):
                st.markdown(f'<p class="mono status-{status_class}">Status: {status.upper()}</p>', unsafe_allow_html=True)
                
                if result.get("ip_addresses"):
                    st.markdown("**IP Addresses:**")
                    for ip in result["ip_addresses"]:
                        st.code(ip)
                
                st.markdown(f"**Lookup Time:** {result.get('lookup_time_ms', '—')} ms")
                st.markdown(f"**Timestamp:** {result.get('timestamp', '—')}")
                
                if result.get("error"):
                    st.error(f"Error: {result['error']}")
    else:
        st.info("No DNS lookup results yet. Enter a hostname.")

    if st.button("Clear DNS history", key="clear_dns"):
        st.session_state.dns_history = []
        st.rerun()

with tab4:
    st.markdown("### Advanced Port Scanner")
    
    # Scan mode selection
    scan_mode = st.radio("Scan Mode", ["Single Port", "Port Range", "Port Presets"], horizontal=True)
    
    col_port_host, col_port_btn = st.columns([3, 1])
    with col_port_host:
        port_host = st.text_input("Host or IP", placeholder="e.g. 8.8.8.8 or example.com", key="port_host")
    with col_port_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_port = st.button("Scan Ports", type="primary", key="run_port")
    
    # Options
    with st.expander("Scan Options", expanded=False):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            grab_banner = st.checkbox("Grab Service Banners", value=False, key="grab_banner")
        with col_opt2:
            timeout = st.number_input("Timeout (s)", min_value=1, max_value=10, value=3, key="port_timeout")
        with col_opt3:
            max_workers = st.number_input("Max Workers", min_value=1, max_value=100, value=50, key="port_workers")
    
    if scan_mode == "Single Port":
        col_port_input, col_port_scan = st.columns([2, 1])
        with col_port_input:
            custom_port = st.number_input("Port Number", min_value=1, max_value=65535, value=80, key="custom_port")
        with col_port_scan:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Scan Single", key="scan_single"):
                selected_port = custom_port
                run_port = True
    
    elif scan_mode == "Port Range":
        col_range1, col_range2 = st.columns(2)
        with col_range1:
            start_port = st.number_input("Start Port", min_value=1, max_value=65535, value=1, key="start_port")
        with col_range2:
            end_port = st.number_input("End Port", min_value=1, max_value=65535, value=1000, key="end_port")
        
        if st.button("Scan Range", key="scan_range"):
            run_port = True
    
    elif scan_mode == "Port Presets":
        st.markdown("**Quick Port Presets**")
        preset_cols = st.columns(4)
        preset_keys = list(PORT_PRESETS.keys())
        
        for i, preset_name in enumerate(preset_keys):
            if preset_cols[i % 4].button(preset_name, key=f"preset_{preset_name}"):
                selected_preset = PORT_PRESETS[preset_name]
                run_port = True
    
    if run_port and port_host:
        if scan_mode == "Single Port" and 'selected_port' in locals():
            with st.spinner(f"Checking port {selected_port} on {port_host}..."):
                result = check_port(port_host.strip(), selected_port, timeout, grab_banner)
            st.session_state.port_history.insert(0, result)
        
        elif scan_mode == "Port Range":
            with st.spinner(f"Scanning ports {start_port}-{end_port} on {port_host}..."):
                results = scan_port_range(port_host.strip(), start_port, end_port, timeout, max_workers)
            st.session_state.port_history.extend(results)
        
        elif scan_mode == "Port Presets" and 'selected_preset' in locals():
            with st.spinner(f"Scanning {len(selected_preset)} ports on {port_host}..."):
                results = batch_port_check(port_host.strip(), selected_preset, timeout, max_workers, grab_banner)
            st.session_state.port_history.extend(results)
    
    if st.session_state.port_history:
        # Display results with enhanced information
        df = pd.DataFrame(st.session_state.port_history)
        
        # Add color coding for status
        def color_status(val):
            if val == "open":
                return "background-color: #d4edda"
            elif val == "closed":
                return "background-color: #f8d7da"
            elif val == "timeout":
                return "background-color: #fff3cd"
            return ""
        
        styled_df = df.style.map(color_status, subset=["status"])
        st.dataframe(styled_df, hide_index=True, **stretch_kwargs())
        
        # Security assessment
        security_assessment = assess_port_security(st.session_state.port_history)
        
        st.markdown("### Security Assessment")
        col_sec1, col_sec2, col_sec3, col_sec4 = st.columns(4)
        col_sec1.metric("Open Ports", security_assessment["total_open_ports"])
        col_sec2.metric("High Risk", security_assessment["high_risk_count"], delta_color="inverse")
        col_sec3.metric("Medium Risk", security_assessment["medium_risk_count"], delta_color="off")
        col_sec4.metric("Security Score", f"{security_assessment['security_score']}/100")
        
        # Recommendations
        if security_assessment["recommendations"]:
            st.markdown("### Security Recommendations")
            for rec in security_assessment["recommendations"]:
                st.warning(rec)
        
        # High risk details
        if security_assessment["high_risk_details"]:
            st.markdown("### High-Risk Ports Details")
            for port in security_assessment["high_risk_details"]:
                st.error(f"🔴 Port {port['port']} ({port['service_name']}): {port['service_description']}")
        
        # Service breakdown
        st.markdown("### Service Breakdown")
        open_ports = [r for r in st.session_state.port_history if r.get("status") == "open"]
        if open_ports:
            services = {}
            for port in open_ports:
                service = port.get("service_name", "Unknown")
                services[service] = services.get(service, 0) + 1
            
            for service, count in sorted(services.items(), key=lambda x: x[1], reverse=True):
                st.info(f"📡 {service}: {count} port(s)")
        
        # Export options
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("Export port scan (JSON)", key="export_port_json"):
                json_str = json.dumps(st.session_state.port_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="port_scan.json",
                    mime="application/json"
                )
        with col_export2:
            if st.button("Export security report (JSON)", key="export_security"):
                json_str = json.dumps(security_assessment, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="security_report.json",
                    mime="application/json"
                )
    else:
        st.info("No port scan results yet. Enter a host and select scan options.")

    if st.button("Clear port history", key="clear_port"):
        st.session_state.port_history = []
        st.rerun()

with tab5:
    st.markdown("### Enhanced Batch Monitor")
    st.markdown("Monitor multiple hosts simultaneously with parallel execution and timing analysis.")
    
    # Initialize thresholds
    if "latency_threshold" not in st.session_state:
        st.session_state.latency_threshold = 100
    if "packet_loss_threshold" not in st.session_state:
        st.session_state.packet_loss_threshold = 5
    if "batch_timing_history" not in st.session_state:
        st.session_state.batch_timing_history = []
    
    # Alert thresholds configuration
    with st.expander("Alert Thresholds", expanded=False):
        col_thresh1, col_thresh2 = st.columns(2)
        with col_thresh1:
            st.number_input("Latency Threshold (ms)", min_value=1, max_value=10000, key="latency_threshold")
        with col_thresh2:
            st.number_input("Packet Loss Threshold (%)", min_value=0, max_value=100, key="packet_loss_threshold")
    
    # Scan configuration
    with st.expander("Scan Configuration", expanded=False):
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            timeout = st.number_input("Timeout (s)", min_value=1, max_value=10, value=2, key="batch_timeout")
        with col_config2:
            max_workers = st.number_input("Max Workers", min_value=1, max_value=100, value=10, key="batch_workers")
    
    # Auto-refresh configuration
    with st.expander("Auto-Refresh Settings", expanded=False):
        col_refresh1, col_refresh2 = st.columns(2)
        with col_refresh1:
            auto_refresh_interval = st.number_input("Refresh Interval (seconds)", min_value=5, max_value=300, value=30, key="refresh_interval")
        with col_refresh2:
            st.markdown("<br>", unsafe_allow_html=True)
            enable_auto_refresh = st.checkbox("Enable Auto-Refresh", key="enable_auto_refresh")
    if enable_auto_refresh and st_autorefresh is not None:
        st_autorefresh(interval=auto_refresh_interval * 1000, key="auto_refresh_timer")
    
    col_batch_input, col_batch_btn = st.columns([3, 1])
    with col_batch_input:
        batch_hosts = st.text_area(
            "Hosts (one per line)", 
            placeholder="8.8.8.8\n1.1.1.1\ngoogle.com\ncloudflare.com",
            height=100,
            key="batch_hosts"
        )
        host_count = len([h for h in batch_hosts.split("\n") if h.strip()]) if batch_hosts else 0
        if host_count > 8:
            st.warning(
                f"You have entered {host_count} hosts. To keep the page responsive, the batch scan will be limited to the first 8 hosts. "
                "Use fewer hosts or increase the timeout for larger scans."
            )
    with col_batch_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_batch = st.button("Run Batch", type="primary", key="run_batch")

    st.markdown("**Quick Batch Presets**")
    preset_cols = st.columns(4)
    if preset_cols[0].button("DNS Servers", key="preset_dns"):
        batch_hosts = "8.8.8.8\n1.1.1.1\n9.9.9.9\n208.67.222.222"
        run_batch = True
    if preset_cols[1].button("Cloud Providers", key="preset_cloud"):
        batch_hosts = "google.com\namazon.com\nmicrosoft.com\ncloudflare.com"
        run_batch = True
    if preset_cols[2].button("Social Media", key="preset_social"):
        batch_hosts = "facebook.com\ntwitter.com\nlinkedin.com\ninstagram.com"
        run_batch = True
    if preset_cols[3].button("Network Infrastructure", key="preset_network"):
        batch_hosts = "8.8.8.8\n1.1.1.1\ncloudflare.com\nfast.com"
        run_batch = True

    if run_batch and batch_hosts:
        hosts = [h.strip() for h in batch_hosts.split("\n") if h.strip()]
        max_batch_hosts = 8
        if len(hosts) > max_batch_hosts:
            st.warning(f"Limiting batch scan to the first {max_batch_hosts} hosts to keep the app responsive.")
            hosts = hosts[:max_batch_hosts]

        results = []
        timing = None
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_result = None

        with st.spinner(f"Pinging {len(hosts)} hosts..."):
            try:
                start_time = time.time()
                for index, host in enumerate(hosts, start=1):
                    status_text.text(f"Pinging {host} ({index}/{len(hosts)})...")
                    result = real_ping(host, timeout=timeout)
                    result["scan_time"] = time.time() - start_time
                    results.append(result)
                    progress_bar.progress(int(index / len(hosts) * 100))

                total_time = time.time() - start_time
                successful_results = [r for r in results if r.get("status") == "up"]
                latencies = [r.get("latency_ms", 0) for r in successful_results if r.get("latency_ms") is not None]
                timing = {
                    "total_time_seconds": round(total_time, 3),
                    "total_hosts": len(hosts),
                    "successful_hosts": len(successful_results),
                    "failed_hosts": len(hosts) - len(successful_results),
                    "hosts_per_second": round(len(hosts) / total_time, 2) if total_time > 0 else 0,
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
                    "min_latency_ms": round(min(latencies), 2) if latencies else 0,
                    "max_latency_ms": round(max(latencies), 2) if latencies else 0,
                }
                batch_result = {"results": results, "timing": timing}
            except Exception as exc:
                st.error(f"Batch scan failed: {exc}")
            finally:
                status_text.text("Batch scan complete.")

        if batch_result is not None:
            results = batch_result["results"]
            timing = batch_result["timing"]

            st.session_state.batch_results = results
            st.session_state.batch_timing_history.insert(0, timing)

            # Check alerts
            for result in results:
                alert_info = check_alert_thresholds(result, st.session_state.latency_threshold, st.session_state.packet_loss_threshold)
                if alert_info["has_alerts"]:
                    st.session_state.alert_history.insert(0, alert_info)

            # Display timing summary
            st.success(f"✅ Scan completed in {timing['total_time_seconds']:.3f}s ({timing['hosts_per_second']:.2f} hosts/sec)")

    # Extract results for display (handle both old and new formats)
    results = None
    if st.session_state.batch_results:
        batch_data = st.session_state.batch_results
        if isinstance(batch_data, dict) and "results" in batch_data:
            results = batch_data["results"]
        else:
            results = batch_data

    if results:
        # Timing summary
        if st.session_state.batch_timing_history:
            latest_timing = st.session_state.batch_timing_history[0]
            st.markdown("### Scan Performance")
            col_time1, col_time2, col_time3, col_time4 = st.columns(4)
            col_time1.metric("Total Time", f"{latest_timing['total_time_seconds']:.3f}s")
            col_time2.metric("Hosts/Sec", f"{latest_timing['hosts_per_second']:.2f}")
            col_time3.metric("Avg Latency", f"{latest_timing['avg_latency_ms']:.2f}ms")
            col_time4.metric("Success Rate", f"{(latest_timing['successful_hosts']/latest_timing['total_hosts']*100):.1f}%")
        
        # Results display
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True, **stretch_kwargs())

        # Show chart
        chart_fig = create_multi_host_chart(results)
        if chart_fig is not None:
            st.plotly_chart(chart_fig, **stretch_kwargs())

        # Summary
        up_count = len([r for r in results if r.get("status") == "up"])
        down_count = len([r for r in results if r.get("status") == "down"])
        error_count = len([r for r in results if r.get("status") == "error"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Up", up_count, delta_color="normal")
        col2.metric("Down", down_count, delta_color="inverse")
        col3.metric("Errors", error_count, delta_color="off")

        # Show alerts
        if st.session_state.alert_history:
            st.markdown("### Recent Alerts")
            for alert in st.session_state.alert_history[:5]:  # Show last 5 alerts
                severity = alert.get("severity", "info")
                if severity == "critical":
                    st.error(f"🔴 {alert.get('host')}")
                elif severity == "warning":
                    st.warning(f"🟡 {alert.get('host')}")
                else:
                    st.info(f"🔵 {alert.get('host')}")
                
                for alert_detail in alert.get("alerts", []):
                    st.caption(f"• {alert_detail.get('message')}")

        # Export
        col_export1, col_export2, col_export3 = st.columns(3)
        with col_export1:
            if st.button("Export batch results (JSON)", key="export_batch"):
                # Export results only for consistency
                json_str = json.dumps(results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="batch_results.json",
                    mime="application/json"
                )
        with col_export2:
            if st.button("Export timing data (JSON)", key="export_timing"):
                json_str = json.dumps(st.session_state.batch_timing_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="batch_timing.json",
                    mime="application/json"
                )
        with col_export3:
            if st.button("Export alerts (JSON)", key="export_alerts"):
                json_str = json.dumps(st.session_state.alert_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="alerts.json",
                    mime="application/json"
                )
    else:
        st.info("No batch results yet. Enter hosts to monitor.")

with tab6:
    st.markdown("### Traceroute")
    st.markdown("Trace the route packets take to reach a destination.")
    
    col_trace_input, col_trace_btn = st.columns([3, 1])
    with col_trace_input:
        trace_host = st.text_input("Host or IP", placeholder="e.g. google.com", key="trace_host")
    with col_trace_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_trace = st.button("Run Traceroute", type="primary", key="run_trace")

    col_hops, col_timeout = st.columns(2)
    with col_hops:
        max_hops = st.number_input("Max Hops", min_value=1, max_value=64, value=30, key="max_hops")
    with col_timeout:
        trace_timeout = st.number_input("Timeout (s)", min_value=1, max_value=10, value=2, key="trace_timeout")

    if run_trace and trace_host:
        with st.spinner(f"Tracing route to {trace_host}..."):
            results = traceroute(trace_host.strip(), max_hops=max_hops, timeout=trace_timeout)
        st.session_state.traceroute_history.insert(0, {"host": trace_host, "hops": results})

    if st.session_state.traceroute_history:
        for trace in st.session_state.traceroute_history:
            with st.expander(f"Traceroute to {trace['host']}", expanded=True):
                hops_df = pd.DataFrame(trace['hops'])
                st.dataframe(hops_df, hide_index=True, **stretch_kwargs())
    else:
        st.info("No traceroute results yet. Enter a host to trace.")

    st.markdown(
        """
        **Note:** This is a simplified traceroute implementation. For full functionality with hop-by-hop
        IP resolution, consider using the `scapy` library (requires admin/root privileges).
        """
    )

with tab7:
    st.markdown("### Analysis Dashboard")
    st.markdown("Comprehensive network performance analysis with insights and recommendations.")
    
    # Generate analyses
    ping_analysis = analyze_ping_history(st.session_state.ping_history)
    http_analysis = analyze_http_history(st.session_state.http_history)
    dns_analysis = analyze_dns_history(st.session_state.dns_history)
    port_security = assess_port_security(st.session_state.port_history)
    
    # Calculate overall health score
    network_health = calculate_network_health_score(ping_analysis, http_analysis, dns_analysis, port_security)
    
    # Overall Health Score
    st.markdown("#### Overall Network Health")
    col_health1, col_health2, col_health3 = st.columns(3)
    col_health1.metric("Health Score", f"{network_health['overall_score']}/100", 
                       delta=f"{network_health['status_emoji']} {network_health['overall_status'].upper()}")
    col_health2.metric("Component Scores", f"Ping: {network_health['component_scores']['ping'] or 'N/A'} | "
                                          f"HTTP: {network_health['component_scores']['http'] or 'N/A'} | "
                                          f"DNS: {network_health['component_scores']['dns'] or 'N/A'} | "
                                          f"Security: {network_health['component_scores']['security'] or 'N/A'}")
    col_health3.metric("Priority Actions", len(network_health['priority_actions']))
    
    # Priority Actions
    if network_health['priority_actions']:
        st.markdown("#### Priority Actions")
        for action in network_health['priority_actions']:
            st.error(action)
    
    # Detailed Analysis Sections
    col_analyze1, col_analyze2 = st.columns(2)
    
    with col_analyze1:
        st.markdown("#### ICMP Ping Analysis")
        if "error" not in ping_analysis:
            col_ping1, col_ping2, col_ping3 = st.columns(3)
            col_ping1.metric("Total Pings", ping_analysis['total_pings'])
            col_ping2.metric("Success Rate", f"{ping_analysis['success_rate']:.1f}%")
            col_ping3.metric("Status", ping_analysis['status'].upper())
            
            if ping_analysis.get('latency'):
                st.markdown("**Latency Metrics:**")
                col_lat1, col_lat2, col_lat3 = st.columns(3)
                col_lat1.metric("Avg", f"{ping_analysis['latency']['avg_ms']}ms")
                col_lat2.metric("Min", f"{ping_analysis['latency']['min_ms']}ms")
                col_lat3.metric("Max", f"{ping_analysis['latency']['max_ms']}ms")
                
                col_lat4, col_lat5 = st.columns(2)
                col_lat4.metric("Std Dev", f"{ping_analysis['latency']['std_dev_ms']}ms")
                col_lat5.metric("Jitter", f"{ping_analysis['latency']['jitter_ms']}ms")
            
            if ping_analysis['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in ping_analysis['recommendations']:
                    st.info(rec)
        else:
            st.warning(ping_analysis['error'])
    
    with col_analyze2:
        st.markdown("#### HTTP Health Analysis")
        if "error" not in http_analysis:
            col_http1, col_http2, col_http3 = st.columns(3)
            col_http1.metric("Total Checks", http_analysis['total_checks'])
            col_http2.metric("Success Rate", f"{http_analysis['success_rate']:.1f}%")
            col_http3.metric("Status", http_analysis['status'].upper())
            
            if http_analysis.get('response_time'):
                st.markdown("**Response Time Metrics:**")
                col_resp1, col_resp2, col_resp3 = st.columns(3)
                col_resp1.metric("Avg", f"{http_analysis['response_time']['avg_ms']}ms")
                col_resp2.metric("Min", f"{http_analysis['response_time']['min_ms']}ms")
                col_resp3.metric("Max", f"{http_analysis['response_time']['max_ms']}ms")
            
            if http_analysis.get('status_codes'):
                st.markdown("**Status Code Distribution:**")
                for code, count in http_analysis['status_codes'].items():
                    st.caption(f"HTTP {code}: {count} occurrences")
            
            if http_analysis['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in http_analysis['recommendations']:
                    st.info(rec)
        else:
            st.warning(http_analysis['error'])
    
    # DNS and Security Analysis
    col_analyze3, col_analyze4 = st.columns(2)
    
    with col_analyze3:
        st.markdown("#### DNS Analysis")
        if "error" not in dns_analysis:
            col_dns1, col_dns2, col_dns3 = st.columns(3)
            col_dns1.metric("Total Lookups", dns_analysis['total_lookups'])
            col_dns2.metric("Success Rate", f"{dns_analysis['success_rate']:.1f}%")
            col_dns3.metric("Status", dns_analysis['status'].upper())
            
            if dns_analysis.get('lookup_time'):
                st.markdown("**Lookup Time Metrics:**")
                col_lookup1, col_lookup2 = st.columns(2)
                col_lookup1.metric("Avg", f"{dns_analysis['lookup_time']['avg_ms']}ms")
                col_lookup2.metric("Median", f"{dns_analysis['lookup_time']['median_ms']}ms")
            
            if dns_analysis['recommendations']:
                st.markdown("**Recommendations:**")
                for rec in dns_analysis['recommendations']:
                    st.info(rec)
        else:
            st.warning(dns_analysis['error'])
    
    with col_analyze4:
        st.markdown("#### Port Security Analysis")
        col_sec1, col_sec2, col_sec3 = st.columns(3)
        col_sec1.metric("Open Ports", port_security['total_open_ports'])
        col_sec2.metric("High Risk", port_security['high_risk_count'], delta_color="inverse")
        col_sec3.metric("Security Score", f"{port_security['security_score']}/100")
        
        if port_security['recommendations']:
            st.markdown("**Security Recommendations:**")
            for rec in port_security['recommendations']:
                st.warning(rec)
    
    # Batch Comparison
    if st.session_state.batch_results:
        st.markdown("#### Batch Host Comparison")
        # Handle both old format (list) and new format (dict with results)
        batch_data = st.session_state.batch_results
        if isinstance(batch_data, dict) and "results" in batch_data:
            batch_data = batch_data["results"]
        batch_comparison = compare_hosts(batch_data)
        
        col_batch1, col_batch2, col_batch3, col_batch4 = st.columns(4)
        col_batch1.metric("Total Hosts", batch_comparison['total_hosts'])
        col_batch2.metric("Success Rate", f"{batch_comparison['success_rate']}%")
        col_batch3.metric("Avg Latency", f"{batch_comparison['average_latency_ms']}ms")
        col_batch4.metric("Failed", batch_comparison['failed_hosts'], delta_color="inverse")
        
        if batch_comparison.get('best_performer'):
            st.success(f"🏆 Best Performer: {batch_comparison['best_performer']['host']} ({batch_comparison['best_performer']['latency_ms']}ms)")
        if batch_comparison.get('worst_performer'):
            st.error(f"⚠️ Worst Performer: {batch_comparison['worst_performer']['host']} ({batch_comparison['worst_performer']['latency_ms']}ms)")
        
        if batch_comparison.get('latency_ranking'):
            st.markdown("**Latency Ranking:**")
            ranking_df = pd.DataFrame(batch_comparison['latency_ranking'])
            st.dataframe(ranking_df, hide_index=True, **stretch_kwargs())
    
    # All Recommendations Summary
    if network_health['all_recommendations']:
        st.markdown("#### All Recommendations Summary")
        with st.expander("View All Recommendations", expanded=False):
            for rec in network_health['all_recommendations']:
                st.caption(f"• {rec}")
    
    # Export Analysis
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("Export Full Analysis (JSON)", key="export_analysis"):
            # Handle both old and new batch results format
            batch_data = st.session_state.batch_results
            if isinstance(batch_data, dict) and "results" in batch_data:
                batch_comparison = compare_hosts(batch_data["results"])
            else:
                batch_comparison = compare_hosts(batch_data) if batch_data else None
            
            full_analysis = {
                "network_health": network_health,
                "ping_analysis": ping_analysis,
                "http_analysis": http_analysis,
                "dns_analysis": dns_analysis,
                "port_security": port_security,
                "batch_comparison": batch_comparison,
            }
            json_str = json.dumps(full_analysis, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="network_analysis.json",
                mime="application/json"
            )
    with col_export2:
        pass
