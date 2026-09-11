"""Operational AI helpers for fault diagnosis, command generation, and predictive insights."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def build_troubleshooting_plan(alert: dict[str, Any], logs: str | None = None) -> dict[str, Any]:
    """Create a structured troubleshooting plan with vendor-ready commands."""
    metric = str(alert.get("metric") or "").lower()
    message = str(alert.get("message") or "").lower()
    device = str(alert.get("device") or "device")
    combined = f"{metric} {message}"

    if any(token in combined for token in ["cpu", "utilization", "resource"]):
        issue_type = "high_cpu"
        likely_cause = "The device is running near or above its CPU threshold, which can cause control-plane slowness or service degradation."
    elif any(token in combined for token in ["latency", "delay", "rtt"]):
        issue_type = "latency"
        likely_cause = "Latency is elevated, often caused by congestion, path issues, or a saturated uplink."
    elif any(token in combined for token in ["packet loss", "loss", "drops"]):
        issue_type = "packet_loss"
        likely_cause = "Packet loss is occurring, which usually points to interface errors, congestion, or a faulty link segment."
    elif any(token in combined for token in ["down", "unreachable", "status"]):
        issue_type = "device_down"
        likely_cause = "The device is unreachable, which may indicate a failed link, power issue, routing problem, or firewall block."
    elif any(token in combined for token in ["dns", "resolver"]):
        issue_type = "dns"
        likely_cause = "DNS resolution appears unreliable, often caused by bad resolver configuration or upstream connectivity issues."
    else:
        issue_type = "general"
        likely_cause = "The alert suggests a network health issue that should be validated against recent logs and interface counters."

    commands = _get_platform_commands(issue_type, device)
    if logs:
        log_analysis = analyze_logs(logs, device_name=device)
        likely_cause = f"{likely_cause} {log_analysis['root_cause']}"
        commands.extend(log_analysis.get("recommended_commands", []))

    return {
        "issue_type": issue_type,
        "likely_cause": likely_cause,
        "device": device,
        "commands": commands,
        "summary": _build_summary(issue_type, device, likely_cause),
    }


def analyze_logs(log_text: str, device_name: str | None = None) -> dict[str, Any]:
    """Analyze pasted logs to infer the most likely root cause and remediation steps."""
    if not log_text or not str(log_text).strip():
        return {
            "root_cause": "No log content provided.",
            "confidence": 0.0,
            "evidence": [],
            "recommended_commands": [],
        }

    text = str(log_text).strip().lower()
    evidence: list[str] = []
    root_cause = "Possible configuration or environmental issue; review device logs around the timestamps."
    confidence = 0.55

    if re.search(r"\b(ospf|bgp|neighbor|adjacency)\b", text):
        root_cause = "Routing adjacency or reachability failure is the most likely cause."
        confidence = 0.82
        evidence.append("Routing protocol adjacency keywords detected")
    elif re.search(r"\b(dhcp|lease|pool|address conflict)\b", text):
        root_cause = "Address assignment or DHCP exhaustion is the most likely cause."
        confidence = 0.8
        evidence.append("DHCP or address assignment keywords detected")
    elif re.search(r"\b(dns|resolver|named|server failure)\b", text):
        root_cause = "DNS resolution failure is the most likely cause."
        confidence = 0.8
        evidence.append("DNS resolver keywords detected")
    elif re.search(r"\b(crc|collisions|duplex|interface|link down|link flap)\b", text):
        root_cause = "Interface or physical link instability is the most likely cause."
        confidence = 0.79
        evidence.append("Interface and link error keywords detected")
    elif re.search(r"\b(cpu|high utilization|resource)\b", text):
        root_cause = "Resource saturation is the most likely cause."
        confidence = 0.74
        evidence.append("Resource saturation keywords detected")
    elif re.search(r"\b(packet loss|drops|queue|congestion)\b", text):
        root_cause = "Congestion or packet drops are the most likely cause."
        confidence = 0.72
        evidence.append("Packet-loss or congestion keywords detected")

    if device_name:
        root_cause = f"{root_cause} Device: {device_name}."

    commands = _get_platform_commands("general", device_name or "device")
    return {
        "root_cause": root_cause,
        "confidence": round(confidence, 2),
        "evidence": evidence,
        "recommended_commands": commands,
    }


def build_predictive_insights(devices_df: pd.DataFrame, traffic_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Create a compact set of predictive issues from current telemetry trends."""
    insights: list[dict[str, Any]] = []
    if devices_df.empty:
        return insights

    if "latency_ms" in devices_df.columns and "cpu_usage" in devices_df.columns:
        high_latency = devices_df[devices_df["latency_ms"] >= 120]
        if not high_latency.empty:
            device = str(high_latency.iloc[0].get("name") or high_latency.iloc[0].get("device") or "device")
            insights.append(
                {
                    "severity": "warning",
                    "title": "Latency trend rising",
                    "summary": f"{device} is already above a high-latency threshold and may degrade user traffic if the path remains congested.",
                    "recommended_action": "Review interface counters and upstream path quality before the issue becomes widespread.",
                }
            )

        high_cpu = devices_df[devices_df["cpu_usage"] >= 85]
        if not high_cpu.empty:
            device = str(high_cpu.iloc[0].get("name") or high_cpu.iloc[0].get("device") or "device")
            insights.append(
                {
                    "severity": "warning",
                    "title": "CPU saturation risk",
                    "summary": f"{device} is approaching control-plane saturation and may fail under additional traffic bursts.",
                    "recommended_action": "Check for process spikes, route churn, or unnecessary services consuming CPU cycles.",
                }
            )

    if not traffic_df.empty and {"bandwidth_mbps", "latency_ms"}.issubset(traffic_df.columns):
        recent = traffic_df.tail(6).copy()
        if len(recent) >= 3:
            rate_change = recent["bandwidth_mbps"].iloc[-1] - recent["bandwidth_mbps"].iloc[0]
            latency_change = recent["latency_ms"].iloc[-1] - recent["latency_ms"].iloc[0]
            if rate_change > 50 and latency_change > 20:
                insights.append(
                    {
                        "severity": "critical",
                        "title": "Capacity pressure likely",
                        "summary": "Bandwidth is climbing while latency is increasing, which often precedes a broader congestion incident.",
                        "recommended_action": "Inspect the bottleneck path, review queued traffic, and scale or re-route capacity before the next peak.",
                    }
                )

    return insights


def build_predictive_signal(
    devices_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Create a predictive risk signal from current telemetry trends."""
    if devices_df is None or devices_df.empty:
        return {"status": "stable", "summary": "No telemetry data available for prediction.", "score": 0}

    device_row = _select_device_row(devices_df, device_name)
    score = 0
    summary_parts: list[str] = []

    if "cpu_usage" in devices_df.columns:
        cpu_value = float(device_row.get("cpu_usage", 0) or 0)
        if cpu_value >= 85:
            score += 35
            summary_parts.append("CPU is approaching saturation.")
    if "latency_ms" in devices_df.columns:
        latency_value = float(device_row.get("latency_ms", 0) or 0)
        if latency_value >= 120:
            score += 35
            summary_parts.append("Latency is already elevated.")
    if "packet_loss_pct" in devices_df.columns:
        loss_value = float(device_row.get("packet_loss_pct", 0) or 0)
        if loss_value >= 1.0:
            score += 20
            summary_parts.append("Packet loss is visible.")

    if traffic_df is not None and not traffic_df.empty and {"bandwidth_mbps", "latency_ms"}.issubset(traffic_df.columns):
        recent = traffic_df.tail(4).copy()
        if len(recent) >= 2:
            growth = recent["bandwidth_mbps"].iloc[-1] - recent["bandwidth_mbps"].iloc[0]
            latency_growth = recent["latency_ms"].iloc[-1] - recent["latency_ms"].iloc[0]
            if growth > 50 or latency_growth > 20:
                score += 20
                summary_parts.append("Traffic is rising while latency is worsening.")

    if score >= 70:
        status = "critical"
    elif score >= 40:
        status = "elevated"
    else:
        status = "stable"

    summary = " ".join(summary_parts) if summary_parts else "Telemetry remains within normal bounds."
    return {"status": status, "score": int(score), "summary": summary}


def build_incident_report(plan: dict[str, Any], root_cause: str | None = None) -> str:
    """Generate a plain-text incident report that can be copied or exported."""
    context = plan.get("context", {})
    lines = [
        "# Incident Report",
        "",
        f"- Incident ID: {context.get('id', 'Not assigned')}",
        f"- Device: {context.get('device', 'Not specified')}",
        f"- Platform: {context.get('platform', 'Not specified')}",
        f"- Location / Service: {context.get('location', 'Not specified')}",
        f"- Owner: {context.get('owner', 'Not assigned')}",
        f"- Business Impact Scope: {context.get('impact_scope', 'Not confirmed')}",
        f"- Severity: {plan.get('severity', 'warning').upper()}",
        f"- Priority: {plan.get('priority', 'P2')}",
        f"- Impact: {plan.get('impact', 'Service degradation possible')}",
        f"- Likely Cause: {plan.get('likely_cause', 'Investigate the issue')}",
        f"- Root Cause: {root_cause or 'Awaiting detailed evidence'}",
        "",
        "## Recommended Actions",
    ]
    for step in plan.get("next_steps", []):
        lines.append(f"- {step}")
    lines.extend(["", "## Operator Checks"])
    for check in plan.get("operator_checks", []):
        lines.append(f"- {check}")
    lines.extend(["", "## Commands"])
    for command in plan.get("commands", []):
        lines.append(f"- {command.get('platform', 'Platform')}: {command.get('command', '')}")
    return "\n".join(lines)


def build_operational_summary(
    alert: dict[str, Any],
    logs: str | None = None,
    devices_df: pd.DataFrame | None = None,
    traffic_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a richer operations summary combining diagnosis, telemetry, and commands."""
    plan = build_troubleshooting_plan(alert, logs=logs)
    insights = []

    if devices_df is not None and not devices_df.empty:
        device_row = _select_device_row(devices_df, str(alert.get("device") or ""))
        if "cpu_usage" in devices_df.columns and device_row.get("cpu_usage", 0) >= 85:
            insights.append("High CPU is present on the primary device and should be investigated immediately.")
        if "latency_ms" in devices_df.columns and device_row.get("latency_ms", 0) >= 120:
            insights.append("Latency is already elevated on the primary device, indicating a worsening path or interface issue.")
        if "packet_loss_pct" in devices_df.columns and device_row.get("packet_loss_pct", 0) >= 1.0:
            insights.append("Packet loss is visible and may be driving the symptom cluster.")

    if traffic_df is not None and not traffic_df.empty and {"bandwidth_mbps", "latency_ms"}.issubset(traffic_df.columns):
        recent = traffic_df.tail(4).copy()
        if len(recent) >= 2:
            growth = recent["bandwidth_mbps"].iloc[-1] - recent["bandwidth_mbps"].iloc[0]
            latency_growth = recent["latency_ms"].iloc[-1] - recent["latency_ms"].iloc[0]
            if growth > 50 or latency_growth > 20:
                insights.append("Traffic is rising while latency is worsening, which often signals capacity pressure.")

    severity = "warning"
    priority = "P2"
    impact = "Service degradation is likely but not yet widespread."
    if any(token in str(alert.get("message") or "").lower() for token in ["critical", "down", "packet loss", "latency", "loss"]):
        severity = "critical"
        priority = "P1"
        impact = "User-facing latency or reachability issues are likely and require rapid mitigation."

    recommended_action = "Isolate the affected path and confirm the interface state before changing policy or routes."
    if plan["issue_type"] == "high_cpu":
        recommended_action = "Reduce the workload or investigate the highest-usage process before the device loses control-plane capacity."
    elif plan["issue_type"] == "latency":
        recommended_action = "Check interface utilization, queue depth, and the upstream path for congestion or errors."
    elif plan["issue_type"] == "packet_loss":
        recommended_action = "Inspect the link for errors, collisions, or a failing cable/port and replace it if needed."
    elif plan["issue_type"] == "device_down":
        recommended_action = "Verify reachability, power, and the interface state before attempting any rollback or config change."
    elif plan["issue_type"] == "dns":
        recommended_action = "Validate DNS resolver reachability and upstream server health before disturbing the routing design."

    next_steps = [
        "Confirm whether the symptom is isolated to one device or is spreading across the path.",
        "Review interface counters, link state, and recent routing or DHCP events.",
        "Apply the first mitigation step and monitor the effect for 5-10 minutes.",
        "Escalate to the on-call networking team if the issue persists or worsens.",
    ]
    operator_checks = [
        "Check whether the device is still forwarding traffic normally.",
        "Compare the alert window with the pasted logs for a matching pattern.",
        "Verify whether other devices in the same segment show the same symptom.",
    ]
    if plan.get("commands"):
        next_steps.append("Run the recommended vendor commands and compare the output with the suspected cause.")

    evidence_tags = [
        plan["issue_type"].replace("_", " "),
        "telemetry-reviewed",
    ]
    if logs:
        evidence_tags.append("log-evidence")

    return {
        "severity": severity,
        "priority": priority,
        "impact": impact,
        "recommended_action": recommended_action,
        "summary": plan["summary"],
        "likely_cause": plan["likely_cause"],
        "commands": plan["commands"],
        "next_steps": next_steps,
        "operator_checks": operator_checks,
        "telemetry_notes": insights,
        "evidence_tags": evidence_tags,
        "context": {
            "id": alert.get("id", ""),
            "device": alert.get("device", ""),
            "platform": alert.get("platform", ""),
            "location": alert.get("location", ""),
            "owner": alert.get("owner", ""),
            "impact_scope": alert.get("impact_scope", ""),
        },
    }


def _get_platform_commands(issue_type: str, device: str) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    device_label = device or "device"

    if issue_type == "high_cpu":
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": f"show processes cpu sorted | include {device_label}",
                    "purpose": "Inspect CPU usage on the affected device.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/system resource print",
                    "purpose": "Review CPU and memory usage from RouterOS.",
                },
            ]
        )
    elif issue_type == "latency":
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": "show interfaces | include line protocol|rate",
                    "purpose": "Verify interface utilization and errors on the path.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/interface ethernet monitor [find]",
                    "purpose": "Check link status and errors on Ethernet interfaces.",
                },
            ]
        )
    elif issue_type == "packet_loss":
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": "show interfaces counters errors",
                    "purpose": "Review CRC, collisions, and input/output errors.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/interface ethernet monitor [find]",
                    "purpose": "Inspect link drops and errors on MikroTik interfaces.",
                },
            ]
        )
    elif issue_type == "device_down":
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": "show ip interface brief",
                    "purpose": "Check whether the interface is up and the device is reachable.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/interface print",
                    "purpose": "Confirm the relevant interface state and link status.",
                },
            ]
        )
    elif issue_type == "dns":
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": "show ip dns view",
                    "purpose": "Inspect the DNS resolver configuration.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/ip dns print",
                    "purpose": "Review the DNS server configuration on RouterOS.",
                },
            ]
        )
    else:
        commands.extend(
            [
                {
                    "platform": "Cisco",
                    "command": "show logging", 
                    "purpose": "Gather recent device events and errors.",
                },
                {
                    "platform": "MikroTik",
                    "command": "/log print", 
                    "purpose": "Review RouterOS system logs for the failure window.",
                },
            ]
        )

    return commands


def _select_device_row(devices_df: pd.DataFrame, device_name: str | None) -> pd.Series:
    """Return telemetry for the requested device, with a safe first-row fallback."""
    if device_name and "name" in devices_df.columns:
        normalized_name = str(device_name).strip().casefold()
        matches = devices_df[
            devices_df["name"].astype(str).str.strip().str.casefold() == normalized_name
        ]
        if not matches.empty:
            return matches.iloc[0]
    return devices_df.iloc[0]


def _build_summary(issue_type: str, device: str, likely_cause: str) -> str:
    return f"{device} is showing {issue_type.replace('_', ' ')} symptoms. {likely_cause}"
