"""AI-generated health and incident reports."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import streamlit as st

from modules.data_sources import get_data_source
from modules.alerts import generate_alerts
from modules.llm_client import get_llm_response
from modules.remediation import get_remediation_engine


class ReportGenerator:
    """Generate AI-powered network health and incident reports."""

    def __init__(self):
        self.data_source = get_data_source()

    def generate_report(
        self,
        time_window_hours: int = 24,
        include_forecasts: bool = True,
        include_remediation: bool = True,
    ) -> str:
        """
        Generate a comprehensive network health report.
        
        Args:
            time_window_hours: Time window for the report
            include_forecasts: Include forecasting analysis
            include_remediation: Include remediation status
            
        Returns:
            Markdown-formatted report
        """
        # Gather data
        traffic_history = self.data_source.get_traffic_history(time_window_hours)
        devices = self.data_source.get_devices()
        host_metrics = self.data_source.get_host_metrics()
        alerts = generate_alerts(devices)
        
        # Calculate summary statistics
        avg_bandwidth = traffic_history['bandwidth_mbps'].mean()
        max_bandwidth = traffic_history['bandwidth_mbps'].max()
        avg_latency = traffic_history['latency_ms'].mean()
        max_latency = traffic_history['latency_ms'].max()
        
        device_up = len(devices[devices['status'] == 'up'])
        device_total = len(devices)
        
        # Get remediation status
        remediation_engine = get_remediation_engine()
        pending_approvals = remediation_engine.get_pending_approvals()
        recent_incidents = remediation_engine.get_incident_history(limit=10)
        
        # Build structured summary for LLM
        summary = {
            "time_window": f"{time_window_hours} hours",
            "report_generated": datetime.now().isoformat(),
            "network_health": {
                "devices_up": device_up,
                "devices_total": device_total,
                "device_availability_pct": (device_up / device_total * 100) if device_total > 0 else 0,
            },
            "traffic_metrics": {
                "avg_bandwidth_mbps": round(avg_bandwidth, 2),
                "max_bandwidth_mbps": round(max_bandwidth, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "max_latency_ms": round(max_latency, 2),
            },
            "host_metrics": {
                "cpu_percent": round(host_metrics.get('cpu_percent', 0), 1),
                "memory_percent": round(host_metrics.get('memory_percent', 0), 1),
                "network_throughput_mbps": round(host_metrics.get('network_throughput_mbps', 0), 1),
            },
            "alerts": {
                "total_alerts": len(alerts),
                "critical_alerts": len([a for a in alerts if a['level'] == 'critical']),
                "warning_alerts": len([a for a in alerts if a['level'] == 'warning']),
                "alert_details": alerts[:5],  # Top 5 alerts
            },
            "remediation": {
                "pending_approvals": len(pending_approvals),
                "recent_incidents": len(recent_incidents),
                "incident_details": [
                    {
                        "device": inc.device,
                        "state": inc.state.value,
                        "issue_type": inc.issue_type,
                    }
                    for inc in recent_incidents[:5]
                ],
            } if include_remediation else None,
        }
        
        # Generate report using LLM
        report = self._generate_llm_report(summary, include_forecasts)
        
        return report

    def _generate_llm_report(self, summary: dict[str, Any], include_forecasts: bool) -> str:
        """Generate report text using LLM."""
        prompt = self._build_report_prompt(summary, include_forecasts)
        
        messages = [
            {
                "role": "system",
                "content": """You are a network operations report generator. 
Generate clear, professional markdown reports for network administrators.
Include sections: Executive Summary, Key Metrics, Alerts Analysis, 
and Recommendations. Use tables and bullet points for readability.
Be specific and actionable in recommendations.
IMPORTANT: You are ONLY allowed to answer questions related to: 
network management, network monitoring, network diagnostics, network security, 
network troubleshooting, network configuration, network protocols, 
network devices (routers, switches, firewalls), network performance, 
network alerts, and network infrastructure. 
If a question is completely outside this scope (e.g., cooking, sports, 
politics, entertainment, general knowledge not related to networking), 
politely refuse and state that you can only help with network-related topics."""
            },
            {"role": "user", "content": prompt},
        ]
        
        try:
            report = get_llm_response(messages, temperature=0.5)
        except Exception as e:
            # Fallback to template if LLM fails
            report = self._generate_fallback_report(summary)
        
        return report

    def _build_report_prompt(self, summary: dict[str, Any], include_forecasts: bool) -> str:
        """Build the prompt for LLM report generation."""
        prompt = f"""Generate a network health report based on the following data:

**Time Window:** {summary['time_window']}
**Report Generated:** {summary['report_generated']}

**Network Health:**
- Devices Up: {summary['network_health']['devices_up']}/{summary['network_health']['devices_total']}
- Availability: {summary['network_health']['device_availability_pct']:.1f}%

**Traffic Metrics:**
- Average Bandwidth: {summary['traffic_metrics']['avg_bandwidth_mbps']} Mbps
- Peak Bandwidth: {summary['traffic_metrics']['max_bandwidth_mbps']} Mbps
- Average Latency: {summary['traffic_metrics']['avg_latency_ms']} ms
- Peak Latency: {summary['traffic_metrics']['max_latency_ms']} ms

**Host Metrics:**
- CPU Usage: {summary['host_metrics']['cpu_percent']}%
- Memory Usage: {summary['host_metrics']['memory_percent']}%
- Network Throughput: {summary['host_metrics']['network_throughput_mbps']} Mbps

**Alerts:**
- Total Alerts: {summary['alerts']['total_alerts']}
- Critical: {summary['alerts']['critical_alerts']}
- Warnings: {summary['alerts']['warning_alerts']}
"""
        
        if summary['alerts']['alert_details']:
            prompt += "\n**Recent Alerts:**\n"
            for alert in summary['alerts']['alert_details']:
                prompt += f"- {alert['level'].upper()}: {alert['message']} (Device: {alert['device']})\n"
        
        if summary.get('remediation'):
            prompt += f"""
**Remediation Status:**
- Pending Approvals: {summary['remediation']['pending_approvals']}
- Recent Incidents: {summary['remediation']['recent_incidents']}
"""
            if summary['remediation']['incident_details']:
                prompt += "\n**Recent Incidents:**\n"
                for inc in summary['remediation']['incident_details']:
                    prompt += f"- {inc['device']}: {inc['issue_type']} ({inc['state']})\n"
        
        if include_forecasts:
            prompt += "\n**Note:** Include a brief section on capacity planning and potential future issues based on current trends.\n"
        
        prompt += """
Generate a professional markdown report with:
1. Executive Summary (2-3 sentences)
2. Key Metrics (table format)
3. Alerts Analysis (with severity breakdown)
4. Recommendations (3-5 actionable items)
5. Capacity Planning (if forecasts requested)
"""
        
        return prompt

    def _generate_fallback_report(self, summary: dict[str, Any]) -> str:
        """Generate a basic report without LLM."""
        report = f"""# Network Health Report

**Time Window:** {summary['time_window']}  
**Generated:** {summary['report_generated']}

## Executive Summary

Network availability is at {summary['network_health']['device_availability_pct']:.1f}% with {summary['alerts']['total_alerts']} active alerts. 
Traffic levels are within normal ranges with average bandwidth of {summary['traffic_metrics']['avg_bandwidth_mbps']:.1f} Mbps.

## Key Metrics

| Metric | Value |
|--------|-------|
| Device Availability | {summary['network_health']['device_availability_pct']:.1f}% |
| Devices Up | {summary['network_health']['devices_up']}/{summary['network_health']['devices_total']} |
| Average Bandwidth | {summary['traffic_metrics']['avg_bandwidth_mbps']:.1f} Mbps |
| Peak Bandwidth | {summary['traffic_metrics']['max_bandwidth_mbps']:.1f} Mbps |
| Average Latency | {summary['traffic_metrics']['avg_latency_ms']:.1f} ms |
| Peak Latency | {summary['traffic_metrics']['max_latency_ms']:.1f} ms |
| CPU Usage | {summary['host_metrics']['cpu_percent']:.1f}% |
| Memory Usage | {summary['host_metrics']['memory_percent']:.1f}% |

## Alerts Analysis

- **Total Alerts:** {summary['alerts']['total_alerts']}
- **Critical:** {summary['alerts']['critical_alerts']}
- **Warnings:** {summary['alerts']['warning_alerts']}

"""
        
        if summary['alerts']['alert_details']:
            report += "### Recent Alerts\n\n"
            for alert in summary['alerts']['alert_details']:
                report += f"- **{alert['level'].upper()}**: {alert['message']} (Device: {alert['device']})\n"
        
        if summary.get('remediation'):
            report += f"""
## Remediation Status

- **Pending Approvals:** {summary['remediation']['pending_approvals']}
- **Recent Incidents:** {summary['remediation']['recent_incidents']}
"""
        
        report += """
## Recommendations

1. Review critical alerts and prioritize remediation actions
2. Monitor devices with high latency for potential issues
3. Schedule regular capacity planning reviews
4. Ensure all pending remediation actions are reviewed
5. Document any manual interventions for future reference

---
*Report generated by AI Smart Bot for Network Management*
"""
        
        return report

    def generate_incident_report(self, incident_id: str) -> str:
        """Generate a detailed report for a specific incident."""
        remediation_engine = get_remediation_engine()
        incident = remediation_engine.incidents.get(incident_id)
        
        if not incident:
            return f"Incident {incident_id} not found."
        
        # Build incident summary
        summary = {
            "incident_id": incident.incident_id,
            "device": incident.device,
            "issue_type": incident.issue_type,
            "state": incident.state.value,
            "detected_at": incident.detected_at.isoformat(),
            "diagnosis": incident.diagnosis,
            "suggested_action": {
                "type": incident.suggested_action.action_type.value if incident.suggested_action else None,
                "description": incident.suggested_action.description if incident.suggested_action else None,
                "severity": incident.suggested_action.severity if incident.suggested_action else None,
            } if incident.suggested_action else None,
            "state_history": incident.state_history,
        }
        
        # Generate incident report using LLM
        prompt = f"""Generate a detailed incident report for the following network incident:

**Incident ID:** {summary['incident_id']}
**Device:** {summary['device']}
**Issue Type:** {summary['issue_type']}
**Current State:** {summary['state']}
**Detected At:** {summary['detected_at']}
**Diagnosis:** {summary['diagnosis']}
"""
        
        if summary['suggested_action']:
            prompt += f"""
**Suggested Action:** {summary['suggested_action']['type']}
**Description:** {summary['suggested_action']['description']}
**Severity:** {summary['suggested_action']['severity']}
"""
        
        prompt += """
Generate a markdown report with:
1. Incident Overview
2. Timeline of Events
3. Root Cause Analysis
4. Impact Assessment
5. Resolution Steps Taken
6. Lessons Learned/Prevention
"""
        
        messages = [
            {
                "role": "system",
                "content": """You are a network incident report generator. Create detailed, professional incident reports for post-incident analysis.
IMPORTANT: You are ONLY allowed to answer questions related to: 
network management, network monitoring, network diagnostics, network security, 
network troubleshooting, network configuration, network protocols, 
network devices (routers, switches, firewalls), network performance, 
network alerts, and network infrastructure. 
If a question is completely outside this scope (e.g., cooking, sports, 
politics, entertainment, general knowledge not related to networking), 
politely refuse and state that you can only help with network-related topics."""
            },
            {"role": "user", "content": prompt},
        ]
        
        try:
            report = get_llm_response(messages, temperature=0.5)
        except Exception as e:
            report = f"# Incident Report\n\nError generating report: {e}\n\n## Incident Details\n\n{summary}"
        
        return report


def get_report_generator() -> ReportGenerator:
    """Get or create the report generator singleton."""
    if "report_generator" not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    return st.session_state.report_generator
