import pandas as pd

from modules.ai_ops import build_incident_report, build_operational_summary, build_predictive_signal
from modules.anomaly_detector import AnomalyDetector
from modules.data_sources import SimulatedDataSource
from modules.diagnostics_bridge import explain_alert
from modules.remediation import RemediationEngine, RemediationActionType
from modules.reports import ReportGenerator


def test_anomaly_detector_detects_outlier():
    detector = AnomalyDetector(contamination=0.1)
    # 24h of simulated traffic (288 points, ~2% anomalies) — enough to train
    # the RF + GB ensemble and exercise single-point prediction end to end.
    history = SimulatedDataSource(seed=42).get_traffic_history(hours=24)
    detector.fit(history)

    extreme = detector.predict({"bandwidth_mbps": 5000.0, "latency_ms": 900.0, "packet_loss_pct": 80.0})
    assert bool(extreme["is_anomaly"]) is True

    normal = detector.predict({"bandwidth_mbps": 50.0, "latency_ms": 20.0, "packet_loss_pct": 0.5})
    assert bool(normal["is_anomaly"]) is False


def test_explain_alert_returns_actionable_text(monkeypatch):
    def fake_get_llm_response(messages, temperature=0.7, stream=False, api_key=None, model=None):
        return "Likely cause: congestion. Steps: 1) inspect queue."

    monkeypatch.setattr("modules.diagnostics_bridge.get_llm_response", fake_get_llm_response)
    alert = {"level": "warning", "device": "Router-Core-01", "metric": "latency_ms", "value": 120, "message": "Latency spike"}
    explanation = explain_alert(alert)
    assert "Likely cause" in explanation


def test_build_operational_summary_produces_actionable_plan():
    devices_df = pd.DataFrame([
        {"name": "Router-Core-01", "status": "up", "cpu_usage": 88, "latency_ms": 140, "packet_loss_pct": 2.5},
    ])
    traffic_df = pd.DataFrame([
        {"bandwidth_mbps": 120, "latency_ms": 80},
        {"bandwidth_mbps": 180, "latency_ms": 110},
        {"bandwidth_mbps": 220, "latency_ms": 145},
    ])
    result = build_operational_summary(
        {"device": "Router-Core-01", "metric": "latency_ms", "message": "High latency and packet loss detected"},
        logs="interface ethernet link flap detected",
        devices_df=devices_df,
        traffic_df=traffic_df,
    )
    assert result["severity"] == "critical"
    assert result["priority"] == "P1"
    assert result["impact"]
    assert result["next_steps"]
    assert result["operator_checks"]
    assert result["commands"]


def test_build_predictive_signal_returns_risk_summary():
    devices_df = pd.DataFrame([
        {"name": "Router-Core-01", "cpu_usage": 90, "latency_ms": 140, "packet_loss_pct": 2.3},
    ])
    traffic_df = pd.DataFrame([
        {"bandwidth_mbps": 100, "latency_ms": 80},
        {"bandwidth_mbps": 140, "latency_ms": 110},
        {"bandwidth_mbps": 220, "latency_ms": 145},
    ])
    signal = build_predictive_signal(devices_df, traffic_df)
    assert signal["status"] in {"elevated", "critical"}
    assert signal["summary"]


def test_build_incident_report_contains_actionable_sections():
    plan = {
        "severity": "critical",
        "priority": "P1",
        "impact": "User traffic is at risk",
        "recommended_action": "Inspect the interface",
        "summary": "Latency is rising",
        "likely_cause": "Congestion",
        "commands": [{"platform": "Cisco", "command": "show interfaces", "purpose": "Inspect interfaces"}],
        "next_steps": ["Check the interface"],
        "operator_checks": ["Confirm the device is forwarding"],
    }
    report = build_incident_report(plan, root_cause="Routing adjacency issue")
    assert "Incident Report" in report
    assert "Recommended Actions" in report
    assert "Cisco" in report


def test_remediation_engine_lifecycle():
    engine = RemediationEngine()
    incident = engine.create_incident({"id": "1", "device": "Router-Core-01", "metric": "cpu_usage", "message": "High CPU"})
    engine.diagnose_incident(incident.incident_id, "CPU saturation")
    engine.suggest_action(incident.incident_id, RemediationActionType.FLAG_FOR_REVIEW, "Inspect the device")
    engine.submit_for_approval(incident.incident_id)
    assert incident.state.name in {"PENDING_APPROVAL", "SUGGESTED"}


def test_report_generator_falls_back_cleanly(monkeypatch):
    class FakeDS:
        def get_traffic_history(self, hours=24):
            return pd.DataFrame([{"bandwidth_mbps": 50, "latency_ms": 20, "packet_loss_pct": 0.1}])

        def get_devices(self):
            return pd.DataFrame([{"name": "Router", "status": "up", "cpu_usage": 40, "latency_ms": 15, "packet_loss_pct": 0.1}])

        def get_host_metrics(self):
            return {"cpu_percent": 40, "memory_percent": 60, "network_throughput_mbps": 80}

    monkeypatch.setattr("modules.reports.get_data_source", lambda: FakeDS())
    monkeypatch.setattr("modules.reports.get_remediation_engine", lambda: type("R", (), {"get_pending_approvals": lambda self: [], "get_incident_history": lambda self, limit=10: []})())
    report = ReportGenerator().generate_report(time_window_hours=6, include_forecasts=False, include_remediation=False)
    assert "Network Health Report" in report
