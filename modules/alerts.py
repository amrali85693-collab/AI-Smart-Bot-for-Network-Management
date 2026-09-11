"""Threshold-based alert generation from device metrics."""

from __future__ import annotations

import pandas as pd

DEFAULT_CPU_WARNING = 85.0
DEFAULT_PACKET_LOSS_WARNING = 1.5
DEFAULT_LATENCY_WARNING_MS = 100.0


def generate_alerts(
    devices_df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> list[dict]:
    thresholds = thresholds or {}
    cpu_limit = float(thresholds.get("cpu_warning", DEFAULT_CPU_WARNING))
    loss_limit = float(thresholds.get("packet_loss_warning", DEFAULT_PACKET_LOSS_WARNING))
    latency_limit = float(thresholds.get("latency_warning_ms", DEFAULT_LATENCY_WARNING_MS))

    alerts: list[dict] = []

    for _, row in devices_df.iterrows():
        device = str(row.get("device") or row.get("name", "unknown"))

        if str(row.get("status", "")).lower() == "down":
            alerts.append(
                {
                    "level": "critical",
                    "message": f"{device} is unreachable (down).",
                    "device": device,
                    "metric": "status",
                    "value": 0.0,
                }
            )
            continue

        cpu = float(row.get("cpu_usage", 0) or 0)
        if cpu >= cpu_limit:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"{device} CPU usage is elevated at {cpu:.1f}%.",
                    "device": device,
                    "metric": "cpu_usage",
                    "value": cpu,
                }
            )

        packet_loss = float(row.get("packet_loss_pct", 0) or 0)
        if packet_loss >= loss_limit:
            alerts.append(
                {
                    "level": "warning",
                    "message": (
                        f"{device} packet loss is {packet_loss:.2f}% "
                        f"(threshold {loss_limit}%)."
                    ),
                    "device": device,
                    "metric": "packet_loss_pct",
                    "value": packet_loss,
                }
            )

        latency = row.get("latency_ms")
        if latency is not None and float(latency) >= latency_limit:
            latency_f = float(latency)
            alerts.append(
                {
                    "level": "warning",
                    "message": (
                        f"{device} latency is {latency_f:.1f} ms "
                        f"(threshold {latency_limit} ms)."
                    ),
                    "device": device,
                    "metric": "latency_ms",
                    "value": latency_f,
                }
            )

    if not alerts:
        alerts.append(
            {
                "level": "ok",
                "message": "All monitored devices are within thresholds.",
                "device": "all",
                "metric": "summary",
                "value": 0.0,
            }
        )

    return alerts


def alert_summary(alerts: list[dict]) -> dict[str, int]:
    actionable = [a for a in alerts if a.get("level") != "ok"]
    return {
        "critical": sum(1 for a in actionable if a.get("level") == "critical"),
        "warning": sum(1 for a in actionable if a.get("level") == "warning"),
        "total": len(actionable),
    }
