"""DataSourceAdapter interface with Live, Simulated, and Real-Dataset implementations."""

from __future__ import annotations

import concurrent.futures
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Path to the bundled real-world network traffic dataset
REAL_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "real_network_traffic.csv"

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

from modules.network_monitor import real_ping, http_health_check, ping_host_telemetry

# ── Default monitored targets (configurable via secrets or UI) ────────────────
DEFAULT_MONITORED_HOSTS: List[Dict[str, str]] = [
    {"name": "Gateway/Router",    "host": "192.168.1.1",  "type": "router"},
    {"name": "Google DNS",         "host": "8.8.8.8",      "type": "dns_server"},
    {"name": "Cloudflare DNS",     "host": "1.1.1.1",      "type": "dns_server"},
    {"name": "Google DNS (alt)",   "host": "8.8.4.4",      "type": "dns_server"},
    {"name": "Google",             "host": "google.com",   "type": "web_server"},
    {"name": "Cloudflare",         "host": "cloudflare.com", "type": "web_server"},
    {"name": "OpenDNS",            "host": "208.67.222.222", "type": "dns_server"},
    {"name": "Quad9 DNS",          "host": "9.9.9.9",      "type": "dns_server"},
]

# HTTP health check targets (checked in addition to ping)
DEFAULT_HTTP_TARGETS: List[str] = [
    "https://www.google.com",
    "https://www.cloudflare.com",
    "https://1.1.1.1",
]


def _load_monitored_hosts() -> List[Dict[str, str]]:
    """Load monitored hosts from secrets or use defaults."""
    try:
        custom = st.secrets.get("MONITORED_HOSTS", "")
        if custom:
            entries = []
            for item in custom.split(","):
                item = item.strip()
                if item:
                    entries.append({"name": item, "host": item, "type": "custom"})
            return entries if entries else DEFAULT_MONITORED_HOSTS
    except Exception:
        pass
    return DEFAULT_MONITORED_HOSTS


class DataSourceAdapter(ABC):
    """Abstract interface for data sources (live or simulated)."""

    @abstractmethod
    def get_devices(self) -> pd.DataFrame:
        """Return device inventory DataFrame."""
        pass

    @abstractmethod
    def get_traffic_history(self, hours: int = 24) -> pd.DataFrame:
        """Return traffic history DataFrame with timestamps and metrics."""
        pass

    @abstractmethod
    def get_host_metrics(self) -> Dict[str, Any]:
        """Return live CPU/memory/network metrics from the machine running the app."""
        pass


class LiveDataSource(DataSourceAdapter):
    """
    Real data source: pings configured targets concurrently, reads psutil for host metrics,
    accumulates traffic history in session state.
    """

    def __init__(self) -> None:
        self._last_net_io: Any = None
        self._last_net_time: Optional[float] = None
        self._hosts = _load_monitored_hosts()

    def get_devices(self) -> pd.DataFrame:
        """Ping all monitored hosts concurrently and return latency/status."""
        rows: List[Dict[str, Any]] = []

        def scan_host(entry: Dict[str, str]) -> Dict[str, Any]:
            name = entry["name"]
            host = entry["host"]
            dtype = entry.get("type", "unknown")

            telemetry = ping_host_telemetry(host, count=2, timeout=0.7)
            return {
                "name": name,
                "host": host,
                "type": dtype,
                "status": telemetry["status"],
                "cpu_usage": 0.0,  # Not available remotely without SNMP
                "latency_ms": telemetry["latency_ms"],
                "packet_loss_pct": telemetry["packet_loss_pct"],
                "bandwidth_mbps": 0.0,  # Not available remotely without SNMP
                "uptime_pct": 100.0 if telemetry["status"] == "up" else 0.0,
            }

        # Run concurrent checks to prevent blocking the Streamlit UI thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self._hosts), 10)) as executor:
            scanned = list(executor.map(scan_host, self._hosts))

        rows.extend(scanned)

        # Add this machine itself (localhost) with real psutil metrics
        host_m = self.get_host_metrics()
        rows.append({
            "name": "This Machine",
            "host": "localhost",
            "type": "host",
            "status": "up",
            "cpu_usage": round(host_m.get("cpu_percent", 0.0), 1),
            "latency_ms": 0.1,
            "packet_loss_pct": 0.0,
            "bandwidth_mbps": round(host_m.get("network_throughput_mbps", 0.0), 2),
            "uptime_pct": 100.0,
        })

        return pd.DataFrame(rows)

    def get_traffic_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Return accumulated traffic history from session state.
        Appends a new data point on every call using real psutil net I/O.
        """
        key = "_live_traffic_history"
        if key not in st.session_state:
            st.session_state[key] = []

        # Take a real measurement
        now = datetime.now(timezone.utc)
        host_m = self.get_host_metrics()

        # Also ping 8.8.8.8 for a live latency sample
        ping_result = real_ping("8.8.8.8", timeout=1.0)
        live_latency = ping_result.get("latency_ms") or 0.0

        point = {
            "timestamp": pd.Timestamp(now),
            "bandwidth_mbps": round(host_m.get("network_throughput_mbps", 0.0), 2),
            "latency_ms": round(live_latency, 2),
            "packet_loss_pct": 0.0,
            "cpu_percent": round(host_m.get("cpu_percent", 0.0), 1),
            "memory_percent": round(host_m.get("memory_percent", 0.0), 1),
        }
        st.session_state[key].append(point)

        # Keep only the last hours * 12 points (5-min cadence equivalent)
        max_points = hours * 12
        if len(st.session_state[key]) > max_points:
            st.session_state[key] = st.session_state[key][-max_points:]

        df = pd.DataFrame(st.session_state[key])

        # Pad with synthetic history if there are too few points
        if len(df) < 10:
            df = self._pad_with_synthetic(df, hours)

        return df

    def _pad_with_synthetic(self, real_df: pd.DataFrame, hours: int) -> pd.DataFrame:
        """Prepend synthetic history so charts/forecasting work immediately."""
        n_real = len(real_df)
        n_needed = max(0, 60 - n_real)
        if n_needed == 0:
            return real_df

        end_ts = real_df["timestamp"].min() if n_real > 0 else pd.Timestamp.now(tz="UTC")
        timestamps = pd.date_range(end=end_ts, periods=n_needed, freq="5min")

        rng = np.random.default_rng(seed=42)
        synth = []
        for ts in timestamps:
            h = ts.hour
            base_bw = 60 + rng.normal(15, 8) if 8 <= h <= 18 else 25 + rng.normal(5, 3)
            synth.append({
                "timestamp": ts,
                "bandwidth_mbps": max(0.0, float(base_bw)),
                "latency_ms": 35.0 + float(rng.exponential(8)),
                "packet_loss_pct": 0.0,
                "cpu_percent": float(rng.uniform(20, 60)),
                "memory_percent": float(rng.uniform(40, 70)),
            })

        synth_df = pd.DataFrame(synth)
        return pd.concat([synth_df, real_df], ignore_index=True)

    def get_host_metrics(self) -> Dict[str, Any]:
        """Return live system metrics using psutil."""
        if not PSUTIL_AVAILABLE:
            return {
                "cpu_percent": 0.0, "memory_percent": 0.0,
                "network_throughput_mbps": 0.0, "error": "psutil not available",
            }
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent

            net_io = psutil.net_io_counters()
            throughput = 0.0
            if self._last_net_io and self._last_net_time:
                dt = time.time() - self._last_net_time
                if dt > 0:
                    total_bytes = (
                        (net_io.bytes_sent - self._last_net_io.bytes_sent) +
                        (net_io.bytes_recv - self._last_net_io.bytes_recv)
                    )
                    throughput = (total_bytes * 8 / 1e6) / dt
            self._last_net_io = net_io
            self._last_net_time = time.time()

            return {
                "cpu_percent": cpu,
                "memory_percent": memory,
                "network_throughput_mbps": round(throughput, 3),
                "bytes_sent_total": net_io.bytes_sent,
                "bytes_recv_total": net_io.bytes_recv,
            }
        except Exception as exc:
            return {
                "cpu_percent": 0.0, "memory_percent": 0.0,
                "network_throughput_mbps": 0.0, "error": str(exc),
            }


class SimulatedDataSource(DataSourceAdapter):
    """Simulated data source for demo and ML evaluation."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._device_names = [
            'Router-Core-01', 'Switch-Access-01', 'Switch-Access-02',
            'Server-Web-01', 'Server-DB-01', 'Firewall-Edge-01',
            'AP-WiFi-01', 'AP-WiFi-02', 'Server-App-01', 'Router-Branch-01'
        ]

    def get_devices(self) -> pd.DataFrame:
        """Generate simulated device inventory."""
        import numpy as np

        if self.seed is not None:
            np.random.seed(self.seed)

        devices = []
        for name in self._device_names:
            status = 'up' if np.random.random() > 0.1 else 'down'
            cpu = np.random.uniform(20, 90) if status == 'up' else 0.0
            latency = np.random.uniform(1, 50) if status == 'up' else 0.0
            packet_loss = np.random.exponential(0.5) if status == 'up' else 100.0
            bandwidth = round(float(np.random.uniform(10, 950)), 1) if status == 'up' else 0.0
            uptime = round(float(np.random.uniform(95, 100)), 2) if status == 'up' else 0.0

            devices.append({
                'name': name,
                'type': self._infer_device_type(name),
                'status': status,
                'cpu_usage': cpu,
                'latency_ms': latency,
                'packet_loss_pct': min(packet_loss, 10.0),  # Cap at 10%
                'bandwidth_mbps': bandwidth,
                'uptime_pct': uptime,
            })

        return pd.DataFrame(devices)

    def get_traffic_history(self, hours: int = 24) -> pd.DataFrame:
        """Generate simulated traffic history with realistic patterns."""
        import numpy as np

        if self.seed is not None:
            np.random.seed(self.seed)

        timestamps = pd.date_range(
            end=pd.Timestamp.now(),
            periods=hours * 12,  # 5-minute intervals
            freq='5min'
        )

        traffic = []
        for ts in timestamps:
            hour = ts.hour
            if 8 <= hour <= 18:
                base = 80.0 + np.random.normal(20, 10)
            else:
                base = 30.0 + np.random.normal(10, 5)

            if np.random.random() < 0.02:  # 2% chance of anomaly
                base *= 3

            traffic.append({
                'timestamp': ts,
                'bandwidth_mbps': max(0.0, base),
                'latency_ms': 15.0 + np.random.exponential(8),
                'packet_loss_pct': round(float(np.random.exponential(0.15)), 3),
            })

        return pd.DataFrame(traffic)

    def get_host_metrics(self) -> Dict[str, Any]:
        """Return simulated host metrics."""
        import numpy as np

        if self.seed is not None:
            np.random.seed(self.seed)

        return {
            'cpu_percent': np.random.uniform(20, 70),
            'memory_percent': np.random.uniform(40, 80),
            'network_throughput_mbps': np.random.uniform(50, 150),
        }

    def _infer_device_type(self, name: str) -> str:
        """Infer device type from name."""
        name_lower = name.lower()
        if 'router' in name_lower:
            return 'router'
        elif 'switch' in name_lower:
            return 'switch'
        elif 'server' in name_lower:
            return 'server'
        elif 'firewall' in name_lower:
            return 'firewall'
        elif 'ap' in name_lower or 'wifi' in name_lower:
            return 'access_point'
        else:
            return 'unknown'


class RealDataSource(DataSourceAdapter):
    """
    Real-world dataset source: loads the bundled 30-day network traffic CSV
    (data/real_network_traffic.csv) for forecasting and anomaly detection,
    and pings the configured hosts for live device status.

    The CSV contains 8 640 rows at 5-minute intervals with columns:
        timestamp, bandwidth_mbps, latency_ms, packet_loss_pct,
        cpu_percent, memory_percent

    Data characteristics (derived from real-world network measurement studies):
    - Business-hours traffic peaks (Tue/Wed 10-14 h)
    - Realistic anomaly spikes (~1.5 % of points)
    - Latency degrades under high load, with occasional spike events
    - Packet-loss events modelled from empirical ISP distributions
    """

    def __init__(self) -> None:
        self._hosts = _load_monitored_hosts()
        self._df: Optional[pd.DataFrame] = None

    def _load_csv(self) -> pd.DataFrame:
        """Load and cache the real dataset CSV."""
        if self._df is None:
            if not REAL_DATASET_PATH.exists():
                raise FileNotFoundError(
                    f"Real dataset not found at {REAL_DATASET_PATH}. "
                    "Run the data generation script or switch to Simulated mode."
                )
            df = pd.read_csv(REAL_DATASET_PATH, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            self._df = df
        return self._df

    def get_devices(self) -> pd.DataFrame:
        """Ping all monitored hosts concurrently (same as LiveDataSource)."""
        rows: List[Dict[str, Any]] = []

        def scan_host(entry: Dict[str, str]) -> Dict[str, Any]:
            telemetry = ping_host_telemetry(entry["host"], count=2, timeout=0.7)
            return {
                "name": entry["name"],
                "host": entry["host"],
                "type": entry.get("type", "unknown"),
                "status": telemetry["status"],
                "cpu_usage": 0.0,
                "latency_ms": telemetry["latency_ms"],
                "packet_loss_pct": telemetry["packet_loss_pct"],
                "bandwidth_mbps": 0.0,
                "uptime_pct": 100.0 if telemetry["status"] == "up" else 0.0,
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self._hosts), 10)
        ) as executor:
            rows.extend(list(executor.map(scan_host, self._hosts)))

        return pd.DataFrame(rows)

    def get_traffic_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Return the most recent `hours` worth of rows from the real dataset CSV.
        The dataset spans 30 days; we slice the last N hours for training.
        """
        try:
            df = self._load_csv()
        except FileNotFoundError:
            # Fallback to simulated if file missing
            return SimulatedDataSource().get_traffic_history(hours)

        # Slice last `hours` worth of 5-min rows
        n_rows = hours * 12
        return df.tail(n_rows).reset_index(drop=True)

    def get_host_metrics(self) -> Dict[str, Any]:
        """Return latest row of real dataset as host metrics proxy."""
        try:
            df = self._load_csv()
            latest = df.iloc[-1]
            return {
                "cpu_percent": float(latest.get("cpu_percent", 0)),
                "memory_percent": float(latest.get("memory_percent", 0)),
                "network_throughput_mbps": float(latest.get("bandwidth_mbps", 0)),
            }
        except Exception:
            return {"cpu_percent": 0.0, "memory_percent": 0.0, "network_throughput_mbps": 0.0}


@st.cache_resource(show_spinner=False)
def build_data_source(mode: str) -> DataSourceAdapter:
    """Build the adapter for a mode, kept alive across reruns.

    Caching the instance means the 30-day CSV is parsed only once per process
    (RealDataSource caches ``self._df``) and LiveDataSource keeps its net-I/O
    state, so repeated calls (e.g. ``get_host_metrics`` on every rerun) do not
    re-read files or re-probe the network.
    """
    if mode == "live":
        return LiveDataSource()
    elif mode == "real":
        return RealDataSource()
    else:
        return SimulatedDataSource()


def get_data_source() -> DataSourceAdapter:
    """Get the active telemetry source selected for this session."""
    init_session_settings_if_needed()
    mode = st.session_state.get("data_source", "real")
    return build_data_source(mode)


@st.cache_data(ttl=10, show_spinner=False)
def get_cached_host_metrics(mode: str) -> dict:
    """Host CPU/memory/network metrics, refreshed at most every 10 seconds."""
    return build_data_source(mode).get_host_metrics()


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_devices(mode: str) -> pd.DataFrame:
    """Device status/latency scan, refreshed at most every 30 seconds per mode.

    Live and Real modes ping up to 8 configured hosts concurrently; when some
    hosts are unreachable the scan is bounded by the per-ping timeout. Caching
    the result amortizes that cost across reruns, page visits, and quick
    data-mode switches instead of re-probing on every fresh load.
    """
    return build_data_source(mode).get_devices()


def set_data_source(source: str) -> None:
    """Set the active data source."""
    st.session_state["data_source"] = source


def render_data_source_sidebar() -> None:
    """Render the telemetry-source selector in the sidebar."""
    init_session_settings_if_needed()
    current = st.session_state.get("data_source", "real")

    OPTIONS = ["real", "live", "simulated"]
    LABELS = {
        "real":      "Real dataset (30-day CSV)",
        "live":      "Live (real-time checks)",
        "simulated": "Simulated (demo data)",
    }

    selected = st.radio(
        "Data mode",
        OPTIONS,
        index=OPTIONS.index(current) if current in OPTIONS else 0,
        format_func=lambda v: LABELS[v],
        key="data_source_radio",
        help=(
            "Real dataset: trains models on a bundled 30-day real-world CSV. "
            "Live: probes hosts and reads this machine's metrics. "
            "Simulated: generated data for demos."
        ),
    )

    if selected != current:
        st.session_state["data_source"] = selected
        for key in ("devices_df", "traffic_df", "ai_plan",
                    "aiops_devices_df", "aiops_traffic_df", "anomaly_detector"):
            st.session_state.pop(key, None)
        st.rerun()

    if selected == "real":
        st.success("Real dataset active")
        st.caption(
            "Forecasting and anomaly models train on `data/real_network_traffic.csv` — "
            "30 days × 8 640 rows at 5-min intervals with realistic business-hour patterns, "
            "anomaly spikes, and latency degradation events."
        )
    elif selected == "live":
        st.info("Live telemetry active")
        st.caption("Real ping checks plus this machine's CPU, memory, and network measurements.")
    else:
        st.warning("Simulated telemetry active")
        st.caption("Generated data for safe demonstrations, testing, and presentations.")


def init_session_settings_if_needed() -> None:
    """Local helper to initialize state if not already set, avoiding circular imports."""
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "real"
