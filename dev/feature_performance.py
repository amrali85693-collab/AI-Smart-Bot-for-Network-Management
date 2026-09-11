import time
import tempfile
import json
from pathlib import Path

import pandas as pd

from modules.knowledge_base import load_knowledge_base
from modules.anomaly_detector import AnomalyDetector
from modules.alerts import generate_alerts
from modules.data_sources import SimulatedDataSource
from modules.storage import init_db, save_chat_message, get_history
from modules.network_monitor import (
    analyze_ping_history,
    http_health_check,
    dns_lookup,
    check_port,
    batch_ping,
    calculate_latency_stats,
)

root = Path(__file__).resolve().parent
kb = load_knowledge_base(root / "data" / "knowledge_base.json")

results = []

# Knowledge search timing
queries = [
    "dns troubleshooting",
    "packet loss causes",
    "router configuration",
    "network latency vs bandwidth",
    "firewall rules troubleshooting",
]
search_times = []
for q in queries:
    t0 = time.perf_counter()
    _ = kb.search(q, top_k=3)
    search_times.append(time.perf_counter() - t0)
results.append(("KnowledgeBase.search", len(queries), min(search_times), max(search_times), sum(search_times)/len(search_times)))

# Simulated devices and traffic timing
sim = SimulatedDataSource(seed=123)
for desc, fn in [
    ("SimulatedDataSource.get_devices", sim.get_devices),
    ("SimulatedDataSource.get_traffic_history", lambda: sim.get_traffic_history(hours=1)),
]:
    t0 = time.perf_counter()
    out = fn()
    results.append((desc, 1, time.perf_counter() - t0, None, None))

# Alert generation timing
devices_df = sim.get_devices()
t0 = time.perf_counter()
alerts = generate_alerts(devices_df)
alert_time = time.perf_counter() - t0
results.append(("generate_alerts", 1, alert_time, None, None))

# Anomaly detector timing
history_df = sim.get_traffic_history(hours=1)

detector = AnomalyDetector()
t0 = time.perf_counter()
detector.fit(history_df)
fit_time = time.perf_counter() - t0
results.append(("AnomalyDetector.fit", 1, fit_time, None, None))

metrics = {
    col: float(history_df.iloc[-1].get(col, 0))
    for col in detector.feature_columns
}
t0 = time.perf_counter()
pred = detector.predict(metrics)
pred_time = time.perf_counter() - t0
results.append(("AnomalyDetector.predict", 1, pred_time, None, None))

# Ping analysis timing
ping_history = [
    {"host": "8.8.8.8", "status": "up", "latency_ms": 12.3},
    {"host": "8.8.8.8", "status": "up", "latency_ms": 14.7},
    {"host": "8.8.8.8", "status": "down", "latency_ms": None},
]
t0 = time.perf_counter()
analysis = analyze_ping_history(ping_history)
ping_analysis_time = time.perf_counter() - t0
results.append(("analyze_ping_history", 1, ping_analysis_time, None, None))

# HTTP and DNS lookup timing
for target in ["https://www.google.com", "https://example.com"]:
    t0 = time.perf_counter()
    r = http_health_check(target, timeout=3.0)
    results.append(("http_health_check", target, time.perf_counter() - t0, r.get("status"), r.get("http_status")))

for hostname in ["localhost", "google.com"]:
    t0 = time.perf_counter()
    r = dns_lookup(hostname)
    results.append(("dns_lookup", hostname, time.perf_counter() - t0, r.get("status"), len(r.get("ip_addresses", []))))

# Port check timing
for port in [22, 80, 443]:
    t0 = time.perf_counter()
    r = check_port("localhost", port, timeout=2.0, grab_banner=False)
    results.append(("check_port", f"localhost:{port}", time.perf_counter() - t0, r.get("status"), r.get("latency_ms")))

# Batch ping timing (limited hosts)
hosts = ["8.8.8.8", "1.1.1.1"]
t0 = time.perf_counter()
batch_result = batch_ping(hosts, timeout=2.0, max_workers=2)
results.append(("batch_ping", len(hosts), time.perf_counter() - t0, batch_result["timing"]["total_time_seconds"], batch_result["timing"]["successful_hosts"]))

# Storage timing
tmp_db = root / "tmp_perf_test.db"
if tmp_db.exists():
    tmp_db.unlink()
init_db(tmp_db)

for i in range(5):
    t0 = time.perf_counter()
    save_chat_message("user", f"msg {i}", ["perf"], tmp_db)
    results.append(("save_chat_message", i + 1, time.perf_counter() - t0, None, None))

t0 = time.perf_counter()
_hist = get_history(limit=5, db_path=tmp_db)
results.append(("get_history", 1, time.perf_counter() - t0, len(_hist["chat"]), None))

print(json.dumps(results, indent=2, default=str))
