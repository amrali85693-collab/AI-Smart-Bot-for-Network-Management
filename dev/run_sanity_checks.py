import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The script prints strings with emoji (e.g. 🔴); Windows consoles default to
# cp1252 which cannot encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from modules.knowledge_base import load_knowledge_base
from modules.anomaly_detector import AnomalyDetector
from modules.alerts import generate_alerts
from modules.data_sources import SimulatedDataSource
from modules.storage import init_db, save_chat_message, get_history
from modules.llm_client import get_llm_response

root = Path(__file__).resolve().parent.parent
kb = load_knowledge_base(root / "data" / "knowledge_base.json")
print('KB entries', len(kb.entries))
search = kb.search('dns troubleshooting', top_k=2)
print('KB search count', len(search))

detector = AnomalyDetector()
df = pd.DataFrame([
    {'bandwidth_mbps': 100, 'latency_ms': 10, 'packet_loss_pct': 0.1, 'cpu_percent': 34.0, 'memory_percent': 60.0},
    {'bandwidth_mbps': 90, 'latency_ms': 12, 'packet_loss_pct': 0.0, 'cpu_percent': 28.0, 'memory_percent': 55.0},
    {'bandwidth_mbps': 1000, 'latency_ms': 200, 'packet_loss_pct': 20.0, 'cpu_percent': 95.0, 'memory_percent': 90.0},
])
detector.fit(df)
print('Anomaly cols', detector.feature_columns)
pred = detector.predict({'bandwidth_mbps': 100, 'latency_ms': 10, 'packet_loss_pct': 0.1, 'cpu_percent': 34.0, 'memory_percent': 60.0})
print('Anomaly result', pred)

alerts_df = pd.DataFrame([
    {'name': 'foo', 'status': 'up', 'cpu_usage': 90, 'latency_ms': 120.0, 'packet_loss_pct': 0.5},
    {'name': 'bar', 'status': 'down', 'cpu_usage': 10, 'latency_ms': None, 'packet_loss_pct': 100.0},
])
print('Alerts', generate_alerts(alerts_df))

sim = SimulatedDataSource(seed=123)
devices = sim.get_devices()
print('Sim devices', len(devices), list(devices.columns))
traffic = sim.get_traffic_history(hours=1)
print('Traffic points', len(traffic), list(traffic.columns))

tmp_db = Path(tempfile.mktemp(suffix='.db'))
init_db(tmp_db)
save_chat_message('user', 'hello', ['test'], tmp_db)
hist = get_history(limit=5, db_path=tmp_db)
print('History chat len', len(hist['chat']))

try:
    get_llm_response([{'role': 'user', 'content': 'hi'}], provider='groq', api_key='')
except Exception as e:
    print('LLM error', type(e).__name__, str(e))

print('Testing HTTP/DNS analysis mismatches...')
from modules.network_monitor import http_health_check, analyze_http_history, dns_lookup, analyze_dns_history

http_sample = [
    {'status': 'up', 'http_status': 200, 'latency_ms': 123.4},
    {'status': 'down', 'http_status': 503, 'latency_ms': None},
]
print('HTTP analysis sample', analyze_http_history(http_sample))

dns_result = dns_lookup('localhost')
print('DNS lookup', dns_result)
print('DNS analysis', analyze_dns_history([dns_result]))

