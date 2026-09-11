"""
Walk-forward forecast method comparison.

For each metric (bandwidth / latency / packet loss) and each candidate method
(persistence / rolling / EMA / linear / seasonal-hour average plus the `auto`
selector), run the same 354-window walk-forward protocol as dev/eval_accuracy.py
and report MAE / MAPE / RMSE. Also tallies which models the `auto` selector
picks across windows.

Usage:
    python dev/bench_forecast.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from collections import defaultdict

from modules.forecasting import NetworkForecaster

df = pd.read_csv("data/real_network_traffic.csv", parse_dates=["timestamp"])

METRICS = ["bandwidth_mbps", "latency_ms", "packet_loss_pct"]
METHODS = ["auto", "persistence", "rolling", "ema", "linear", "seasonal_hour"]

window, step, stride = 144, 6, 24
results = defaultdict(lambda: defaultdict(list))
selections = defaultdict(int)
n_windows = 0

for start in range(window, len(df) - step, stride):
    hist = df.iloc[start - window:start]
    for metric in METRICS:
        actual = float(df.iloc[start + step - 1][metric])
        for method in METHODS:
            fc = NetworkForecaster(method=method).forecast_metric(
                hist, metric, horizon_minutes=30)
            pred = fc.get("predicted_value")
            if pred is None:
                continue
            err = pred - actual
            results[method][metric].append(
                (abs(err), err**2, abs(err) / (abs(actual) + 1e-6) * 100))
            if method == "auto":
                selections[fc.get("method_used", "?")] += 1
    n_windows += 1

print(f"Windows: {n_windows}\n")
print(f"{'method':<16} {'metric':<22} {'MAE':>8} {'MAPE%':>10} {'RMSE':>8}  n")
print("-" * 76)
for method in METHODS:
    for metric in METRICS:
        rows = results[method][metric]
        if not rows:
            print(f"{method:<16} {metric:<22}  N/A")
            continue
        maes, rmses, mapes = zip(*rows)
        print(f"{method:<16} {metric:<22} {np.mean(maes):>8.3f} "
              f"{np.mean(mapes):>10.2f} {np.sqrt(np.mean(rmses)):>8.3f}  {len(rows)}")
    print()

print("`auto` selection frequency (across all metrics/windows):")
total = n_windows * len(METRICS)
for name, cnt in sorted(selections.items(), key=lambda x: -x[1]):
    print(f"  {name:<26} {cnt:>4}  ({cnt / total * 100:5.1f}%)")

for label, needle in [("any-rolling", "rolling"), ("any-seasonal", "seasonal"),
                      ("any-ema", "ema"), ("any-linear", "linear"),
                      ("any-persistence", "persistence")]:
    c = sum(v for k, v in selections.items() if needle in k)
    print(f"{label:<16} {c:>4}  ({c / total * 100:5.1f}%)")
