"""
Full accuracy evaluation:
  1. Anomaly detection  — time-aware RF+GB ensemble
  2. Forecasting        — walk-forward MAE / RMSE / MAPE
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, confusion_matrix)
from modules.anomaly_detector import (
    AnomalyDetector, _auto_label, inject_synthetic_anomalies
)
from modules.forecasting import NetworkForecaster

df = pd.read_csv("data/real_network_traffic.csv", parse_dates=["timestamp"])
FEATURES = ["bandwidth_mbps", "latency_ms", "packet_loss_pct",
            "cpu_percent", "memory_percent"]
print(f"Dataset: {len(df)} rows  |  columns: {FEATURES}")

split    = int(len(df) * 0.80)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════
# 1. ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ANOMALY DETECTION  —  Time-aware RF + GB Ensemble")
print("="*65)

det = AnomalyDetector(random_state=42)
det.fit(train_df)
print(f"  Training: {det._n_normal_train} normal  +  {det._n_anomaly_train} anomaly")
print(f"  Engineered features: {len(det._eng_columns)}")

# ── A: Proper evaluation — same domain rules as ground truth ──────
y_true = _auto_label(test_df, FEATURES)
batch  = det.predict_batch(test_df.copy())
y_pred = batch["is_anomaly"].astype(int).values

p  = precision_score(y_true, y_pred, zero_division=0)
r  = recall_score(y_true, y_pred, zero_division=0)
f  = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"\n  [A] Domain-rule ground truth on clean test set")
print(f"      Anomalies in test: {y_true.sum()} / {len(y_true)}  ({y_true.mean()*100:.1f}%)")
print(f"      Precision : {p:.4f}")
print(f"      Recall    : {r:.4f}")
print(f"      F1 Score  : {f:.4f}")
print(f"      FPR       : {fp/(fp+tn):.4f}")
print(f"      TP={tp}  FP={fp}  TN={tn}  FN={fn}")

# ── B: Synthetic injection at varying severity ────────────────────
print(f"\n  [B] Synthetic injection  (rate=10%, vary severity)")
print(f"  {'severity':>10}  {'P':>8}  {'R':>8}  {'F1':>8}  {'FPR':>8}  TP  FP  TN  FN")
print("  " + "-"*70)
for sev in [2.0, 3.0, 5.0, 10.0]:
    mod, labels = inject_synthetic_anomalies(test_df.copy(), 0.10, sev)
    m = det.evaluate(mod, labels)
    print(f"  {sev:>10.1f}x  {m['precision']:>8.4f}  {m['recall']:>8.4f}  "
          f"{m['f1']:>8.4f}  {m['false_positive_rate']:>8.4f}  "
          f"{m['true_positives']:>3}  {m['false_positives']:>3}  "
          f"{m['true_negatives']:>4}  {m['false_negatives']:>3}")

# ── C: Synthetic injection at varying rates ───────────────────────
print(f"\n  [C] Synthetic injection  (severity=4×, vary rate)")
print(f"  {'rate':>8}  {'P':>8}  {'R':>8}  {'F1':>8}  {'FPR':>8}  TP  FP  TN  FN")
print("  " + "-"*67)
for rate in [0.03, 0.05, 0.10, 0.15, 0.20]:
    mod, labels = inject_synthetic_anomalies(test_df.copy(), rate, 4.0)
    m = det.evaluate(mod, labels)
    print(f"  {rate*100:>7.0f}%  {m['precision']:>8.4f}  {m['recall']:>8.4f}  "
          f"{m['f1']:>8.4f}  {m['false_positive_rate']:>8.4f}  "
          f"{m['true_positives']:>3}  {m['false_positives']:>3}  "
          f"{m['true_negatives']:>4}  {m['false_negatives']:>3}")

# ════════════════════════════════════════════════════════════════════
# 2. FORECASTING
# ════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("FORECASTING  —  Walk-forward, 30-min horizon")
print("="*65)
forecaster = NetworkForecaster(method="auto")
window, step, stride = 144, 6, 24

print(f"  {'metric':<22}  {'MAE':>8}  {'MAPE%':>10}  {'RMSE':>8}  {'method':<15}  n")
print("  " + "-"*75)

for metric in ["bandwidth_mbps", "latency_ms", "packet_loss_pct"]:
    maes, mapes, rmses, last_method = [], [], [], "?"
    for start in range(window, len(df) - step, stride):
        hist   = df.iloc[start - window : start]
        actual = float(df.iloc[start + step - 1][metric])
        fc     = forecaster.forecast_metric(hist, metric, horizon_minutes=30)
        pred   = fc.get("predicted_value")
        if pred is None:
            continue
        last_method = fc.get("method_used", "?")
        err  = pred - actual
        maes.append(abs(err))
        rmses.append(err**2)
        mapes.append(abs(err) / (abs(actual) + 1e-6) * 100)

    if maes:
        print(f"  {metric:<22}  {np.mean(maes):>8.3f}  {np.mean(mapes):>10.2f}  "
              f"{np.sqrt(np.mean(rmses)):>8.3f}  {last_method:<15}  {len(maes)}")

print("\n  * MAPE is inflated for near-zero metrics like packet_loss; MAE is the reliable metric there.")
print("\nDone.")
