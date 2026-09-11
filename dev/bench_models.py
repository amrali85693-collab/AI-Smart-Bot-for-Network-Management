"""Benchmark anomaly detection models side by side on the real dataset."""
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from modules.anomaly_detector import inject_synthetic_anomalies

df = pd.read_csv("data/real_network_traffic.csv", parse_dates=["timestamp"])
features = ["bandwidth_mbps", "latency_ms", "packet_loss_pct", "cpu_percent", "memory_percent"]

train_df = df.iloc[:7000][features].fillna(0)
test_df  = df.iloc[7000:][features].fillna(0).reset_index(drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df)
X_test_base = scaler.transform(test_df)

RATES = [0.05, 0.10, 0.15]
MODELS = {
    "IsolationForest (old)": IsolationForest(contamination=0.05, n_estimators=200, random_state=42),
    "LOF (novelty)":         LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True),
    "OneClassSVM":           OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
}

def evaluate(y_true, y_pred):
    tp = sum(1 for p,t in zip(y_pred, y_true) if p==1 and t==1)
    fp = sum(1 for p,t in zip(y_pred, y_true) if p==1 and t==0)
    tn = sum(1 for p,t in zip(y_pred, y_true) if p==0 and t==0)
    fn = sum(1 for p,t in zip(y_pred, y_true) if p==0 and t==1)
    prec   = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    f1     = 2*prec*recall/(prec+recall) if (prec+recall)>0 else 0
    fpr    = fp/(fp+tn) if (fp+tn)>0 else 0
    return dict(precision=prec, recall=recall, f1=f1, fpr=fpr, tp=tp, fp=fp, tn=tn, fn=fn)

print(f"\n{'Model':<26} {'Rate':>6}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}")
print("-"*70)

for name, model in MODELS.items():
    model.fit(X_train)
    for rate in RATES:
        modified_raw, labels = inject_synthetic_anomalies(
            df.iloc[7000:][features].fillna(0).reset_index(drop=True),
            anomaly_rate=rate, severity=3.0
        )
        X_test = scaler.transform(modified_raw)
        raw_preds = model.predict(X_test)        # 1=normal, -1=anomaly
        preds = [1 if p==-1 else 0 for p in raw_preds]
        m = evaluate(labels, preds)
        print(f"  {name:<24} {rate*100:>5.0f}%  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['fpr']:>7.4f}")
    print()

# ── Also test LOF with better n_neighbors ────────────────────────────────────
print("\n=== LOF sensitivity to n_neighbors (rate=10%) ===")
print(f"  {'n_neighbors':>12}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'FPR':>7}")
modified_raw, labels = inject_synthetic_anomalies(
    df.iloc[7000:][features].fillna(0).reset_index(drop=True),
    anomaly_rate=0.10, severity=3.0
)
X_test_m = scaler.transform(modified_raw)
for k in [5, 10, 20, 30, 50]:
    lof = LocalOutlierFactor(n_neighbors=k, contamination=0.05, novelty=True)
    lof.fit(X_train)
    raw_preds = lof.predict(X_test_m)
    preds = [1 if p==-1 else 0 for p in raw_preds]
    m = evaluate(labels, preds)
    print(f"  {k:>12}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['fpr']:>7.4f}")
