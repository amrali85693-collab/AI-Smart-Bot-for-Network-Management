"""Tune decision threshold on the new time-aware ensemble."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from modules.anomaly_detector import AnomalyDetector, _auto_label

df = pd.read_csv("data/real_network_traffic.csv", parse_dates=["timestamp"])

split    = int(len(df) * 0.80)
train_df = df.iloc[:split].reset_index(drop=True)
test_df  = df.iloc[split:].reset_index(drop=True)

print("Training time-aware ensemble...")
det = AnomalyDetector(random_state=42, threshold=0.0)  # threshold=0 -> get raw probs
det.fit(train_df)
print(f"Auto-labeled: {det._n_anomaly_train} anomalies / {det._n_anomaly_train + det._n_normal_train} rows")
print(f"Engineered features: {len(det._eng_columns)}")

# Get probabilities on test set
from modules.anomaly_detector import _engineer_features
eng = _engineer_features(test_df, det.feature_columns)
for c in det._eng_columns:
    if c not in eng.columns:
        eng[c] = 0.0
eng = eng[det._eng_columns]
X   = det.scaler.transform(eng.values.astype(float))
prob = (det.rf.predict_proba(X)[:, 1] + det.gb.predict_proba(X)[:, 1]) / 2.0

# Ground truth = same auto-label rules on test
y_true = _auto_label(test_df, det.feature_columns)
print(f"\nTest set: {len(test_df)} rows, {y_true.sum()} anomalies ({y_true.mean()*100:.1f}%)")

print()
print(f"  {'thr':>6}  {'P':>8}  {'R':>8}  {'F1':>8}  {'FPR':>8}")
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.20, 0.70, 0.05):
    yp = (prob > thr).astype(int)
    p  = precision_score(y_true, yp, zero_division=0)
    r  = recall_score(y_true, yp, zero_division=0)
    f  = f1_score(y_true, yp, zero_division=0)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_true, yp).ravel()
    fpr_ = fp_ / (fp_ + tn_)
    marker = " <--" if f > best_f1 else ""
    print(f"  {thr:>6.2f}  {p:>8.4f}  {r:>8.4f}  {f:>8.4f}  {fpr_:>8.4f}{marker}")
    if f > best_f1:
        best_f1, best_thr = f, thr

yp_best = (prob > best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, yp_best).ravel()
print(f"\nBest threshold={best_thr:.2f}")
print(f"  Precision={precision_score(y_true,yp_best):.4f}  Recall={recall_score(y_true,yp_best):.4f}  F1={best_f1:.4f}")
print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}  FPR={fp/(fp+tn):.4f}")
print(f"\nSet DECISION_THRESHOLD = {best_thr:.2f} in anomaly_detector.py")
