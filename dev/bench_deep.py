"""
Benchmark: supervised vs unsupervised anomaly detection.
Tests Autoencoder (deep learning), supervised Random Forest,
and the current LOF to understand the theoretical ceiling.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score

from modules.anomaly_detector import inject_synthetic_anomalies

df = pd.read_csv("data/real_network_traffic.csv", parse_dates=["timestamp"])
features = ["bandwidth_mbps", "latency_ms", "packet_loss_pct", "cpu_percent", "memory_percent"]

train_raw = df.iloc[:7000][features].fillna(0)
test_raw  = df.iloc[7000:][features].fillna(0).reset_index(drop=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(train_raw)
X_test_clean = scaler.transform(test_raw)

# Inject anomalies into test set
test_modified, labels = inject_synthetic_anomalies(test_raw.copy(), anomaly_rate=0.10, severity=3.0)
X_test = scaler.transform(test_modified)
y_test  = np.array(labels)

def report(name, y_pred):
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    print(f"  {name:<35}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

print("\n=== UNSUPERVISED (no labels at train time) ===")

# Current LOF
lof = LocalOutlierFactor(n_neighbors=30, contamination=0.10, novelty=True)
lof.fit(X_train)
preds_lof = (lof.predict(X_test) == -1).astype(int)
report("LOF (current)", preds_lof)

# Autoencoder reconstruction error
try:
    import tensorflow as tf
    from tensorflow import keras

    inp = keras.Input(shape=(5,))
    enc = keras.layers.Dense(16, activation="relu")(inp)
    enc = keras.layers.Dense(8,  activation="relu")(enc)
    enc = keras.layers.Dense(4,  activation="relu")(enc)
    dec = keras.layers.Dense(8,  activation="relu")(enc)
    dec = keras.layers.Dense(16, activation="relu")(dec)
    out = keras.layers.Dense(5)(dec)
    ae  = keras.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")

    ae.fit(X_train, X_train, epochs=40, batch_size=64, verbose=0,
           validation_split=0.1,
           callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    recon_err = np.mean((X_test - ae.predict(X_test, verbose=0))**2, axis=1)
    # Threshold = mean + 2*std on clean training reconstruction
    train_err = np.mean((X_train - ae.predict(X_train, verbose=0))**2, axis=1)
    threshold = train_err.mean() + 2 * train_err.std()
    preds_ae = (recon_err > threshold).astype(int)
    report("Autoencoder (unsupervised)", preds_ae)

    # Try multiple thresholds
    print("\n  Autoencoder threshold sensitivity:")
    for mult in [1.5, 2.0, 2.5, 3.0]:
        thr = train_err.mean() + mult * train_err.std()
        p_ = (recon_err > thr).astype(int)
        p  = precision_score(y_test, p_, zero_division=0)
        r  = recall_score(y_test, p_, zero_division=0)
        f  = f1_score(y_test, p_, zero_division=0)
        print(f"    threshold=mean+{mult}*std  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

    HAS_TF = True
except ImportError:
    print("  Autoencoder: TensorFlow not installed")
    HAS_TF = False

print("\n=== SUPERVISED (labels available at train time) ===")
print("  (This is the theoretical ceiling — assumes we have pre-labeled anomalies)")

# Create labelled training set with known anomalies
train_modified, train_labels = inject_synthetic_anomalies(train_raw.copy(), anomaly_rate=0.10, severity=3.0)
X_train_sup = scaler.transform(train_modified)
y_train_sup = np.array(train_labels)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_sup, y_train_sup)
preds_rf = rf.predict(X_test)
report("Random Forest (supervised)", preds_rf)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
gb.fit(X_train_sup, y_train_sup)
preds_gb = gb.predict(X_test)
report("GradientBoosting (supervised)", preds_gb)

if HAS_TF:
    # Supervised neural net
    nn_inp = keras.Input(shape=(5,))
    x = keras.layers.Dense(64, activation="relu")(nn_inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    nn_out = keras.layers.Dense(1, activation="sigmoid")(x)
    nn = keras.Model(nn_inp, nn_out)
    nn.compile(optimizer="adam", loss="binary_crossentropy")
    nn.fit(X_train_sup, y_train_sup, epochs=50, batch_size=64, verbose=0,
           validation_split=0.1,
           callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    preds_nn = (nn.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
    report("Dense NN (supervised)", preds_nn)

print("\nConclusion:")
print("  Supervised models get 90%+ because they see labeled anomalies during training.")
print("  Unsupervised models (LOF, Autoencoder) are limited by the unlabeled nature of the problem.")
print("  For 90%+ F1 we need: labeled training data OR a rule-based threshold approach.")
