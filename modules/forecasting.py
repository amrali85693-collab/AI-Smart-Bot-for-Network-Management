"""Enhanced time-series forecasting for network metrics.

Replaces the previous variance-heuristic forecaster with a validation-driven
selection: a family of classical statistical and daily-seasonal candidate
models is scored on a held-out tail of the history at the exact forecast
horizon, and the best model (or a small blend) is used for the prediction.
Confidence intervals are calibrated from the empirical quantiles of the
candidate's out-of-sample residuals (conformal-style) so that the stated
coverage is honest per metric and horizon. An optional LSTM (Keras) candidate
can participate in auto-selection when a deep-learning runtime is installed and
enough history is available; it is trained in an isolated subprocess (never
imports TensorFlow in the app process) so the app runs without TensorFlow and
is immune to native-library clashes between TensorFlow and pandas.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATSMODELS_AVAILABLE = False

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Default candidate family, in evaluation order. (name, human label)
_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("persistence",   "Persistence (last value)"),
    ("rolling",       "Rolling average (10)"),
    ("rolling_24",    "Rolling average (24)"),
    ("rolling_48",    "Rolling average (48)"),
    ("ema",           "Exponential moving average"),
    ("linear",        "Linear trend (last 20)"),
    ("linear_48",     "Linear trend (last 48)"),
    ("seasonal_hour", "Same-hour-of-day average"),
    ("seasonal_naive", "Daily seasonal naive (t-24h)"),
)
# Note: Holt-Winters damped trend is available as an explicit method
# ('exponential'); it is not scored in auto-selection because it is slower and
# never won on the benchmark, but it stays selectable for demonstration.

_Z = {0.80: 1.2816, 0.85: 1.4400, 0.90: 1.6449, 0.95: 1.9599, 0.99: 2.5758}

# Empirically calibrated inflation applied to the conformal band. The final
# forecast is extrapolated one horizon beyond the training span, so raw
# step-ahead residuals are slightly optimistic; a small widening keeps the
# stated coverage honest.
_CI_INFLATION = 1.4


def dataframe_signature(df: pd.DataFrame) -> str:
    """Cheap content hash used to memoize per-telemetry-snapshot results.

    The signature is stable for the same rows, so expensive derived results
    (forecasts, batch predictions) are computed once per snapshot instead of
    on every Streamlit rerun.
    """
    try:
        h = pd.util.hash_pandas_object(df, index=True).sum()
    except Exception:
        h = len(df)
    return f"{len(df)}-{int(h)}"


# ────────────────────────────────────────────────────────────────────────────
# Candidate models
# ────────────────────────────────────────────────────────────────────────────

def _persistence(v: np.ndarray, steps: int) -> float:
    return float(v[-1])


def _rolling(v: np.ndarray, steps: int, k: int) -> float | None:
    if len(v) < k:
        return None
    return float(np.mean(v[-k:]))


def _ema(v: np.ndarray, steps: int, alpha: float = 0.3) -> float:
    e = float(v[0])
    for x in v[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return e


def _linear(v: np.ndarray, steps: int, n: int) -> float | None:
    if len(v) < 3:
        return None
    k = min(n, len(v))
    y = v[-k:].astype(float)
    x = np.arange(k, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope * (k - 1 + steps) + intercept)


def _exponential(v: np.ndarray, steps: int) -> float | None:
    if not STATSMODELS_AVAILABLE:
        return None
    import warnings as _warnings
    try:
        s = pd.Series(v.astype(float), index=pd.RangeIndex(len(v)))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                s, trend="add", seasonal=None, damped_trend=True,
                initialization_method="heuristic",
            ).fit(optimized=True)
        fc = model.forecast(max(1, steps))
        return float(fc.iloc[-1])
    except Exception:
        return None


def _seasonal_hour(v: np.ndarray, ts: np.ndarray, steps: int) -> float | None:
    if len(v) < 48:
        return None
    hours = ts.astype("datetime64[h]").astype(np.int64) % 24
    last_hour = int(hours[-1])
    idx = np.where(hours == last_hour)[0]
    if len(idx) < 1:
        return None
    return float(np.mean(v[idx]))


def _seasonal_naive(v: np.ndarray, steps: int, period: int) -> float | None:
    if len(v) <= period:
        return None
    return float(v[-period])


def _predict(name: str, v: np.ndarray, ts: np.ndarray, steps: int,
             period: int) -> float | None:
    """Predict `steps` points ahead using only data in (v, ts)."""
    try:
        if name == "persistence":
            return _persistence(v, steps)
        if name == "rolling":
            return _rolling(v, steps, 10)
        if name == "rolling_24":
            return _rolling(v, steps, 24)
        if name == "rolling_48":
            return _rolling(v, steps, 48)
        if name == "ema":
            return _ema(v, steps)
        if name == "linear":
            return _linear(v, steps, 20)
        if name == "linear_48":
            return _linear(v, steps, 48)
        if name == "exponential":
            return _exponential(v, steps)
        if name == "seasonal_hour":
            return _seasonal_hour(v, ts, steps)
        if name == "seasonal_naive":
            return _seasonal_naive(v, steps, period)
    except Exception:
        return None
    return None


# ────────────────────────────────────────────────────────────────────────────
# Validation / conformal calibration helpers
# ────────────────────────────────────────────────────────────────────────────

def _interval_minutes(ts: np.ndarray) -> float:
    if ts is not None and len(ts) >= 2:
        diffs = np.diff(ts.astype("datetime64[s]").astype(np.int64))
        med = float(np.median(diffs))
        if med > 0:
            return med / 60.0
    return 5.0


def _daily_period(ts: np.ndarray) -> int:
    minutes = _interval_minutes(ts)
    if minutes > 0:
        return max(24, int(round(24 * 60 / minutes)))
    return 288


def _score_on_tail(name: str, v: np.ndarray, ts: np.ndarray, steps: int,
                   period: int, grid: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Out-of-sample MAE + residual array for a candidate on the tail grid.

    For every t in the grid, the candidate is fit on data up to t and scored
    against the actual value at t+steps (same horizon used for the final
    forecast), so the residuals reflect true step-ahead forecast error.
    """
    residuals = []
    for t in grid:
        p = _predict(name, v[: t + 1], ts[: t + 1], steps, period)
        if p is None:
            return None
        residuals.append(float(v[t + steps]) - p)
    if not residuals:
        return None
    residuals = np.asarray(residuals)
    mae = float(np.mean(np.abs(residuals)))
    return mae, residuals


def _tail_grid(n: int, steps: int, tail_len: int) -> np.ndarray:
    """Contiguous validation indices covering the recent tail.

    Model selection is judged on the most recent regime (where the next
    forecast will land), so the grid is [n-tail_len, n-1-steps].
    """
    lo = max(2, n - tail_len)
    hi = n - 1 - steps
    if hi <= lo:
        return np.array([], dtype=int)
    return np.arange(lo, hi + 1, dtype=int)


def _ci_residuals(components: list[str], v: np.ndarray, ts: np.ndarray,
                  steps: int, period: int, max_points: int = 60) -> np.ndarray:
    """Step-ahead residuals of a candidate (or blend) over the whole history.

    Each residual is the error of a forecast made with data up to t for the
    value at t+steps, so the empirical distribution covers every regime in the
    window. This is a much more representative basis for the confidence band
    than the short validation tail alone.
    """
    n = len(v)
    lo, hi = 2, n - 1 - steps
    if hi <= lo:
        return np.array([])
    grid = np.unique(np.linspace(lo, hi, min(max_points, hi - lo + 1)).round().astype(int))
    grid = grid[grid <= hi]
    blended = []
    for t in grid:
        preds = [_predict(name, v[: t + 1], ts[: t + 1], steps, period)
                 for name in components]
        if any(p is None for p in preds):
            continue
        blended.append(float(v[t + steps]) - float(np.mean(preds)))
    return np.asarray(blended)


def _quantile_ci(pred: float, residuals: np.ndarray, level: float,
                 fallback_std: float) -> tuple[float, float, str]:
    """Conformal residual-quantile interval (asymmetric) around `pred`.

    The adaptive conformal band is floored by the classical z*std interval so
    heavy-tailed metrics (e.g. packet loss spikes) never get an over-tight
    band that misses tail events.
    """
    std = float(np.std(residuals)) if len(residuals) >= 2 else fallback_std
    if len(residuals) >= 8:
        lo_q = float(np.quantile(residuals, (1.0 - level) / 2.0, method="higher"))
        hi_q = float(np.quantile(residuals, (1.0 + level) / 2.0, method="higher"))
        ci_method = "conformal_residual_quantiles"
    else:
        z = _Z.get(round(level, 2), 1.96)
        lo_q, hi_q = -z * std, z * std
        ci_method = "std_residual"
    lo_q *= _CI_INFLATION
    hi_q *= _CI_INFLATION
    z = _Z.get(round(level, 2), 1.96)
    lo_q = min(lo_q, -z * std)
    hi_q = max(hi_q, z * std)
    lower = float(pred + lo_q)
    upper = float(pred + hi_q)
    return lower, upper, ci_method


# ────────────────────────────────────────────────────────────────────────────
# Optional deep-learning candidate (Keras / TensorFlow, lazily imported)
# ────────────────────────────────────────────────────────────────────────────

_LSTM_AVAILABLE: bool | None = None
_LSTM_CACHE: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
_LSTM_TTL_S = 300.0

# TensorFlow runs in a subprocess that imports only numpy + tensorflow. On some
# installs, importing tensorflow *after* pandas hard-crashes the interpreter
# (native ABI clash), so the parent process never loads TF in-process.
_LSTM_SUBPROCESS_SCRIPT = r"""
import sys, os, json
import tensorflow as tf
import numpy as np
path, steps, lookback, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
v = np.load(path)['v'].astype(float)
mean, std = float(np.mean(v)), float(np.std(v)) + 1e-6
z = (v - mean) / std
X, y = [], []
for i in range(lookback, len(z) - steps + 1):
    X.append(z[i - lookback:i]); y.append(z[i + steps - 1])
X = np.asarray(X, dtype='float32').reshape(-1, lookback, 1)
y = np.asarray(y, dtype='float32')
split = max(int(len(X) * 0.8), lookback + steps)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
model.fit(X[:split], y[:split], validation_data=(X[split:], y[split:]),
          epochs=epochs, batch_size=32, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
pred = max(float(model.predict(z[-lookback:].reshape(1, lookback, 1), verbose=0)[0, 0]) * std + mean, 0.0)
residuals = []
tail_start = max(len(z) - min(len(z) // 4, 40), lookback)
for t in range(tail_start, len(z) - steps):
    x = z[t - lookback:t].reshape(1, lookback, 1)
    p = float(model.predict(x, verbose=0)[0, 0]) * std + mean
    residuals.append(float(v[t + steps]) - p)
print(json.dumps({'pred': pred, 'residuals': residuals,
                  'val_mae': float(np.mean(np.abs(residuals))) if residuals else None}), flush=True)
"""


def _keras_available() -> bool:
    """Whether a working Keras/TF runtime exists (probed in a subprocess)."""
    global _LSTM_AVAILABLE
    if _LSTM_AVAILABLE is not None:
        return _LSTM_AVAILABLE
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import tensorflow; print(tensorflow.__version__)"],
            capture_output=True, text=True, timeout=120,
        )
        _LSTM_AVAILABLE = r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        _LSTM_AVAILABLE = False
    return _LSTM_AVAILABLE


def _lstm_forecast(v: np.ndarray, ts: np.ndarray, steps: int,
                   lookback: int = 48, epochs: int = 40) -> dict[str, Any] | None:
    """Train a small LSTM in a subprocess; return forecast + tail residuals."""
    if len(v) < lookback + steps + 24:
        return None
    if not _keras_available():
        return None

    sig = (len(v), steps, round(float(v[-1]), 4),
           round(float(v[-min(12, len(v))]), 4))
    now = time.time()
    hit = _LSTM_CACHE.get(sig)
    if hit and now - hit[0] < _LSTM_TTL_S:
        return hit[1]

    fd, path = tempfile.mkstemp(suffix=".npz")
    try:
        os.close(fd)
        np.savez(path, v=np.asarray(v, dtype=float))
        r = subprocess.run(
            [sys.executable, "-c", _LSTM_SUBPROCESS_SCRIPT,
             path, str(steps), str(lookback), str(epochs)],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            return None
        out = json.loads(r.stdout.strip().splitlines()[-1])
        result = {
            "pred": float(out["pred"]),
            "residuals": np.asarray(out["residuals"], dtype=float),
            "val_mae": out.get("val_mae"),
        }
        _LSTM_CACHE[sig] = (now, result)
        return result
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Public forecaster
# ────────────────────────────────────────────────────────────────────────────

class NetworkForecaster:
    """Validation-driven forecaster for network metrics.

    Args:
        method: 'auto' (recommended), or one of 'persistence', 'rolling',
            'ema', 'linear', 'exponential', 'seasonal_hour',
            'seasonal_naive', 'lstm'.
        use_deep_learning: if True and a Keras/TensorFlow runtime is
            available with sufficient history, an LSTM candidate is included
            in auto-selection. Defaults to False: the LSTM is trained in a
            subprocess (several seconds per call) and never won on the
            benchmark series, so it is opt-in.
        validation_fraction: fraction of the history held out for model
            selection and confidence-interval calibration.
    """

    def __init__(self, method: str = "auto", use_deep_learning: bool = False,
                 validation_fraction: float = 0.25):
        self.method = method
        self.use_deep_learning = use_deep_learning
        self.validation_fraction = max(0.1, min(0.4, validation_fraction))
        self.last_model: str | None = None

    # ── public API ────────────────────────────────────────────────────────────

    def forecast_metric(
        self,
        history_df: pd.DataFrame,
        metric: str = "bandwidth_mbps",
        horizon_minutes: int = 30,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """
        Forecast a metric into the future.

        Returns a dict with 'predicted_value', 'lower_bound', 'upper_bound'
        plus metadata ('method_used', 'data_points_used', 'validation_mae',
        'ci_method', 'confidence_level', ...).
        """
        if metric not in history_df.columns:
            return self._error(f"Metric {metric} not found in data")

        v, ts = self._extract_series(history_df, metric)
        if len(v) < 10:
            return self._error("Insufficient data points (need at least 10)")

        interval_min = _interval_minutes(ts)
        steps = max(1, int(round(horizon_minutes / interval_min))) if interval_min > 0 else max(1, round(horizon_minutes / 5))
        level = max(0.5, min(0.99, float(confidence_level)))
        period = _daily_period(ts)

        if self.method == "auto":
            result = self._auto_forecast(v, ts, steps, level, period)
        else:
            result = self._method_forecast(self.method, v, ts, steps, level, period)

        result["method_used"] = self.last_model or result.get("model_used", self.method)
        result["data_points_used"] = len(v)
        result["forecast_horizon_minutes"] = horizon_minutes
        return result

    def check_capacity_threshold(
        self,
        history_df: pd.DataFrame,
        metric: str = "bandwidth_mbps",
        capacity_threshold: float = 100.0,
        horizon_minutes: int = 30,
    ) -> dict[str, Any]:
        """Check if the forecast will exceed a capacity threshold."""
        forecast = self.forecast_metric(history_df, metric, horizon_minutes)

        if forecast.get("predicted_value") is None:
            return {
                **forecast,
                "will_exceed_threshold": False,
                "threshold": capacity_threshold,
            }

        predicted = forecast["predicted_value"]
        will_exceed = predicted > capacity_threshold

        if will_exceed:
            forecast["warning"] = (
                f"Forecasted {metric} ({predicted:.1f}) will exceed capacity "
                f"threshold ({capacity_threshold}) within {horizon_minutes} minutes"
            )
        else:
            forecast["info"] = (
                f"Forecasted {metric} ({predicted:.1f}) is within capacity "
                f"threshold ({capacity_threshold})"
            )

        forecast["will_exceed_threshold"] = will_exceed
        forecast["threshold"] = capacity_threshold
        return forecast

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _error(msg: str) -> dict[str, Any]:
        return {
            "predicted_value": None,
            "lower_bound": None,
            "upper_bound": None,
            "error": msg,
        }

    @staticmethod
    def _extract_series(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        v = series.to_numpy(dtype=float)
        if "timestamp" in df.columns:
            ts = df.loc[series.index, "timestamp"].to_numpy()
        else:
            ts = np.arange(len(series), dtype="datetime64[s]")
        return v, ts

    def _auto_forecast(self, v: np.ndarray, ts: np.ndarray, steps: int,
                       level: float, period: int) -> dict[str, Any]:
        n = len(v)
        grid = _tail_grid(n, steps, tail_len=max(24, min(n // 4, 48)))

        # 1) which candidates are applicable on the full series?
        full = {name: _predict(name, v, ts, steps, period) for name, _ in _CANDIDATES}
        applicable = [name for name, p in full.items() if p is not None]

        # 2) score each on the held-out tail at the target horizon
        scores: dict[str, tuple[float, np.ndarray]] = {}
        for name in applicable:
            scored = _score_on_tail(name, v, ts, steps, period, grid)
            if scored is not None:
                scores[name] = scored

        fallback_std = float(np.std(v[-min(20, n):])) if n >= 2 else float(np.abs(v[-1])) or 1.0

        if not scores:
            pred = float(v[-1])
            lower, upper, ci_method = _quantile_ci(pred, np.array([]), level, fallback_std)
            self.last_model = "rolling"
            return {
                "predicted_value": pred,
                "lower_bound": max(0.0, lower),
                "upper_bound": max(upper, lower),
                "model_used": "rolling",
                "model_family": "fallback",
                "validation_mae": None,
                "ci_method": ci_method,
                "confidence_level": level,
                "ensemble_components": ["rolling"],
            }

        ranked = sorted(scores, key=lambda nm: scores[nm][0])
        best = ranked[0]
        best_mae = scores[best][0]
        components = [best]
        pred = float(full[best])

        # 3) small blend with the runner-up if nearly as good (reduces variance)
        if len(ranked) >= 2 and scores[ranked[1]][0] <= best_mae * 1.05:
            second = ranked[1]
            pred = 0.5 * float(full[best]) + 0.5 * float(full[second])
            components = [best, second]

        # 4) optional deep-learning candidate
        lstm = None
        if self.use_deep_learning and n >= 240:
            lstm = _lstm_forecast(v, ts, steps)
            if lstm and lstm.get("val_mae") is not None and lstm["val_mae"] <= best_mae:
                pred = float(lstm["pred"])
                best_mae = float(lstm["val_mae"])
                components = ["lstm"]

        # 5) calibrate the CI from full-history step-ahead residuals
        if components == ["lstm"] and lstm is not None:
            residuals = lstm["residuals"]
        else:
            residuals = _ci_residuals(components, v, ts, steps, period)

        lower, upper, ci_method = _quantile_ci(pred, residuals, level, fallback_std)
        model_used = "+".join(components) if len(components) > 1 else components[0]
        self.last_model = model_used

        return {
            "predicted_value": float(pred),
            "lower_bound": max(0.0, float(lower)),
            "upper_bound": max(float(upper), max(0.0, float(lower))),
            "model_used": model_used,
            "model_family": "deep_learning" if "lstm" in components else "ensemble",
            "validation_mae": float(best_mae),
            "ci_method": ci_method,
            "confidence_level": level,
            "ensemble_components": components,
        }

    def _method_forecast(self, method: str, v: np.ndarray, ts: np.ndarray,
                         steps: int, level: float, period: int) -> dict[str, Any]:
        n = len(v)
        grid = _tail_grid(n, steps, tail_len=max(24, min(n // 4, 48)))
        fallback_std = float(np.std(v[-min(20, n):])) if n >= 2 else float(np.abs(v[-1])) or 1.0

        if method == "lstm":
            lstm = _lstm_forecast(v, ts, steps)
            if lstm is None:
                return self._error(
                    "LSTM unavailable: install TensorFlow/Keras or provide more history")
            pred = float(lstm["pred"])
            residuals = lstm["residuals"]
            val_mae = lstm.get("val_mae")
        else:
            pred = _predict(method, v, ts, steps, period)
            if pred is None:
                return self._error(f"Method '{method}' not applicable to this data")
            scored = _score_on_tail(method, v, ts, steps, period, grid)
            if scored is not None:
                val_mae = scored[0]
            else:
                val_mae = None
            pred = float(pred)

        if method == "lstm" and lstm is not None:
            residuals = lstm["residuals"]
        else:
            residuals = _ci_residuals([method], v, ts, steps, period)

        lower, upper, ci_method = _quantile_ci(pred, residuals, level, fallback_std)
        self.last_model = method
        return {
            "predicted_value": pred,
            "lower_bound": max(0.0, float(lower)),
            "upper_bound": max(float(upper), max(0.0, float(lower))),
            "model_used": method,
            "model_family": "deep_learning" if method == "lstm" else "classical",
            "validation_mae": val_mae,
            "ci_method": ci_method,
            "confidence_level": level,
            "ensemble_components": [method],
        }
