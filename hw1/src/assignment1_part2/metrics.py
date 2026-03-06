import math

import numpy as np


def log_returns(prices):
    values = np.asarray(prices, dtype=float)
    if len(values) < 2:
        return []
    return np.diff(np.log(values)).tolist()


def lag1_autocorr(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or np.std(arr) == 0:
        return 0.0
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def summary_stats(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {key: 0.0 for key in ("mean", "median", "std", "skew", "kurtosis", "min", "max", "autocorr1")}
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if std == 0:
        skew = 0.0
        kurtosis = 0.0
    else:
        z = (arr - mean) / std
        skew = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "autocorr1": lag1_autocorr(arr),
    }


def histogram(values, bins=30):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return counts.tolist(), edges.tolist()


def normal_curve(values, points=120):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return [], []
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    mean = float(np.mean(arr))
    if std == 0:
        return [mean], [1.0]
    xs = np.linspace(float(np.min(arr)), float(np.max(arr)), points)
    scale = 1 / (std * math.sqrt(2 * math.pi))
    ys = scale * np.exp(-0.5 * ((xs - mean) / std) ** 2)
    return xs.tolist(), ys.tolist()

