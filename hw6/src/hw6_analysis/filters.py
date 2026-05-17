from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


def kalman_filter_1d(series: pd.Series, process_var: float = 1.0, measurement_var: float = 400.0) -> pd.Series:
    observations = series.to_numpy(dtype=float)
    state = np.zeros_like(observations)
    variance = np.zeros_like(observations)
    state[0] = observations[0]
    variance[0] = 1.0

    for idx in range(1, len(observations)):
        pred_state = state[idx - 1]
        pred_var = variance[idx - 1] + process_var
        gain = pred_var / (pred_var + measurement_var)
        state[idx] = pred_state + gain * (observations[idx] - pred_state)
        variance[idx] = (1 - gain) * pred_var

    return pd.Series(state, index=series.index, name="kalman")


def hp_trend(series: pd.Series, lamb: float = 1e5) -> pd.Series:
    values = series.to_numpy(dtype=float)
    n = len(values)
    identity = sparse.eye(n, format="csc")
    diff = sparse.diags(
        diagonals=[np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    trend = spsolve(identity + lamb * (diff.T @ diff), values)
    return pd.Series(trend, index=series.index, name="hp_trend")


def add_causal_filters(features: pd.DataFrame) -> pd.DataFrame:
    close = features["close"]
    features["kalman"] = kalman_filter_1d(close)
    return features
