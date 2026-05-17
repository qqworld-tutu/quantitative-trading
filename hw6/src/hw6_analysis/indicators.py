from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(features: pd.DataFrame) -> pd.DataFrame:
    close = features["close"]

    features["sma20"] = close.rolling(20).mean()
    features["sma60"] = close.rolling(60).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    features["rsi14"] = 100 - 100 / (1 + rs)

    low_9 = features["low"].rolling(9).min()
    high_9 = features["high"].rolling(9).max()
    rsv = 100 * (close - low_9) / (high_9 - low_9).replace(0, np.nan)
    features["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    features["kdj_d"] = features["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    features["kdj_j"] = 3 * features["kdj_k"] - 2 * features["kdj_d"]
    return features
