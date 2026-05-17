from __future__ import annotations

import pandas as pd


SIGNAL_FAMILY = {
    "MA gap (20)": "technical_indicator",
    "MACD hist": "technical_indicator",
    "RSI centered": "technical_indicator",
    "KDJ J centered": "technical_indicator",
    "Kalman slope": "causal_filter",
}


def signal_series(features: pd.DataFrame) -> dict[str, pd.Series]:
    close = features["close"]
    return {
        "MA gap (20)": close / features["sma20"] - 1,
        "MACD hist": features["macd_hist"] / close,
        "RSI centered": (features["rsi14"] - 50) / 50,
        "KDJ J centered": (features["kdj_j"] - 50) / 50,
        "Kalman slope": features["kalman"].pct_change(5),
    }


def method_catalog() -> pd.DataFrame:
    rows = [
        ("MA gap (20)", "technical_indicator", "Price deviation from 20-minute moving average", "trend or contrarian"),
        ("MACD hist", "technical_indicator", "Difference between MACD and signal line", "trend or reversal near extremes"),
        ("RSI centered", "technical_indicator", "Recent upside movement as a share of total movement", "contrarian near overbought/oversold"),
        ("KDJ J centered", "technical_indicator", "Current close location in recent high-low range", "contrarian near range extremes"),
        ("Kalman slope", "causal_filter", "Slope of estimated hidden price state", "trend"),
        ("HP trend", "ex_post_filter", "Penalized second-difference smoother", "visual trend decomposition"),
    ]
    return pd.DataFrame(rows, columns=["method", "family", "definition", "typical_use"])
