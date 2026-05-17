from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import pandas as pd


def load_future_dict(path: Path) -> dict[str, pd.DataFrame]:
    """Read the provided pickle dictionary directly from the zip archive."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("future_data.pkl", "r") as handle:
            return pickle.load(handle)


def load_contract(path: Path, symbol: str) -> pd.DataFrame:
    data = load_future_dict(path)
    if symbol not in data:
        available = ", ".join(list(data)[:10])
        raise KeyError(f"{symbol} is not in the data dictionary. First keys: {available}")
    return data[symbol].sort_index()


def sample_info(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    close = df["close"].dropna()
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "rows_raw": len(df),
                "rows_non_missing_close": len(close),
                "start": close.index.min(),
                "end": close.index.max(),
                "mean_volume": float(df["volume"].fillna(0).mean()),
                "missing_close_ratio": float(df["close"].isna().mean()),
            }
        ]
    )
