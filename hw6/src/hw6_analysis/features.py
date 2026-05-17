from __future__ import annotations

import numpy as np
import pandas as pd

from hw6_analysis.filters import add_causal_filters
from hw6_analysis.indicators import add_technical_indicators


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[["open", "close", "high", "low", "volume", "money"]].dropna(subset=["close"]).copy()
    features = pd.DataFrame(index=df.index)
    for column in ["open", "close", "high", "low", "volume", "money"]:
        features[column] = df[column]
    features["log_close"] = np.log(features["close"])
    features = add_technical_indicators(features)
    features = add_causal_filters(features)
    return features
