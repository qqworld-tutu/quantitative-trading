from __future__ import annotations

import numpy as np
import pandas as pd

from hw6_analysis.signals import SIGNAL_FAMILY, signal_series


def evaluate_signals(features: pd.DataFrame, horizons: list[int], main_horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = signal_series(features)
    log_close = features["log_close"]
    summary_rows: list[dict[str, float | int | str]] = []
    horizon_rows: list[dict[str, float | int | str]] = []

    for horizon in horizons:
        future_ret = log_close.shift(-horizon) - log_close
        for name, signal in signals.items():
            joined = pd.concat([signal.rename("signal"), future_ret.rename("future_ret")], axis=1).dropna()
            joined = joined[joined["signal"] != 0]
            if joined.empty:
                continue
            pearson = float(joined["signal"].corr(joined["future_ret"]))
            spearman = float(joined["signal"].rank().corr(joined["future_ret"].rank()))
            raw_hit = float((np.sign(joined["signal"]) == np.sign(joined["future_ret"])).mean())
            side = "trend_following" if raw_hit >= 0.5 else "contrarian"
            usable_hit = max(raw_hit, 1 - raw_hit)

            row = {
                "signal": name,
                "family": SIGNAL_FAMILY[name],
                "horizon_minutes": horizon,
                "n_obs": int(len(joined)),
                "pearson_ic": pearson,
                "spearman_ic": spearman,
                "preferred_direction": side,
                "usable_direction_accuracy": usable_hit,
            }
            horizon_rows.append(row)
            if horizon == main_horizon:
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("pearson_ic")
    horizon_summary = pd.DataFrame(horizon_rows)
    return summary, horizon_summary


def horizon_best_summary(horizon_summary: pd.DataFrame) -> pd.DataFrame:
    table = horizon_summary.copy()
    table["abs_pearson_ic"] = table["pearson_ic"].abs()
    return (
        table.sort_values(["horizon_minutes", "abs_pearson_ic"], ascending=[True, False])
        .groupby("horizon_minutes", as_index=False)
        .head(3)
        .reset_index(drop=True)
    )
