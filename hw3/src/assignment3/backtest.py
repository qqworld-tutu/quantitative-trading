import math

import numpy as np


def apply_weights_for_period(start_value, returns, weights):
    values = []
    current = float(start_value)
    for row in returns:
        current *= 1 + float(np.dot(row, weights))
        values.append(current)
    return values


def portfolio_curve(periods, initial_capital):
    values = [float(initial_capital)]
    current = float(initial_capital)
    for period in periods:
        period_values = apply_weights_for_period(current, period["returns"], period["weights"])
        values.extend(period_values)
        current = values[-1]
    return values


def daily_returns(values):
    arr = np.asarray(values, dtype=float)
    return arr[1:] / arr[:-1] - 1


def annualized_return(values):
    years = max((len(values) - 1) / 252.0, 1e-9)
    return float((values[-1] / values[0]) ** (1 / years) - 1)


def annualized_vol(values):
    rets = daily_returns(values)
    return float(np.std(rets, ddof=1) * math.sqrt(252)) if len(rets) > 1 else 0.0


def sharpe_ratio(values, rf_daily=0.0):
    rets = daily_returns(values)
    if len(rets) < 2:
        return 0.0
    excess = rets - rf_daily
    vol = np.std(excess, ddof=1)
    if vol == 0:
        return 0.0
    return float(np.mean(excess) / vol * math.sqrt(252))


def max_drawdown(values):
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return float(worst)


def monthly_stats(dates, values):
    rets = daily_returns(values)
    grouped = {}
    for dt, ret in zip(dates[1:], rets):
        key = f"{dt.year:04d}-{dt.month:02d}"
        grouped.setdefault(key, []).append(float(ret))
    rows = []
    for key, vals in sorted(grouped.items()):
        rows.append([key, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0])
    return rows
