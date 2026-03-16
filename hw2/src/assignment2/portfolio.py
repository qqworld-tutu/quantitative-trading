import math

import numpy as np


def buy_and_hold_curve(prices, initial_capital):
    if not prices:
        return []
    base = prices[0]
    return [float(initial_capital * price / base) for price in prices]


def equal_weight_curve(price_map, initial_capital):
    names = list(price_map)
    if not names:
        return []
    per_asset = initial_capital / len(names)
    curves = [buy_and_hold_curve(price_map[name], per_asset) for name in names]
    return [float(sum(row)) for row in zip(*curves)]


def daily_returns(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return []
    return (arr[1:] / arr[:-1] - 1).tolist()


def annualized_return(values):
    if len(values) < 2:
        return 0.0
    years = max((len(values) - 1) / 252.0, 1e-9)
    return float((values[-1] / values[0]) ** (1 / years) - 1)


def annualized_vol(values):
    rets = daily_returns(values)
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1) * math.sqrt(252))


def sharpe_ratio(values):
    vol = annualized_vol(values)
    if vol == 0:
        return 0.0
    return annualized_return(values) / vol


def max_drawdown(values):
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        drawdown = value / peak - 1
        worst = min(worst, drawdown)
    return float(worst)


def yearly_summary(dates, values):
    rets = daily_returns(values)
    grouped = {}
    for dt, ret in zip(dates[1:], rets):
        grouped.setdefault(dt.year, []).append(ret)
    rows = []
    for year, year_rets in sorted(grouped.items()):
        rows.append([year, float(np.mean(year_rets)), float(np.std(year_rets, ddof=1)) if len(year_rets) > 1 else 0.0])
    return rows


def performance_row(name, values):
    return [
        name,
        round(annualized_return(values), 6),
        round(annualized_vol(values), 6),
        round(sharpe_ratio(values), 6),
        round(max_drawdown(values), 6),
    ]

