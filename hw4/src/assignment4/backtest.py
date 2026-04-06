import math

from assignment4.config import TOP_SELECTION_RATIO, TRADING_DAYS_PER_YEAR, split_name


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def rank_values(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[pos][1]:
            end += 1
        average_rank = (pos + end) / 2.0 + 1.0
        for inner in range(pos, end + 1):
            ranks[indexed[inner][0]] = average_rank
        pos = end + 1
    return ranks


def pearson_corr(x_values, y_values):
    if len(x_values) < 2 or len(y_values) < 2:
        return None
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered))
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0 or y_norm == 0:
        return None
    return numerator / (x_norm * y_norm)


def spearman_corr(x_values, y_values):
    return pearson_corr(rank_values(x_values), rank_values(y_values))


def top_quantile_weights(signal_map, top_ratio=TOP_SELECTION_RATIO):
    ranked = sorted(signal_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    if not ranked:
        return {}
    count = max(1, int(len(ranked) * top_ratio))
    selected = ranked[:count]
    weight = 1.0 / count
    return {code: weight for code, _ in selected}


def cumulative_sum(rows):
    total = 0.0
    result = []
    for label, value in rows:
        total += value
        result.append((label, total))
    return result


def nav_curve(daily_returns, initial_capital=1.0):
    nav = initial_capital
    curve = []
    for date_text, daily_return in daily_returns:
        nav *= 1.0 + daily_return
        curve.append((date_text, nav))
    return curve


def annualized_return(daily_returns):
    if not daily_returns:
        return 0.0
    nav = 1.0
    for _, daily_return in daily_returns:
        nav *= 1.0 + daily_return
    return nav ** (TRADING_DAYS_PER_YEAR / len(daily_returns)) - 1.0


def annualized_vol(daily_returns):
    return stdev([value for _, value in daily_returns]) * math.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(daily_returns):
    vol = annualized_vol(daily_returns)
    if vol == 0:
        return 0.0
    return annualized_return(daily_returns) / vol


def max_drawdown(daily_returns):
    peak = 1.0
    nav = 1.0
    drawdown = 0.0
    for _, daily_return in daily_returns:
        nav *= 1.0 + daily_return
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
    return drawdown


def compute_forward_returns(data, rebalance_dates):
    forward_returns = {}
    for current_date, next_date in zip(rebalance_dates, rebalance_dates[1:]):
        current_close = data.close_by_date[current_date]
        next_close = data.close_by_date[next_date]
        row = {}
        for code in data.codes:
            start = current_close.get(code)
            end = next_close.get(code)
            if start is None or end is None or start <= 0 or end <= 0:
                continue
            row[code] = end / start - 1.0
        forward_returns[current_date] = row
    return forward_returns


def compute_ic_series(signal_panels, forward_returns):
    ic_rows = []
    for date_text, forward_row in forward_returns.items():
        signal_row = signal_panels[date_text]["signal"]
        common_codes = [code for code in signal_row if code in forward_row]
        x_values = [signal_row[code] for code in common_codes]
        y_values = [forward_row[code] for code in common_codes]
        ic_value = spearman_corr(x_values, y_values)
        ic_rows.append((date_text, ic_value if ic_value is not None else 0.0))
    return ic_rows


def compute_weight_panels(signal_panels, rebalance_dates):
    weight_panels = {}
    for date_text in rebalance_dates:
        weight_panels[date_text] = top_quantile_weights(signal_panels[date_text]["signal"])
    return weight_panels


def compute_daily_strategy_returns(data, weight_panels, rebalance_dates):
    returns = []
    for current_date, next_date in zip(rebalance_dates, rebalance_dates[1:]):
        weights = weight_panels[current_date]
        current_idx = data.date_to_index[current_date]
        next_idx = data.date_to_index[next_date]
        for idx in range(current_idx + 1, next_idx + 1):
            date_text = data.dates[idx]
            daily_row = data.daily_return_by_date[date_text]
            portfolio_return = 0.0
            for code, weight in weights.items():
                value = daily_row.get(code)
                if value is None:
                    continue
                portfolio_return += weight * value
            returns.append((date_text, portfolio_return))
    return returns


def summarize_performance(daily_returns, ic_rows):
    by_split_daily = {"train": [], "validation": [], "test": []}
    by_split_ic = {"train": [], "validation": [], "test": []}
    for date_text, value in daily_returns:
        by_split_daily[split_name(date_text)].append((date_text, value))
    for date_text, value in ic_rows:
        by_split_ic[split_name(date_text)].append(value)
    summaries = []
    for split in ("train", "validation", "test"):
        split_daily = by_split_daily[split]
        split_ic = by_split_ic[split]
        ic_mean = mean(split_ic) if split_ic else 0.0
        ic_std = stdev(split_ic) if len(split_ic) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std else 0.0
        summaries.append(
            {
                "split": split,
                "days": len(split_daily),
                "avg_daily_return": mean([value for _, value in split_daily]) if split_daily else 0.0,
                "daily_vol": stdev([value for _, value in split_daily]) if len(split_daily) > 1 else 0.0,
                "annualized_return": annualized_return(split_daily),
                "annualized_vol": annualized_vol(split_daily),
                "sharpe": sharpe_ratio(split_daily),
                "max_drawdown": max_drawdown(split_daily),
                "ic_mean": ic_mean,
                "ic_ir": ic_ir,
            }
        )
    return summaries
