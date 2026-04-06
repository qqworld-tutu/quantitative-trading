import math

from assignment4.config import MOMENTUM_WINDOW, WINSORIZE_PCT


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return sorted_values[left]
    weight = position - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def winsorize_map(value_map, pct=WINSORIZE_PCT):
    if not value_map:
        return {}
    ordered = sorted(value_map.values())
    lower = percentile(ordered, pct)
    upper = percentile(ordered, 1.0 - pct)
    adjusted = {}
    for code, value in value_map.items():
        adjusted[code] = min(max(value, lower), upper)
    return adjusted


def zscore_map(value_map):
    if not value_map:
        return {}
    values = list(value_map.values())
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = math.sqrt(variance)
    if std_value == 0:
        return {code: 0.0 for code in value_map}
    return {code: (value - mean_value) / std_value for code, value in value_map.items()}


def preprocess_factor(raw_map):
    return zscore_map(winsorize_map(raw_map))


def build_factor_panels(data):
    eligible_dates = []
    factor_panels = {
        "momentum20": {},
        "bp": {},
    }
    month_ends = data.month_end_dates
    for month_idx in range(len(month_ends) - 1):
        date_text = month_ends[month_idx]
        date_pos = data.date_to_index[date_text]
        if date_pos < MOMENTUM_WINDOW:
            continue
        eligible_dates.append(date_text)
        lookback_date = data.dates[date_pos - MOMENTUM_WINDOW]
        raw_momentum = {}
        raw_bp = {}
        paused_row = data.paused_by_date[date_text]
        close_row = data.close_by_date[date_text]
        close_lookback = data.close_by_date[lookback_date]
        pb_row = data.pb_ratio_by_date[date_text]
        for code in data.codes:
            if paused_row.get(code, 0.0) >= 1.0:
                continue
            close_now = close_row.get(code)
            close_then = close_lookback.get(code)
            if close_now is not None and close_then is not None and close_now > 0 and close_then > 0:
                raw_momentum[code] = close_now / close_then - 1.0
            pb_ratio = pb_row.get(code)
            if pb_ratio is not None and pb_ratio > 0:
                raw_bp[code] = 1.0 / pb_ratio
        factor_panels["momentum20"][date_text] = {
            "raw": raw_momentum,
            "signal": preprocess_factor(raw_momentum),
        }
        factor_panels["bp"][date_text] = {
            "raw": raw_bp,
            "signal": preprocess_factor(raw_bp),
        }
    return eligible_dates, factor_panels
