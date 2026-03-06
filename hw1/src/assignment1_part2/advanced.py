from collections import defaultdict

import numpy as np

from assignment1_part2.data import to_float
from assignment1_part2.metrics import log_returns


def _period_key(dt, freq):
    if freq == "D":
        return (dt.year, dt.month, dt.day)
    if freq == "W":
        iso = dt.isocalendar()
        return (iso.year, iso.week)
    if freq == "M":
        return (dt.year, dt.month)
    raise ValueError(f"unsupported freq: {freq}")


def resample_last(dates, prices, freq):
    groups = {}
    for dt, price in zip(dates, prices):
        groups[_period_key(dt, freq)] = (dt, price)
    return [groups[key] for key in sorted(groups)]


def overnight_intraday_returns(rows):
    result = []
    previous_close = None
    for row in rows:
        dt = row["datetime"]
        open_price = to_float(row["open"])
        close_price = to_float(row["close"])
        if open_price <= 0 or close_price <= 0:
            continue
        overnight = np.log(open_price / previous_close) if previous_close else None
        intraday = np.log(close_price / open_price)
        result.append({"date": dt.date(), "overnight_return": overnight, "intraday_return": intraday})
        previous_close = close_price
    return result


def intraday_session_returns(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["datetime"].date()].append(row)
    result = []
    for day, day_rows in sorted(grouped.items()):
        day_rows.sort(key=lambda row: row["datetime"])
        morning = [row for row in day_rows if row["datetime"].hour < 12]
        afternoon = [row for row in day_rows if row["datetime"].hour >= 12]
        if not morning or not afternoon:
            continue
        morning_return = np.log(to_float(morning[-1]["close"]) / to_float(morning[0]["open"]))
        afternoon_return = np.log(to_float(afternoon[-1]["close"]) / to_float(afternoon[0]["open"]))
        result.append({"date": day, "morning_return": morning_return, "afternoon_return": afternoon_return})
    return result


def equal_volume_bins(rows, bucket_count=20):
    total_volume = sum(max(to_float(row["volume"]), 0.0) for row in rows)
    if total_volume <= 0 or bucket_count <= 0:
        return []
    target = total_volume / bucket_count
    bins = []
    running = 0.0
    for row in rows:
        running += max(to_float(row["volume"]), 0.0)
        if running >= target:
            bins.append(to_float(row["close"]))
            running = 0.0
    if rows and (not bins or bins[-1] != to_float(rows[-1]["close"])):
        bins.append(to_float(rows[-1]["close"]))
    return bins[:bucket_count]


def weekday_effect(dates, returns):
    groups = defaultdict(list)
    for dt, value in zip(dates[1:], returns):
        groups[dt.weekday()].append(value)
    return [(weekday, float(np.mean(values))) for weekday, values in sorted(groups.items())]


def month_effect(dates, returns):
    groups = defaultdict(list)
    for dt, value in zip(dates[1:], returns):
        groups[dt.month].append(value)
    return [(month, float(np.mean(values))) for month, values in sorted(groups.items())]


def holiday_gap_effect(dates, returns):
    result = []
    for idx in range(1, len(dates) - 1):
        gap = (dates[idx] - dates[idx - 1]).days
        if gap >= 4:
            result.append((dates[idx].strftime("%Y-%m-%d"), returns[idx - 1]))
    return result


def intraday_seasonality(rows, slot_minutes=30):
    groups = defaultdict(list)
    for row in rows:
        dt = row["datetime"]
        slot = (dt.hour, (dt.minute // slot_minutes) * slot_minutes)
        open_price = to_float(row["open"])
        close_price = to_float(row["close"])
        if open_price > 0 and close_price > 0:
            groups[slot].append(np.log(close_price / open_price))
    return [(f"{hour:02d}:{minute:02d}", float(np.mean(values))) for (hour, minute), values in sorted(groups.items())]


def aligned_return_correlation(named_series):
    maps = {}
    for name, (dates, prices) in named_series.items():
        returns = log_returns(prices)
        maps[name] = {dates[idx + 1].date(): returns[idx] for idx in range(len(returns))}
    common_dates = set.intersection(*(set(item.keys()) for item in maps.values()))
    ordered_names = list(named_series)
    matrix = []
    for left in ordered_names:
        row = []
        left_values = [maps[left][dt] for dt in sorted(common_dates)]
        for right in ordered_names:
            right_values = [maps[right][dt] for dt in sorted(common_dates)]
            if not left_values or np.std(left_values) == 0 or np.std(right_values) == 0:
                row.append(0.0)
            else:
                row.append(float(np.corrcoef(left_values, right_values)[0, 1]))
        matrix.append(row)
    return ordered_names, matrix

