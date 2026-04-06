import csv
import io
from dataclasses import dataclass
from zipfile import ZipFile

from assignment4.config import DATA_ZIP


@dataclass
class MarketData:
    dates: list
    codes: list
    close_by_date: dict
    pb_ratio_by_date: dict
    paused_by_date: dict
    daily_return_by_date: dict
    month_end_dates: list
    date_to_index: dict


def parse_float(text):
    if text in (None, ""):
        return None
    return float(text)


def find_month_ends(dates):
    month_ends = []
    for idx, date_text in enumerate(dates):
        if idx == len(dates) - 1 or dates[idx + 1][:7] != date_text[:7]:
            month_ends.append(date_text)
    return month_ends


def compute_daily_returns(dates, codes, close_by_date):
    daily_return_by_date = {}
    for idx in range(1, len(dates)):
        current = dates[idx]
        previous = dates[idx - 1]
        current_close = close_by_date[current]
        previous_close = close_by_date[previous]
        row = {}
        for code in codes:
            cur = current_close.get(code)
            prev = previous_close.get(code)
            if cur is None or prev is None or cur <= 0 or prev <= 0:
                row[code] = None
            else:
                row[code] = cur / prev - 1.0
        daily_return_by_date[current] = row
    return daily_return_by_date


def load_market_data(zip_path=DATA_ZIP):
    with ZipFile(zip_path) as zf:
        members = sorted(name for name in zf.namelist() if name.endswith(".csv"))
        dates = []
        close_by_date = {}
        pb_ratio_by_date = {}
        paused_by_date = {}
        codes = []
        for pos, member in enumerate(members):
            date_text = member.split("/")[-1].replace(".csv", "")
            dates.append(date_text)
            close_row = {}
            pb_row = {}
            paused_row = {}
            with zf.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                current_codes = []
                for row in reader:
                    code = row["code"]
                    current_codes.append(code)
                    close_row[code] = parse_float(row["close"])
                    pb_row[code] = parse_float(row["pb_ratio"])
                    paused_row[code] = parse_float(row["paused"]) or 0.0
            if pos == 0:
                codes = current_codes
            close_by_date[date_text] = close_row
            pb_ratio_by_date[date_text] = pb_row
            paused_by_date[date_text] = paused_row
    date_to_index = {date_text: idx for idx, date_text in enumerate(dates)}
    daily_return_by_date = compute_daily_returns(dates, codes, close_by_date)
    return MarketData(
        dates=dates,
        codes=codes,
        close_by_date=close_by_date,
        pb_ratio_by_date=pb_ratio_by_date,
        paused_by_date=paused_by_date,
        daily_return_by_date=daily_return_by_date,
        month_end_dates=find_month_ends(dates),
        date_to_index=date_to_index,
    )
