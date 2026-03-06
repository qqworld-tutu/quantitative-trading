from datetime import datetime

from assignment1_part2.advanced import equal_volume_bins, intraday_session_returns, resample_last


def test_resample_last_keeps_period_end_price():
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 8),
        datetime(2024, 1, 9),
    ]
    prices = [10.0, 11.0, 12.0, 13.0]
    result = resample_last(dates, prices, "W")
    assert [row[1] for row in result] == [11.0, 13.0]


def test_intraday_session_returns_split_morning_and_afternoon():
    rows = [
        {"datetime": datetime(2024, 1, 2, 9, 30), "open": 10.0, "close": 10.2},
        {"datetime": datetime(2024, 1, 2, 11, 30), "open": 10.2, "close": 10.4},
        {"datetime": datetime(2024, 1, 2, 13, 0), "open": 10.4, "close": 10.5},
        {"datetime": datetime(2024, 1, 2, 15, 0), "open": 10.5, "close": 10.8},
    ]
    result = intraday_session_returns(rows)
    assert len(result) == 1
    assert result[0]["date"].strftime("%Y-%m-%d") == "2024-01-02"
    assert result[0]["morning_return"] > 0
    assert result[0]["afternoon_return"] > 0


def test_equal_volume_bins_produces_last_close_per_bucket():
    rows = [
        {"close": 10.0, "volume": 2},
        {"close": 10.2, "volume": 1},
        {"close": 10.1, "volume": 3},
        {"close": 10.4, "volume": 2},
    ]
    closes = equal_volume_bins(rows, bucket_count=2)
    assert closes == [10.1, 10.4]

