from assignment1_part2.data import choose_stock_universe, excel_to_datetime, log_price_series


def test_excel_to_datetime_supports_daily_and_intraday():
    assert excel_to_datetime(37986).strftime("%Y-%m-%d") == "2003-12-31"
    assert excel_to_datetime(44642.395833333336).strftime("%Y-%m-%d %H:%M") == "2022-03-22 09:30"


def test_log_price_series_skips_non_positive_values():
    dates = ["a", "b", "c", "d"]
    prices = [0.0, 10.0, 10.5, 0.0]
    clean_dates, clean_prices = log_price_series(dates, prices)
    assert clean_dates == ["b", "c"]
    assert clean_prices == [10.0, 10.5]


def test_choose_stock_universe_uses_full_history_and_wind_industry():
    stock_info = [
        {"代码": "A", "权重": 5.0, "Wind一级行业": "金融", "申银万国一级行业": "银行"},
        {"代码": "B", "权重": 4.0, "Wind一级行业": "金融", "申银万国一级行业": "非银金融"},
        {"代码": "C", "权重": 3.0, "Wind一级行业": "消费", "申银万国一级行业": "食品饮料"},
    ]
    stock_table = {
        "dates": ["d1", "d2", "d3"],
        "series": {
            "A": {"name": "A股", "prices": [1.0, 2.0, 3.0]},
            "B": {"name": "B股", "prices": [0.0, 2.0, 3.0]},
            "C": {"name": "C股", "prices": [1.0, 2.0, 3.0]},
        },
    }
    chosen = choose_stock_universe(stock_info, stock_table, n=2)
    assert [row["code"] for row in chosen] == ["A", "C"]
