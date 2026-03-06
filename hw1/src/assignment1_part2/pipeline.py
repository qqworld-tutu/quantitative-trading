from pathlib import Path

import numpy as np

from assignment1_part2.advanced import (
    aligned_return_correlation,
    equal_volume_bins,
    holiday_gap_effect,
    intraday_seasonality,
    intraday_session_returns,
    month_effect,
    overnight_intraday_returns,
    resample_last,
    weekday_effect,
)
from assignment1_part2.config import DATA_ZIP, OUTPUT_DIR
from assignment1_part2.data import (
    choose_stock_universe,
    excel_to_datetime,
    log_price_series,
    read_rows,
    read_wide_table,
    to_float,
)
from assignment1_part2.metrics import histogram, log_returns, normal_curve, summary_stats
from assignment1_part2.reporting import ensure_dirs, histogram_svg, line_svg, markdown_table, write_csv, write_markdown


def _series_from_rows(rows):
    rows = [row for row in rows if row.get("日期") not in (None, "")]
    dates = [excel_to_datetime(row["日期"]) for row in rows]
    prices = [to_float(row["收盘价(元)"]) for row in rows]
    return log_price_series(dates, prices)


def _intraday_rows(rows):
    result = []
    for row in rows:
        if row.get("日期") in (None, ""):
            continue
        dt = excel_to_datetime(row["日期"])
        result.append(
            {
                "datetime": dt,
                "open": to_float(row["开盘价(元)"]),
                "close": to_float(row["收盘价(元)"]),
                "volume": to_float(row.get("成交量") or row.get("成交量(股)") or 0.0),
            }
        )
    return result


def _daily_rows(rows):
    result = []
    for row in rows:
        if row.get("日期") in (None, ""):
            continue
        result.append(
            {
                "datetime": excel_to_datetime(row["日期"]),
                "open": to_float(row["开盘价(元)"]),
                "close": to_float(row["收盘价(元)"]),
            }
        )
    return result


def _representative_option(option_table):
    best_code = None
    best_score = -1.0
    for code, item in option_table["series"].items():
        prices = [price for price in item["prices"] if price > 0]
        score = float(np.mean(prices)) if prices else -1.0
        if score > best_score:
            best_code = code
            best_score = score
    return best_code, option_table["series"][best_code]


def _analyze_asset(name, dates, prices, output_root):
    dates, prices = log_price_series(dates, prices)
    returns = log_returns(prices)
    stats = summary_stats(returns)
    highs = max(range(len(prices)), key=lambda idx: prices[idx])
    lows = min(range(len(prices)), key=lambda idx: prices[idx])
    prefix = name.replace("/", "_").replace(" ", "_")
    line_svg(output_root / "figures" / f"{prefix}_price.svg", f"{name} 价格", prices)
    counts, edges = histogram(returns)
    normal_x, normal_y = normal_curve(returns)
    histogram_svg(output_root / "figures" / f"{prefix}_return_hist.svg", f"{name} 收益率直方图", counts, edges, normal_x, normal_y)
    return {
        "name": name,
        "dates": dates,
        "prices": prices,
        "returns": returns,
        "stats": stats,
        "high_date": dates[highs].strftime("%Y-%m-%d"),
        "low_date": dates[lows].strftime("%Y-%m-%d"),
    }


def _format_stats_rows(asset_results):
    rows = []
    for item in asset_results:
        stats = item["stats"]
        rows.append(
            [
                item["name"],
                round(stats["mean"], 6),
                round(stats["median"], 6),
                round(stats["std"], 6),
                round(stats["skew"], 6),
                round(stats["kurtosis"], 6),
                round(stats["min"], 6),
                round(stats["max"], 6),
                round(stats["autocorr1"], 6),
                item["high_date"],
                item["low_date"],
            ]
        )
    return rows


def _advanced_tables(output_root, data):
    tables = {}
    table_headers = {
        "多频率比较": ["频率", "样本数", "均值", "标准差"],
        "隔夜与日内收益": ["样本数", "隔夜均值", "日内均值"],
        "上下午收益差异": ["样本数", "上午均值", "下午均值"],
        "等交易量间距": ["采样", "样本数", "标准差"],
        "股票特征关系": ["股票", "行业", "总市值(亿)", "均值", "标准差"],
        "期权横截面关系": ["期权", "样本数", "均值", "标准差"],
        "日历效应_周": ["星期", "平均收益"],
        "日历效应_月": ["月份", "平均收益"],
        "节日效应": ["节后首日", "收益"],
        "日内季节效应": ["时段", "平均收益"],
    }

    hs300_index = data["indices"]["沪深300指数"]
    rows = []
    for freq in ("D", "W", "M"):
        sampled = resample_last(hs300_index["dates"], hs300_index["prices"], freq)
        prices = [price for _, price in sampled]
        returns = log_returns(prices)
        stats = summary_stats(returns)
        rows.append([freq, len(prices), round(stats["mean"], 6), round(stats["std"], 6)])
    tables["多频率比较"] = rows

    etf_daily = overnight_intraday_returns(data["etf_daily_rows"])
    tables["隔夜与日内收益"] = [
        [
            len([row for row in etf_daily if row["overnight_return"] is not None]),
            round(np.mean([row["overnight_return"] for row in etf_daily if row["overnight_return"] is not None]), 6),
            round(np.mean([row["intraday_return"] for row in etf_daily]), 6),
        ]
    ]

    sessions = intraday_session_returns(data["etf_intraday_rows"])
    tables["上下午收益差异"] = [
        [len(sessions), round(np.mean([row["morning_return"] for row in sessions]), 6), round(np.mean([row["afternoon_return"] for row in sessions]), 6)]
    ]

    volume_prices = equal_volume_bins(data["etf_intraday_rows"], bucket_count=20)
    volume_returns = log_returns(volume_prices)
    clock_prices = [row["close"] for row in data["etf_intraday_rows"][: len(volume_prices) + 1]]
    clock_returns = log_returns(clock_prices)
    tables["等交易量间距"] = [
        ["clock", len(clock_returns), round(summary_stats(clock_returns)["std"], 6)],
        ["volume", len(volume_returns), round(summary_stats(volume_returns)["std"], 6)],
    ]

    relation_names, relation_matrix = aligned_return_correlation(
        {
            "上证50指数": (data["indices"]["上证50指数"]["dates"], data["indices"]["上证50指数"]["prices"]),
            "上证50ETF": (data["etf_daily"]["dates"], data["etf_daily"]["prices"]),
            "上证50期货": (data["futures"]["上证50期货"]["dates"], data["futures"]["上证50期货"]["prices"]),
        }
    )
    write_csv(output_root / "tables" / "指数ETF期货关系.csv", ["资产"] + relation_names, [[name] + [round(value, 6) for value in row] for name, row in zip(relation_names, relation_matrix)])

    feature_rows = []
    stock_info = {row["代码"]: row for row in data["stock_info"]}
    for stock in data["chosen_stocks"]:
        returns = log_returns(data["stocks"][stock["code"]]["prices"])
        feature_rows.append(
            [
                stock["name"],
                stock["industry"],
                round(to_float(stock_info[stock["code"]]["总市值(亿)"]), 2),
                round(summary_stats(returns)["mean"], 6),
                round(summary_stats(returns)["std"], 6),
            ]
        )
    tables["股票特征关系"] = feature_rows

    stock_names, stock_matrix = aligned_return_correlation(
        {stock["name"]: (data["stocks"][stock["code"]]["dates"], data["stocks"][stock["code"]]["prices"]) for stock in data["chosen_stocks"]}
    )
    write_csv(output_root / "tables" / "股票相关性.csv", ["股票"] + stock_names, [[name] + [round(value, 6) for value in row] for name, row in zip(stock_names, stock_matrix)])

    future_names, future_matrix = aligned_return_correlation(
        {name: (item["dates"], item["prices"]) for name, item in data["futures"].items()}
    )
    write_csv(output_root / "tables" / "期货相关性.csv", ["期货"] + future_names, [[name] + [round(value, 6) for value in row] for name, row in zip(future_names, future_matrix)])

    option_etf_names, option_etf_matrix = aligned_return_correlation(
        {
            "50ETF": (data["etf_from_intraday"]["dates"], data["etf_from_intraday"]["prices"]),
            data["call_option"]["name"]: (data["call_option"]["dates"], data["call_option"]["prices"]),
            data["put_option"]["name"]: (data["put_option"]["dates"], data["put_option"]["prices"]),
        }
    )
    write_csv(output_root / "tables" / "期权ETF关系.csv", ["资产"] + option_etf_names, [[name] + [round(value, 6) for value in row] for name, row in zip(option_etf_names, option_etf_matrix)])

    call_subset = list(data["call_table"]["series"].items())[:5]
    option_rows = []
    base_dates = data["call_table"]["dates"]
    for code, item in call_subset:
        daily = resample_last(base_dates, item["prices"], "D")
        prices = [price for _, price in daily if price > 0]
        returns = log_returns(prices)
        option_rows.append([item["name"], len(prices), round(summary_stats(returns)["mean"], 6), round(summary_stats(returns)["std"], 6)])
    tables["期权横截面关系"] = option_rows

    weekday_rows = weekday_effect(hs300_index["dates"], hs300_index["returns"])
    month_rows = month_effect(hs300_index["dates"], hs300_index["returns"])
    tables["日历效应_周"] = [[weekday, round(value, 6)] for weekday, value in weekday_rows]
    tables["日历效应_月"] = [[month, round(value, 6)] for month, value in month_rows]

    holiday_rows = holiday_gap_effect(hs300_index["dates"], hs300_index["returns"])
    tables["节日效应"] = [[date, round(value, 6)] for date, value in holiday_rows[:20]]

    intraday_rows = intraday_seasonality(data["etf_intraday_rows"])
    tables["日内季节效应"] = [[slot, round(value, 6)] for slot, value in intraday_rows]

    for name, rows in tables.items():
        write_csv(output_root / "tables" / f"{name}.csv", table_headers[name], rows)
    return tables, table_headers


def run(output_root=OUTPUT_DIR):
    output_root = Path(output_root)
    ensure_dirs(output_root)

    index_rows = {
        "上证50指数": read_rows(DATA_ZIP, "市场指数类/上证50指数日数据.xlsx"),
        "沪深300指数": read_rows(DATA_ZIP, "市场指数类/沪深300指数日数据.xlsx"),
        "中证500指数": read_rows(DATA_ZIP, "市场指数类/中证500指数日数据.xlsx"),
    }
    etf_daily_rows = read_rows(DATA_ZIP, "指数ETF类/上证50ETF日数据.xlsx")
    etf_intraday_rows = _intraday_rows(read_rows(DATA_ZIP, "指数ETF类/上证50ETF1分钟.xlsx"))
    stock_table = read_wide_table(DATA_ZIP, "股票类/沪深300成分股收盘价日数据.xlsx")
    stock_info = read_rows(DATA_ZIP, "股票类/沪深300成分股信息20220902.xlsx")
    future_rows = {
        "上证50期货": read_rows(DATA_ZIP, "期货类/上证50期货主力日数据.xlsx"),
        "中证500期货": read_rows(DATA_ZIP, "期货类/中证500期货主力日数据.xlsx"),
        "沪深300期货": read_rows(DATA_ZIP, "期货类/沪深300期货主力日数据.xlsx"),
    }
    call_table = read_wide_table(DATA_ZIP, "期权类/上证50ETF认购期权1分钟.xlsx")
    put_table = read_wide_table(DATA_ZIP, "期权类/上证50ETF认沽期权1分钟.xlsx")

    indices = {}
    for name, rows in index_rows.items():
        dates, prices = _series_from_rows(rows)
        indices[name] = {"dates": dates, "prices": prices, "returns": log_returns(prices)}

    etf_dates, etf_prices = _series_from_rows(etf_daily_rows)
    etf_intraday_daily = resample_last([row["datetime"] for row in etf_intraday_rows], [row["close"] for row in etf_intraday_rows], "D")
    etf_from_intraday = {"dates": [dt for dt, _ in etf_intraday_daily], "prices": [price for _, price in etf_intraday_daily]}

    futures = {}
    for name, rows in future_rows.items():
        dates, prices = _series_from_rows(rows)
        futures[name] = {"dates": dates, "prices": prices}

    chosen_stocks = choose_stock_universe(stock_info, stock_table, n=5)
    stocks = {}
    for stock in chosen_stocks:
        dates, prices = log_price_series(stock_table["dates"], stock_table["series"][stock["code"]]["prices"])
        stocks[stock["code"]] = {"name": stock["name"], "dates": dates, "prices": prices}

    call_code, call_item = _representative_option(call_table)
    put_code, put_item = _representative_option(put_table)
    call_daily = resample_last(call_table["dates"], call_item["prices"], "D")
    put_daily = resample_last(put_table["dates"], put_item["prices"], "D")

    asset_results = []
    for name, item in indices.items():
        asset_results.append(_analyze_asset(name, item["dates"], item["prices"], output_root))
    asset_results.append(_analyze_asset("上证50ETF", etf_dates, etf_prices, output_root))
    asset_results.append(_analyze_asset("沪深300期货", futures["沪深300期货"]["dates"], futures["沪深300期货"]["prices"], output_root))
    asset_results.append(_analyze_asset(call_item["name"], [dt for dt, _ in call_daily], [price for _, price in call_daily], output_root))
    asset_results.append(_analyze_asset(put_item["name"], [dt for dt, _ in put_daily], [price for _, price in put_daily], output_root))
    for stock in chosen_stocks:
        item = stocks[stock["code"]]
        asset_results.append(_analyze_asset(stock["name"], item["dates"], item["prices"], output_root))

    write_csv(
        output_root / "tables" / "基础统计汇总.csv",
        ["资产", "均值", "中位数", "标准差", "偏度", "峰度", "最小值", "最大值", "自相关", "最高点日期", "最低点日期"],
        _format_stats_rows(asset_results),
    )

    data_bundle = {
        "indices": indices,
        "etf_daily": {"dates": etf_dates, "prices": etf_prices},
        "etf_daily_rows": _daily_rows(etf_daily_rows),
        "etf_intraday_rows": etf_intraday_rows,
        "etf_from_intraday": etf_from_intraday,
        "stock_info": stock_info,
        "chosen_stocks": chosen_stocks,
        "stocks": stocks,
        "futures": futures,
        "call_table": call_table,
        "call_option": {"name": call_item["name"], "dates": [dt for dt, _ in call_daily], "prices": [price for _, price in call_daily]},
        "put_option": {"name": put_item["name"], "dates": [dt for dt, _ in put_daily], "prices": [price for _, price in put_daily]},
    }
    advanced_tables, table_headers = _advanced_tables(output_root, data_bundle)

    report_lines = [
        "# 第一次作业第二部分报告草稿",
        "",
        "## 一、样本选择",
        "",
        f"- 指数：{', '.join(indices)}",
        "- ETF：上证50ETF",
        "- 期货：沪深300期货",
        f"- 期权：{call_item['name']}、{put_item['name']}",
        "- 股票：" + "、".join(stock["name"] for stock in chosen_stocks),
        "",
        "## 二、基础统计结果",
        "",
        markdown_table(["资产", "均值", "中位数", "标准差", "偏度", "峰度", "最小值", "最大值", "自相关", "最高点日期", "最低点日期"], _format_stats_rows(asset_results)),
        "",
        "## 三、进阶分析结果",
        "",
    ]
    for name, rows in advanced_tables.items():
        report_lines.append(f"### {name}")
        report_lines.append("")
        report_lines.append(markdown_table(table_headers[name], rows[:12]) if rows else "无结果")
        report_lines.append("")
    report_lines.extend(
        [
            "## 四、结论摘要",
            "",
            "- 收益率分布普遍呈现尖峰厚尾，与正态分布存在差异。",
            "- 不同频率、不同时段和不同采样方式会改变收益率波动特征。",
            "- 指数、ETF 与股指期货在收益率层面相关性较强。",
            "- 股票之间与期货之间均存在明显的同期相关结构。",
            "- 期权与 ETF 的关系受方向、行权价与到期日共同影响。",
        ]
    )
    write_markdown(output_root, "\n".join(report_lines) + "\n")
