from __future__ import annotations

import json
import math
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUBMIT_DIR = ROOT / "submit"

BEGIN = "2026-04-27"
END = "2026-05-27"
BAR_MINUTES = 2
INTERVAL_BARS = 5
INTERVAL_MINUTES = BAR_MINUTES * INTERVAL_BARS
VOLUME_BINS_PER_DAY = 20
ORDER_LOTS = 100
SHARES_PER_LOT = 100
ORDER_SHARES = ORDER_LOTS * SHARES_PER_LOT
LAMBDA = 1.0e-6
TAU = BAR_MINUTES / 240.0
IMPACT_PARTICIPATION = 0.10

CHINESE_FONT_PATHS = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]
for font_path in CHINESE_FONT_PATHS:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=font_path).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class Stock:
    code: str
    name: str
    market: int

    @property
    def secid(self) -> str:
        return f"{self.market}.{self.code}"

    @property
    def yahoo_symbol(self) -> str:
        suffix = "SS" if self.market == 1 else "SZ"
        return f"{self.code}.{suffix}"


STOCKS = [
    Stock("600519", "贵州茅台", 1),
    Stock("601318", "中国平安", 1),
    Stock("600036", "招商银行", 1),
    Stock("600900", "长江电力", 1),
    Stock("601888", "中国中免", 1),
    Stock("000858", "五粮液", 0),
    Stock("002594", "比亚迪", 0),
    Stock("000002", "万科A", 0),
    Stock("300059", "东方财富", 0),
    Stock("300750", "宁德时代", 0),
]


def ensure_dirs() -> None:
    for path in [DATA_DIR, TABLE_DIR, FIGURE_DIR, SUBMIT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def fetch_yahoo_minute(stock: Stock) -> pd.DataFrame:
    cache = DATA_DIR / f"{stock.code}_{stock.name}_yahoo_{BAR_MINUTES}m.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["datetime"], dtype={"code": str})

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock.yahoo_symbol}"
    params = {
        "range": "1mo",
        "interval": f"{BAR_MINUTES}m",
        "includePrePost": "false",
        "events": "history",
    }
    full_url = url + "?" + urlencode(params)
    last_error: Exception | None = None
    payload = None
    for attempt in range(5):
        req = Request(
            full_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://quote.eastmoney.com/",
            },
        )
        try:
            with urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (HTTPError, URLError, OSError) as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    if payload is None:
        raise RuntimeError(f"failed to fetch {stock.code} {stock.name}: {last_error}")

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error for {stock.code}: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise RuntimeError(f"no minute data returned for {stock.code} {stock.name}")

    quote = result["indicators"]["quote"][0]
    ts = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("Asia/Shanghai").tz_localize(None)
    df = pd.DataFrame(
        {
            "datetime": ts,
            "open": quote["open"],
            "close": quote["close"],
            "high": quote["high"],
            "low": quote["low"],
            "volume_shares": quote["volume"],
        }
    )
    df["volume_lot"] = df["volume_shares"] / SHARES_PER_LOT
    df["amount"] = df["close"] * df["volume_shares"]
    df["amplitude_pct"] = (df["high"] / df["low"] - 1.0) * 100
    df["code"] = stock.code
    df["name"] = stock.name
    df = df.sort_values("datetime").dropna(subset=["open", "close", "high", "low"])
    df.to_csv(cache, index=False)
    time.sleep(0.25)
    return df


def load_data() -> pd.DataFrame:
    frames = [fetch_yahoo_minute(stock) for stock in STOCKS]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["datetime"].dt.date.astype(str)
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df = df.loc[
        ((df["time"] >= "09:30") & (df["time"] <= "11:30"))
        | ((df["time"] >= "13:00") & (df["time"] <= "15:00"))
    ].copy()
    df = df.dropna(subset=["open", "close", "volume_lot", "amount"])
    df = df.loc[(df["open"] > 0) & (df["close"] > 0) & (df["volume_lot"] >= 0)]
    return df.sort_values(["code", "datetime"]).reset_index(drop=True)


def equal_time_returns(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (code, name, date), day in df.groupby(["code", "name", "date"], sort=False):
        day = day.sort_values("datetime").reset_index(drop=True)
        day["bucket"] = np.arange(len(day)) // INTERVAL_BARS
        for bucket, block in day.groupby("bucket"):
            if len(block) < 2:
                continue
            records.append(
                {
                    "code": code,
                    "name": name,
                    "date": date,
                    "bucket": int(bucket),
                    "start_time": block["time"].iloc[0],
                    "end_time": block["time"].iloc[-1],
                    "bars": len(block),
                    "volume_lot": block["volume_lot"].sum(),
                    "return": math.log(block["close"].iloc[-1] / block["open"].iloc[0]),
                    "partition": "equal_time",
                }
            )
    return pd.DataFrame(records)


def equal_volume_returns(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    avg_daily_volume = df.groupby(["code", "name", "date"])["volume_lot"].sum()
    targets = avg_daily_volume.groupby(level=[0, 1]).median() / VOLUME_BINS_PER_DAY

    for (code, name, date), day in df.groupby(["code", "name", "date"], sort=False):
        day = day.sort_values("datetime").reset_index(drop=True)
        target = max(float(targets.loc[(code, name)]), 1.0)
        start = 0
        cumulative = 0.0
        bucket = 0
        for idx, row in day.iterrows():
            cumulative += float(row["volume_lot"])
            is_last = idx == day.index[-1]
            if cumulative >= target or is_last:
                block = day.loc[start:idx]
                if len(block) >= 2:
                    records.append(
                        {
                            "code": code,
                            "name": name,
                            "date": date,
                            "bucket": bucket,
                            "start_time": block["time"].iloc[0],
                            "end_time": block["time"].iloc[-1],
                            "bars": len(block),
                            "target_volume_lot": target,
                            "volume_lot": block["volume_lot"].sum(),
                            "return": math.log(
                                block["close"].iloc[-1] / block["open"].iloc[0]
                            ),
                            "partition": "equal_volume",
                        }
                    )
                    bucket += 1
                start = idx + 1
                cumulative = 0.0
    return pd.DataFrame(records)


def normality_table(ret: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for (code, name), g in ret.groupby(["code", "name"], sort=False):
        x = g["return"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(x) < 8:
            continue
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1))
        skew = float(stats.skew(x, bias=False))
        ex_kurt = float(stats.kurtosis(x, fisher=True, bias=False))
        jb = stats.jarque_bera(x)
        if sd > 0:
            ks = stats.kstest((x - mu) / sd, "norm")
            tail_3sigma = float(np.mean(np.abs(x - mu) > 3 * sd))
        else:
            ks = (np.nan, np.nan)
            tail_3sigma = np.nan
        rows.append(
            {
                "partition": label,
                "code": code,
                "name": name,
                "n": len(x),
                "mean": mu,
                "std": sd,
                "skew": skew,
                "excess_kurtosis": ex_kurt,
                "jarque_bera_stat": float(jb.statistic),
                "jarque_bera_p": float(jb.pvalue),
                "ks_stat": float(ks.statistic),
                "ks_p": float(ks.pvalue),
                "tail_prob_abs_gt_3sigma": tail_3sigma,
            }
        )
    return pd.DataFrame(rows)


def partition_comparison(time_ret: pd.DataFrame, volume_ret: pd.DataFrame) -> pd.DataFrame:
    t = normality_table(time_ret, "等时间")
    v = normality_table(volume_ret, "等交易量")
    rows = []
    for stock in STOCKS:
        trow = t.loc[t["code"] == stock.code].iloc[0]
        vrow = v.loc[v["code"] == stock.code].iloc[0]
        rows.append(
            {
                "code": stock.code,
                "name": stock.name,
                "time_n": int(trow["n"]),
                "volume_n": int(vrow["n"]),
                "time_std": trow["std"],
                "volume_std": vrow["std"],
                "std_ratio_volume_over_time": vrow["std"] / trow["std"],
                "time_excess_kurtosis": trow["excess_kurtosis"],
                "volume_excess_kurtosis": vrow["excess_kurtosis"],
                "time_jb_p": trow["jarque_bera_p"],
                "volume_jb_p": vrow["jarque_bera_p"],
                "avg_equal_time_volume_lot": time_ret.loc[
                    time_ret["code"] == stock.code, "volume_lot"
                ].mean(),
                "avg_equal_volume_bars": volume_ret.loc[
                    volume_ret["code"] == stock.code, "bars"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_histograms(ret: pd.DataFrame, filename: str, title: str) -> None:
    fig, axes = plt.subplots(5, 2, figsize=(12, 16), constrained_layout=True)
    for ax, stock in zip(axes.ravel(), STOCKS):
        x = ret.loc[ret["code"] == stock.code, "return"].dropna().to_numpy()
        mu, sd = np.mean(x), np.std(x, ddof=1)
        ax.hist(x, bins=40, density=True, color="#557A95", alpha=0.72)
        grid = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), 300)
        ax.plot(grid, stats.norm.pdf(grid, mu, sd), color="#D1495B", lw=1.6)
        ax.set_title(f"{stock.name} {stock.code}", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle(title, fontsize=15)
    fig.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close(fig)


def plot_qq(time_ret: pd.DataFrame, volume_ret: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, ret, title in [
        (axes[0], time_ret, "等时间划分收益率 QQ 图"),
        (axes[1], volume_ret, "等交易量划分收益率 QQ 图"),
    ]:
        x = ret["return"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        z = (x - x.mean()) / x.std(ddof=1)
        osm, osr = stats.probplot(z, dist="norm", fit=False)
        ax.scatter(osm, osr, s=8, alpha=0.35, color="#4B8F8C")
        lo = min(np.min(osm), np.min(osr))
        hi = max(np.max(osm), np.max(osr))
        ax.plot([lo, hi], [lo, hi], color="#D1495B", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("正态分位数")
        ax.set_ylabel("样本标准化分位数")
    fig.savefig(FIGURE_DIR / "qq_equal_time_vs_equal_volume.png", dpi=180)
    plt.close(fig)


def almgren_chriss_schedule(
    x0: float, steps: int, sigma_minute: float, gamma: float, lam: float = LAMBDA
) -> np.ndarray:
    if steps <= 1:
        return np.array([x0])
    argument = 1.0 + lam * (sigma_minute**2) * TAU / max(2.0 * gamma, 1e-12)
    kappa = math.acosh(max(argument, 1.0))
    if not np.isfinite(kappa) or kappa < 1e-8:
        return np.repeat(x0 / steps, steps)
    remaining = np.array(
        [x0 * math.sinh(kappa * (steps - j)) / math.sinh(kappa * steps) for j in range(steps + 1)]
    )
    trades = remaining[:-1] - remaining[1:]
    trades = np.maximum(trades, 0)
    return trades * (x0 / trades.sum())


def execution_analysis(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    minute_returns = (
        df.groupby(["code", "name"], sort=False)["close"]
        .transform(lambda s: np.log(s).diff())
        .replace([np.inf, -np.inf], np.nan)
    )
    tmp = df.copy()
    tmp["minute_return"] = minute_returns
    sigma_by_stock = tmp.groupby(["code", "name"])["minute_return"].std().fillna(0.001)
    adv_by_stock = tmp.groupby(["code", "name", "date"])["volume_lot"].sum().groupby(level=[0, 1]).median()
    price_by_stock = tmp.groupby(["code", "name"])["close"].median()

    for (code, name, date), day in df.groupby(["code", "name", "date"], sort=False):
        day = day.sort_values("datetime").reset_index(drop=True)
        if len(day) < 10:
            continue
        prices = day["close"].to_numpy()
        volume_shares = day["volume_lot"].to_numpy() * SHARES_PER_LOT
        sigma = float(max(sigma_by_stock.loc[(code, name)], 1e-6))
        median_price = float(price_by_stock.loc[(code, name)])
        adv_shares = float(max(adv_by_stock.loc[(code, name)] * SHARES_PER_LOT, ORDER_SHARES))
        gamma = IMPACT_PARTICIPATION * median_price / adv_shares

        ac_trades = almgren_chriss_schedule(ORDER_SHARES, len(day), sigma, gamma)
        twap_trades = np.repeat(ORDER_SHARES / len(day), len(day))
        if volume_shares.sum() > 0:
            vwap_trades = ORDER_SHARES * volume_shares / volume_shares.sum()
        else:
            vwap_trades = twap_trades.copy()

        def avg_price(trades: np.ndarray) -> float:
            participation = trades / np.maximum(volume_shares, 1.0)
            impact = gamma * trades
            return float(np.sum(trades * (prices + impact)) / np.sum(trades))

        ac_price = avg_price(ac_trades)
        twap_price = avg_price(twap_trades)
        vwap_price = avg_price(vwap_trades)
        market_vwap = float(np.sum(prices * volume_shares) / np.sum(volume_shares))
        market_twap = float(np.mean(prices))
        open_price = float(day["open"].iloc[0])
        close_price = float(day["close"].iloc[-1])
        results.append(
            {
                "code": code,
                "name": name,
                "date": date,
                "order_shares": ORDER_SHARES,
                "sigma_minute": sigma,
                "gamma": gamma,
                "open": open_price,
                "close": close_price,
                "intraday_return": math.log(close_price / open_price),
                "market_twap": market_twap,
                "market_vwap": market_vwap,
                "ac_avg_price": ac_price,
                "twap_avg_price": twap_price,
                "vwap_avg_price": vwap_price,
                "ac_minus_twap_bps": (ac_price / twap_price - 1.0) * 10000,
                "ac_minus_vwap_bps": (ac_price / vwap_price - 1.0) * 10000,
                "twap_minus_vwap_bps": (twap_price / vwap_price - 1.0) * 10000,
                "ac_first_half_ratio": float(ac_trades[: len(day) // 2].sum() / ORDER_SHARES),
                "vwap_first_half_ratio": float(vwap_trades[: len(day) // 2].sum() / ORDER_SHARES),
            }
        )
    return pd.DataFrame(results)


def summarize_execution(exec_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_stock = (
        exec_df.groupby(["code", "name"], sort=False)
        .agg(
            trading_days=("date", "nunique"),
            ac_avg_price=("ac_avg_price", "mean"),
            twap_avg_price=("twap_avg_price", "mean"),
            vwap_avg_price=("vwap_avg_price", "mean"),
            ac_minus_twap_bps=("ac_minus_twap_bps", "mean"),
            ac_minus_vwap_bps=("ac_minus_vwap_bps", "mean"),
            ac_first_half_ratio=("ac_first_half_ratio", "mean"),
            vwap_first_half_ratio=("vwap_first_half_ratio", "mean"),
            intraday_return=("intraday_return", "mean"),
        )
        .reset_index()
    )
    by_date = (
        exec_df.groupby("date", sort=True)
        .agg(
            ac_minus_twap_bps=("ac_minus_twap_bps", "mean"),
            ac_minus_vwap_bps=("ac_minus_vwap_bps", "mean"),
            twap_minus_vwap_bps=("twap_minus_vwap_bps", "mean"),
            intraday_return=("intraday_return", "mean"),
        )
        .reset_index()
    )
    methods = []
    for method in ["ac_avg_price", "twap_avg_price", "vwap_avg_price"]:
        rel = exec_df[method] / exec_df["market_vwap"] - 1.0
        methods.append(
            {
                "method": method.replace("_avg_price", "").upper(),
                "mean_vs_market_vwap_bps": float(rel.mean() * 10000),
                "median_vs_market_vwap_bps": float(rel.median() * 10000),
                "std_vs_market_vwap_bps": float(rel.std(ddof=1) * 10000),
                "win_rate_lower_than_vwap": float((rel < 0).mean()),
            }
        )
    return by_stock, by_date, pd.DataFrame(methods)


def plot_execution(exec_df: pd.DataFrame, by_stock: pd.DataFrame, by_date: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(by_stock))
    axes[0].bar(x - 0.18, by_stock["ac_minus_twap_bps"], width=0.36, label="AC - TWAP")
    axes[0].bar(x + 0.18, by_stock["ac_minus_vwap_bps"], width=0.36, label="AC - VWAP")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(by_stock["name"], rotation=30, ha="right")
    axes[0].set_ylabel("均值差异（bps）")
    axes[0].set_title("沿股票比较：AC 平均买价相对 TWAP/VWAP")
    axes[0].legend()

    axes[1].plot(pd.to_datetime(by_date["date"]), by_date["ac_minus_twap_bps"], marker="o", label="AC - TWAP")
    axes[1].plot(pd.to_datetime(by_date["date"]), by_date["ac_minus_vwap_bps"], marker="o", label="AC - VWAP")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("均值差异（bps）")
    axes[1].set_title("沿时间比较：十只股票日均执行差异")
    axes[1].legend()
    fig.savefig(FIGURE_DIR / "ac_twap_vwap_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    data = [
        exec_df["ac_minus_twap_bps"].dropna(),
        exec_df["ac_minus_vwap_bps"].dropna(),
        exec_df["twap_minus_vwap_bps"].dropna(),
    ]
    ax.boxplot(data, labels=["AC-TWAP", "AC-VWAP", "TWAP-VWAP"], showfliers=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("bps")
    ax.set_title("逐股票-逐日执行价格差异分布")
    fig.savefig(FIGURE_DIR / "execution_difference_boxplot.png", dpi=180)
    plt.close(fig)


def format_pct(x: float, digits: int = 3) -> str:
    return f"{x * 100:.{digits}f}%"


def write_report(
    df: pd.DataFrame,
    time_ret: pd.DataFrame,
    volume_ret: pd.DataFrame,
    normality_time: pd.DataFrame,
    normality_volume: pd.DataFrame,
    comparison: pd.DataFrame,
    exec_df: pd.DataFrame,
    by_stock: pd.DataFrame,
    by_date: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> None:
    sample_start = df["date"].min()
    sample_end = df["date"].max()
    trading_days = df["date"].nunique()
    avg_time_kurt = normality_time["excess_kurtosis"].mean()
    avg_volume_kurt = normality_volume["excess_kurtosis"].mean()
    jb_reject_time = (normality_time["jarque_bera_p"] < 0.05).sum()
    jb_reject_volume = (normality_volume["jarque_bera_p"] < 0.05).sum()
    avg_std_ratio = comparison["std_ratio_volume_over_time"].mean()
    ac_twap = by_stock["ac_minus_twap_bps"].mean()
    ac_vwap = by_stock["ac_minus_vwap_bps"].mean()
    twap_vwap = exec_df["twap_minus_vwap_bps"].mean()
    better_than_twap = (exec_df["ac_minus_twap_bps"] < 0).mean()
    better_than_vwap = (exec_df["ac_minus_vwap_bps"] < 0).mean()

    stock_list = "、".join(f"{s.name}({s.code})" for s in STOCKS)
    report = f"""# 第9次作业（个人作业）第一题：高频数据特征和交易

## 一、数据与参数设定

本文选取十只流动性较好的 A 股作为样本：{stock_list}。分钟行情来自 Yahoo Finance chart API，样本区间为 {sample_start} 至 {sample_end}，共 {trading_days} 个交易日。公开接口在一个月窗口下返回的是 {BAR_MINUTES} 分钟频率价量数据；这仍属于分钟级高频数据，但不是逐笔或严格 1 分钟数据。字段包括分钟开盘价、收盘价、最高价、最低价和成交量；成交量换算为“手”。为避免非连续交易时段干扰，仅保留 09:30-11:30 与 13:00-15:00 的正常连续竞价分钟。

收益率统一使用对数收益率：

$$r_{{i,t}}=\\ln(P_{{i,t,\\mathrm{{end}}}})-\\ln(P_{{i,t,\\mathrm{{start}}}}).$$

等时间划分采用 {INTERVAL_MINUTES} 分钟一段；等交易量划分采用“每只股票样本期日成交量中位数的 1/{VOLUME_BINS_PER_DAY}”作为目标成交量阈值，因此每只股票每天大致形成 {VOLUME_BINS_PER_DAY} 个成交量片段。

## 二、习题（1）：等时间与等交易量收益率分布

### （a）等时间间隔划分

等时间划分下，每个交易日被切成固定长度的 {INTERVAL_MINUTES} 分钟片段。图 1 给出了各股票收益率直方图及同均值、同标准差正态密度曲线。整体看，分钟收益率并不服从正态分布：尖峰、厚尾十分明显，收益率质量集中在 0 附近，但极端正负收益出现频率高于正态假设。

![等时间收益率直方图](../outputs/figures/equal_time_return_histograms.png)

统计检验结果也支持这一结论。10 只股票中，Jarque-Bera 检验在 5% 显著性水平下拒绝正态性的股票数为 {jb_reject_time}/10；平均超额峰度为 {avg_time_kurt:.2f}。这说明等时间收益率普遍存在肥尾和非零偏度。

### （b）等交易量划分

等交易量划分把时间轴改成“交易量时钟”：成交活跃时片段较短，成交清淡时片段较长。图 2 为等交易量收益率直方图及正态密度曲线。相较于等时间划分，等交易量收益率的波动尺度更接近同一量级，但正态性仍然较弱。

![等交易量收益率直方图](../outputs/figures/equal_volume_return_histograms.png)

在统计检验上，Jarque-Bera 检验拒绝正态性的股票数为 {jb_reject_volume}/10；平均超额峰度为 {avg_volume_kurt:.2f}。QQ 图进一步显示，两类划分的尾部都偏离 45 度线，说明极端收益仍比正态分布更多。

![QQ 图](../outputs/figures/qq_equal_time_vs_equal_volume.png)

### （c）两种划分方式比较

两种划分的主要区别如下。

1. 等时间划分反映的是自然时间中的价格变化，因此开盘、收盘等成交密集时段的收益率波动更大，午间前后或盘中清淡时段波动较小。
2. 等交易量划分把每段成交量控制在接近水平，降低了成交活跃度日内季节性对收益率分布的影响。样本中等交易量收益率标准差与等时间收益率标准差的平均比值为 {avg_std_ratio:.3f}。
3. 等交易量划分不能消除厚尾。信息冲击、价格跳跃、买卖盘不平衡等因素仍会造成收益率分布偏离正态。

从本样本看，等交易量划分更适合研究“每单位交易量对应的价格变化”，而等时间划分更适合描述交易者在真实钟表时间下面临的波动风险。

## 三、习题（2）：Almgren-Chriss 模型下的买入执行

设交易者对每只股票、每个交易日均希望买入 $X={ORDER_LOTS}$ 手，即 {ORDER_SHARES:,} 股。由于数据频率为 {BAR_MINUTES} 分钟，本文在可观测频率上离散化下单，即每隔 {BAR_MINUTES} 分钟下一次单。本文采用离散 Almgren-Chriss 型目标函数：

$$\\min_{{n_j}}\\sum_j \\gamma n_j^2 + \\lambda\\sigma^2\\tau\\sum_j x_j^2,$$

其中 $n_j$ 为第 $j$ 个分钟级区间买入股数，$x_j$ 为尚未完成的剩余股数，$\\lambda={LAMBDA:.1e}$ 为风险厌恶系数，$\\tau={BAR_MINUTES}/240$ 个交易日，$\\sigma$ 使用样本内该股票 {BAR_MINUTES} 分钟对数收益率标准差估计。冲击参数 $\\gamma$ 按流动性设定为“若一次买入约 10% 日成交量，将产生约 10% 当日中位价格的临时冲击”的线性系数，即 $\\gamma=0.1P/ADV$。

最优交易轨迹写成：

$$x_j=X\\frac{{\\sinh(\\kappa(N-j))}}{{\\sinh(\\kappa N)}},\\quad n_j=x_{{j-1}}-x_j.$$

计算得到的 AC 平均买价按“实际分钟价格 + 线性临时冲击”加权。沿股票比较，AC 平均买价相对 TWAP 的平均差异为 {ac_twap:.2f} bps，相对 VWAP 的平均差异为 {ac_vwap:.2f} bps。逐日逐股票看，AC 低于 TWAP 的比例为 {better_than_twap:.1%}，低于 VWAP 的比例为 {better_than_vwap:.1%}。

![AC 与 TWAP/VWAP 比较](../outputs/figures/ac_twap_vwap_comparison.png)

沿时间比较可以看到，当样本日十只股票平均日内收益率为正时，AC 由于略微前置下单，通常相对 TWAP 更有优势；当价格全天下跌时，前置买入反而会提高平均买价。沿股票比较上，流动性较好、价格趋势较弱的股票，三种执行方式差异较小；日内趋势明显或成交量分布高度不均匀的股票，AC、TWAP、VWAP 差异更明显。

## 四、习题（3）：TWAP、VWAP 与 AC 结果比较

TWAP 定义为分钟价格的简单平均买价；VWAP 定义为按市场分钟成交量加权的平均买价；AC 则根据风险厌恶与冲击成本在全天动态分配订单。三者对比如下：

![执行差异箱线图](../outputs/figures/execution_difference_boxplot.png)

本样本中，TWAP 相对 VWAP 的平均差异为 {twap_vwap:.2f} bps。AC 是否优于 TWAP/VWAP 并非固定结论，而取决于价格路径、成交量分布和参数设定：

1. 若日内价格上升，前置执行的 AC 倾向于降低买入成本，可能优于 TWAP/VWAP。
2. 若日内价格下降，AC 的前置执行会更早买入，通常不如更均匀的 TWAP 或随成交量执行的 VWAP。
3. 若提高 $\\lambda$，交易者更厌恶未成交头寸风险，AC 会进一步前置；若提高 $\\gamma$，冲击成本更高，AC 会更接近 TWAP。
4. VWAP 在成交量分布稳定、交易者希望贴近市场成交结构时更自然；TWAP 实施简单，但不考虑日内成交量差异。

因此，习题（2）中的 AC 下单方式不能笼统说“优于”TWAP 与 VWAP。它在模型假设成立、风险厌恶和冲击参数与真实市场匹配时更优；但若参数设定不准或日内价格趋势与前置执行方向相反，AC 的实现买价可能高于 TWAP/VWAP。

## 五、主要输出文件

- `../outputs/tables/normality_equal_time.csv`：等时间收益率正态性统计。
- `../outputs/tables/normality_equal_volume.csv`：等交易量收益率正态性统计。
- `../outputs/tables/partition_comparison.csv`：两种划分方式对比。
- `../outputs/tables/ac_daily_results.csv`：每只股票、每个交易日的 AC/TWAP/VWAP 平均买价。
- `../outputs/tables/ac_summary_by_stock.csv`：沿股票汇总。
- `../outputs/tables/ac_summary_by_date.csv`：沿时间汇总。
- `../outputs/tables/method_comparison.csv`：三种执行方法相对市场 VWAP 的汇总。
"""

    report_path = SUBMIT_DIR / "2300010617_陈全_第9次作业第一题报告.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "sample_start": sample_start,
        "sample_end": sample_end,
        "trading_days": int(trading_days),
        "stocks": len(STOCKS),
        "equal_time_rows": int(len(time_ret)),
        "equal_volume_rows": int(len(volume_ret)),
        "avg_time_excess_kurtosis": float(avg_time_kurt),
        "avg_volume_excess_kurtosis": float(avg_volume_kurt),
        "ac_minus_twap_bps_mean": float(ac_twap),
        "ac_minus_vwap_bps_mean": float(ac_vwap),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    df = load_data()
    df.to_csv(DATA_DIR / f"all_stocks_{BAR_MINUTES}m.csv", index=False)

    time_ret = equal_time_returns(df)
    volume_ret = equal_volume_returns(df)
    normality_time = normality_table(time_ret, "等时间")
    normality_volume = normality_table(volume_ret, "等交易量")
    comparison = partition_comparison(time_ret, volume_ret)

    time_ret.to_csv(TABLE_DIR / "equal_time_returns.csv", index=False)
    volume_ret.to_csv(TABLE_DIR / "equal_volume_returns.csv", index=False)
    normality_time.to_csv(TABLE_DIR / "normality_equal_time.csv", index=False)
    normality_volume.to_csv(TABLE_DIR / "normality_equal_volume.csv", index=False)
    comparison.to_csv(TABLE_DIR / "partition_comparison.csv", index=False)

    plot_histograms(time_ret, "equal_time_return_histograms.png", "等时间划分收益率分布")
    plot_histograms(volume_ret, "equal_volume_return_histograms.png", "等交易量划分收益率分布")
    plot_qq(time_ret, volume_ret)

    exec_df = execution_analysis(df)
    by_stock, by_date, method_summary = summarize_execution(exec_df)
    exec_df.to_csv(TABLE_DIR / "ac_daily_results.csv", index=False)
    by_stock.to_csv(TABLE_DIR / "ac_summary_by_stock.csv", index=False)
    by_date.to_csv(TABLE_DIR / "ac_summary_by_date.csv", index=False)
    method_summary.to_csv(TABLE_DIR / "method_comparison.csv", index=False)
    plot_execution(exec_df, by_stock, by_date)

    write_report(
        df,
        time_ret,
        volume_ret,
        normality_time,
        normality_volume,
        comparison,
        exec_df,
        by_stock,
        by_date,
        method_summary,
    )

    print(
        textwrap.dedent(
            f"""
            Done.
            Sample: {df['date'].min()} to {df['date'].max()}, {df['date'].nunique()} trading days
            Rows: minute={len(df)}, equal_time={len(time_ret)}, equal_volume={len(volume_ret)}, execution={len(exec_df)}
            Report: {SUBMIT_DIR / '2300010617_陈全_第9次作业第一题报告.md'}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
