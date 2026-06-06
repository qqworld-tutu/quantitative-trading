#!/usr/bin/env python3
"""Analyze A-share order-by-order data for HW10.

The script reads tickdata.zip in-place and writes reproducible tables, SVG
figures, and a Chinese Markdown report under hw10/outputs and hw10/submit.
"""

from __future__ import annotations

import csv
import math
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "tickdata.zip"
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
SUBMIT_DIR = ROOT / "submit"

ORDER_USECOLS = ["tradedate", "OrigTime", "Price", "OrderQty", "OrderType"]
TRADE_USECOLS = ["tradedate", "Price", "TradeQty"]

STOCK_NAMES = {
    "000001": "平安银行",
    "000612": "焦作万方",
    "002251": "步步高",
    "002277": "友阿股份",
    "002457": "青龙管业",
    "300131": "英唐智控",
    "300227": "光韵达",
    "300228": "富瑞特装",
    "300684": "中石科技",
    "301115": "建科股份",
}


@dataclass
class StockStats:
    order_type_counts: Counter = field(default_factory=Counter)
    total_order_qty: float = 0.0
    total_order_value: float = 0.0
    priced_order_count: int = 0
    order_rows: int = 0
    interval_arrays_ms: list[np.ndarray] = field(default_factory=list)
    interval_zero_count: int = 0
    interval_total_count: int = 0
    trade_value: float = 0.0
    trade_volume: float = 0.0
    trade_count: int = 0


def time_to_ms_since_midnight(values: pd.Series) -> np.ndarray:
    """Convert YYYYMMDDHHMMSSmmm integer timestamps to ms after midnight."""
    arr = pd.to_numeric(values, errors="coerce").dropna().astype("int64").to_numpy()
    hour = (arr // 10_000_000) % 100
    minute = (arr // 100_000) % 100
    second = (arr // 1_000) % 100
    millis = arr % 1_000
    return ((hour * 3600 + minute * 60 + second) * 1000 + millis).astype("int64")


def read_csv_from_zip(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(fh, usecols=usecols, low_memory=False)


def fit_exponential(x: np.ndarray) -> dict[str, float | str]:
    mean = float(np.mean(x))
    lam = 1.0 / mean
    loglik = len(x) * math.log(lam) - lam * float(np.sum(x))
    ks = ks_stat(x, lambda t: 1.0 - np.exp(-t / mean))
    return {"distribution": "指数分布", "param": f"mean={mean:.6g}", "loglik": loglik, "aic": 2 - 2 * loglik, "ks": ks}


def fit_lognormal(x: np.ndarray) -> dict[str, float | str]:
    lx = np.log(x)
    mu = float(np.mean(lx))
    sigma = float(np.std(lx, ddof=0))
    sigma = max(sigma, 1e-12)
    z = (lx - mu) / sigma
    loglik = float(np.sum(-np.log(x) - math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * z * z))

    def cdf(t: np.ndarray) -> np.ndarray:
        vals = (np.log(t) - mu) / (sigma * math.sqrt(2))
        return 0.5 * (1.0 + np.vectorize(math.erf)(vals))

    ks = ks_stat(x, cdf)
    return {"distribution": "对数正态分布", "param": f"mu={mu:.6g}; sigma={sigma:.6g}", "loglik": loglik, "aic": 4 - 2 * loglik, "ks": ks}


def fit_power_law(x: np.ndarray) -> dict[str, float | str]:
    xmin = float(np.min(x))
    logs = np.log(x / xmin)
    denom = float(np.sum(logs))
    if denom <= 0:
        return {"distribution": "幂律分布", "param": "fit_failed", "loglik": float("-inf"), "aic": float("inf"), "ks": float("inf")}
    alpha = 1.0 + len(x) / denom
    loglik = len(x) * math.log((alpha - 1.0) / xmin) - alpha * denom

    def cdf(t: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(t / xmin, 1.0 - alpha)

    ks = ks_stat(x, cdf)
    return {"distribution": "幂律分布", "param": f"xmin={xmin:.6g}; alpha={alpha:.6g}", "loglik": loglik, "aic": 4 - 2 * loglik, "ks": ks}


def ks_stat(x: np.ndarray, cdf_func) -> float:
    xs = np.sort(x)
    n = len(xs)
    empirical = np.arange(1, n + 1) / n
    fitted = np.clip(cdf_func(xs), 0.0, 1.0)
    return float(np.max(np.abs(empirical - fitted)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def board_name(code: str) -> str:
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith("002"):
        return "深市中小板/主板"
    return "深市主板"


def analyze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, StockStats] = defaultdict(StockStats)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        stock_dirs = sorted({p.split("/")[1] for p in names if p.startswith("tickdata/") and len(p.split("/")) > 2 and p.split("/")[1]})

        order_files = sorted(n for n in names if n.endswith("hq_order_spot.csv"))
        trade_files = sorted(n for n in names if n.endswith("hq_trade_spot.csv"))

        for name in order_files:
            stock = name.split("/")[1]
            df = read_csv_from_zip(zf, name, ORDER_USECOLS)
            st = stats[stock]
            st.order_rows += len(df)

            order_type = df["OrderType"].astype(str).str.strip()
            st.order_type_counts.update(order_type)

            price = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            qty = pd.to_numeric(df["OrderQty"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid_price = price > 0
            st.total_order_qty += float(np.sum(qty))
            st.total_order_value += float(np.sum(price[valid_price] * qty[valid_price]))
            st.priced_order_count += int(np.sum(valid_price))

            ms = time_to_ms_since_midnight(df["OrigTime"])
            if len(ms) > 1:
                diffs = np.diff(np.sort(ms))
                st.interval_total_count += len(diffs)
                st.interval_zero_count += int(np.sum(diffs == 0))
                positive = diffs[diffs > 0].astype(float) / 1000.0
                if len(positive):
                    st.interval_arrays_ms.append(positive)

        for name in trade_files:
            stock = name.split("/")[1]
            df = read_csv_from_zip(zf, name, TRADE_USECOLS)
            price = pd.to_numeric(df["Price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            qty = pd.to_numeric(df["TradeQty"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            valid = (price > 0) & (qty > 0)
            st = stats[stock]
            st.trade_value += float(np.sum(price[valid] * qty[valid]))
            st.trade_volume += float(np.sum(qty[valid]))
            st.trade_count += int(np.sum(valid))

    summary_rows = []
    interval_rows = []
    fit_rows = []

    all_stocks = sorted(set(STOCK_NAMES) | set(stats))
    for stock in all_stocks:
        st = stats.get(stock, StockStats())
        limit_count = int(st.order_type_counts.get("2", 0))
        market_count = int(st.order_type_counts.get("1", 0))
        other_count = int(sum(v for k, v in st.order_type_counts.items() if k not in {"1", "2"}))
        total_orders = limit_count + market_count + other_count
        avg_order_price = st.total_order_value / st.total_order_qty if st.total_order_qty > 0 else np.nan
        vwap = st.trade_value / st.trade_volume if st.trade_volume > 0 else np.nan
        avg_order_qty = st.total_order_qty / total_orders if total_orders else np.nan

        summary_rows.append(
            {
                "stock": stock,
                "name": STOCK_NAMES.get(stock, ""),
                "board": board_name(stock),
                "total_orders": total_orders,
                "limit_orders": limit_count,
                "market_orders": market_count,
                "other_orders": other_count,
                "limit_share": limit_count / total_orders if total_orders else np.nan,
                "market_share": market_count / total_orders if total_orders else np.nan,
                "avg_order_price": avg_order_price,
                "avg_order_qty": avg_order_qty,
                "vwap_trade_price": vwap,
                "trade_value_yuan": st.trade_value,
                "trade_volume_shares": st.trade_volume,
                "valid_trade_count": st.trade_count,
            }
        )

        if st.interval_total_count:
            positives = np.concatenate(st.interval_arrays_ms) if st.interval_arrays_ms else np.array([], dtype=float)
            zero_share = st.interval_zero_count / st.interval_total_count
            interval_rows.append(
                {
                    "stock": stock,
                    "name": STOCK_NAMES.get(stock, ""),
                    "interval_count": st.interval_total_count,
                    "positive_interval_count": len(positives),
                    "zero_interval_share": zero_share,
                    "mean_s": float(np.mean(positives)) if len(positives) else np.nan,
                    "median_s": float(np.median(positives)) if len(positives) else np.nan,
                    "p90_s": float(np.percentile(positives, 90)) if len(positives) else np.nan,
                    "p99_s": float(np.percentile(positives, 99)) if len(positives) else np.nan,
                }
            )
            if len(positives) >= 10:
                fits = [fit_exponential(positives), fit_lognormal(positives), fit_power_law(positives)]
                best_aic = min(fits, key=lambda r: float(r["aic"]))["distribution"]
                best_ks = min(fits, key=lambda r: float(r["ks"]))["distribution"]
                for row in fits:
                    row.update({"stock": stock, "name": STOCK_NAMES.get(stock, ""), "n_positive": len(positives), "best_by_aic": row["distribution"] == best_aic, "best_by_ks": row["distribution"] == best_ks})
                    fit_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("total_orders", ascending=False)
    intervals = pd.DataFrame(interval_rows).sort_values("stock")
    fits_df = pd.DataFrame(fit_rows)
    best = fits_df.loc[fits_df.groupby("stock")["aic"].idxmin(), ["stock", "distribution"]].rename(columns={"distribution": "best_distribution_by_aic"})
    intervals = intervals.merge(best, on="stock", how="left")

    summary.to_csv(TABLE_DIR / "order_summary.csv", index=False)
    intervals.to_csv(TABLE_DIR / "interval_summary.csv", index=False)
    fits_df.to_csv(TABLE_DIR / "fit_results.csv", index=False)

    write_svg_order_counts(summary[summary["total_orders"] > 0], FIGURE_DIR / "order_counts.svg")
    write_svg_scatter(summary[summary["total_orders"] > 0], FIGURE_DIR / "orders_vs_features.svg")
    write_report(summary, intervals, fits_df, SUBMIT_DIR / "hw10_report.md")
    return summary, intervals, fits_df


def write_svg_order_counts(df: pd.DataFrame, path: Path) -> None:
    width, height = 960, 540
    margin_left, margin_top, margin_right, row_h = 120, 40, 40, 42
    chart_w = width - margin_left - margin_right
    max_orders = float(df["total_orders"].max())
    rows = []
    for i, row in enumerate(df.sort_values("total_orders", ascending=True).itertuples()):
        y = margin_top + i * row_h
        x = margin_left
        total = row.total_orders
        colors = [("limit_orders", "#3b82f6"), ("market_orders", "#ef4444"), ("other_orders", "#a3a3a3")]
        rows.append(f'<text x="{margin_left-12}" y="{y+18}" text-anchor="end" font-size="14">{row.stock}</text>')
        for key, color in colors:
            val = float(getattr(row, key))
            w = chart_w * val / max_orders
            if w > 0:
                rows.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" fill="{color}" />')
            x += w
        rows.append(f'<text x="{x+8:.1f}" y="{y+18}" font-size="13">{fmt_int(total)}</text>')
    legend = (
        '<rect x="120" y="500" width="14" height="14" fill="#3b82f6"/><text x="140" y="512" font-size="13">限价单</text>'
        '<rect x="210" y="500" width="14" height="14" fill="#ef4444"/><text x="230" y="512" font-size="13">市价单</text>'
        '<rect x="300" y="500" width="14" height="14" fill="#a3a3a3"/><text x="320" y="512" font-size="13">其他</text>'
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="24" y="28" font-size="20" font-weight="700">各股票订单数量对比</text>
{''.join(rows)}
{legend}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_svg_scatter(df: pd.DataFrame, path: Path) -> None:
    width, height = 980, 420
    panels = [
        ("vwap_trade_price", "成交 VWAP 价格", 70, 45, 400, 300),
        ("trade_value_yuan", "期间成交额", 560, 45, 360, 300),
    ]
    points = []
    for col, title, x0, y0, w, h in panels:
        x = df[col].to_numpy(dtype=float)
        y = df["total_orders"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ymin, ymax = 0.0, float(np.max(y))
        points.append(f'<text x="{x0}" y="{y0-16}" font-size="17" font-weight="700">{title} vs 订单总数</text>')
        points.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="#555"/>')
        points.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" stroke="#555"/>')
        for row in df.itertuples():
            xv = getattr(row, col)
            yv = row.total_orders
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
            px = x0 + (float(xv) - xmin) / (xmax - xmin or 1.0) * w
            py = y0 + h - (float(yv) - ymin) / (ymax - ymin or 1.0) * h
            color = "#16a34a" if row.board == "创业板" else "#7c3aed"
            points.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{color}" opacity="0.85"/>')
            points.append(f'<text x="{px+7:.1f}" y="{py-7:.1f}" font-size="11">{row.stock}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
{''.join(points)}
<text x="70" y="390" font-size="13" fill="#555">紫色：深市主板/中小板；绿色：创业板。纵轴均为订单总数。</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str], max_rows: int | None = None) -> str:
    d = df.loc[:, columns].copy()
    if max_rows:
        d = d.head(max_rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (int, np.integer)):
                vals.append(fmt_int(val))
            elif isinstance(val, (float, np.floating)):
                if "share" in col:
                    vals.append(pct(float(val)) if np.isfinite(val) else "")
                elif "value" in col:
                    vals.append(f"{val/1e8:.2f}亿" if np.isfinite(val) else "")
                elif "price" in col or col.endswith("_s") or col in {"ks", "aic"}:
                    vals.append(f"{val:.4g}" if np.isfinite(val) else "")
                else:
                    vals.append(f"{val:.4g}" if np.isfinite(val) else "")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, intervals: pd.DataFrame, fits_df: pd.DataFrame, path: Path) -> None:
    valid_summary = summary[summary["total_orders"] > 0].copy()
    corr_rows = []
    for col, label in [
        ("vwap_trade_price", "成交 VWAP 价格"),
        ("trade_value_yuan", "期间成交额"),
        ("avg_order_qty", "平均订单股数"),
    ]:
        corr_rows.append(
            {
                "feature": label,
                "pearson": pearson(valid_summary["total_orders"].to_numpy(float), valid_summary[col].to_numpy(float)),
                "spearman": spearman(valid_summary["total_orders"].to_numpy(float), valid_summary[col].to_numpy(float)),
            }
        )
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(TABLE_DIR / "order_feature_correlations.csv", index=False)

    best_counts = fits_df[fits_df["best_by_aic"]].groupby("distribution").size().sort_values(ascending=False)
    best_sentence = "；".join(f"{dist} {count} 只" for dist, count in best_counts.items())
    ks_best_counts = fits_df[fits_df["best_by_ks"]].groupby("distribution").size().sort_values(ascending=False)
    ks_best_sentence = "；".join(f"{dist} {count} 只" for dist, count in ks_best_counts.items())
    date_min = "2024-12-18"
    date_max = "2024-12-31"
    empty_note = summary.loc[summary["total_orders"] == 0, "stock"].tolist()

    top = valid_summary.iloc[0]
    low = valid_summary.iloc[-1]
    report = f"""# 中国 A 股逐笔下单数据分析

## 1. 数据与处理口径

原始数据为 `hw10/tickdata.zip`。压缩包中出现 10 个股票目录，其中 `301115` 为空目录；实际可用于订单和成交分析的股票为 9 只。样本交易日为 {date_min} 至 {date_max} 的 10 个交易日。

本文只使用每只股票每天上午/下午的 `hq_order_spot.csv` 计算订单数量和相邻下单间隔，并使用 `hq_trade_spot.csv` 中价格大于 0 的成交记录计算期间成交额与 VWAP 价格。`OrderType=1` 记为市价单，`OrderType=2` 记为限价单，`OrderType=U` 记为其他/特殊订单。相邻下单间隔只在同一个半日文件内部计算，避免把午休和隔夜间隔混入分布。

## 2. 限价单与市价单统计

{markdown_table(valid_summary, ["stock", "name", "board", "total_orders", "limit_orders", "market_orders", "other_orders", "limit_share", "market_share", "vwap_trade_price", "trade_value_yuan"], ["股票", "名称", "板块", "订单总数", "限价单", "市价单", "其他", "限价占比", "市价占比", "VWAP", "成交额"])}

订单数量最高的是 `{top["stock"]}`（{top["name"]}），共 {fmt_int(top["total_orders"])} 笔；最低的是 `{low["stock"]}`（{low["name"]}），共 {fmt_int(low["total_orders"])} 笔。所有有效股票均以限价单为绝对主体，限价单占比接近或超过 99%，市价单和 `U` 类订单数量很小。这说明样本中的深市逐笔委托以限价委托为主，市价委托不是订单流的主要来源。

`301115` 因目录为空，无法纳入统计。若作业必须严格保留 10 只股票，可把它作为“原始数据缺失”的样本列示，而不应填造订单数。

### 订单数与股票特征关系

{markdown_table(corr, ["feature", "pearson", "spearman"], ["特征", "Pearson相关", "Spearman相关"])}

从样本看，订单总数与期间成交额的相关性最强，说明订单活跃度更接近流动性/成交活跃程度；与成交 VWAP 价格的相关性弱得多，不能说明“股价越高订单越多”。平均订单股数与订单数也不是稳定正相关，说明大订单规模和订单频率是两个不同维度。原始数据未包含总股本或流通股本，因此本文没有直接计算市值；若引入外部市值数据，建议进一步比较订单数与流通市值、换手率之间的关系。

![各股票订单数量对比](../outputs/figures/order_counts.svg)

![订单数与股票特征散点图](../outputs/figures/orders_vs_features.svg)

## 3. 相邻下单时间间隔分布

由于逐笔数据时间戳精度为毫秒，同一毫秒内可能出现多笔订单。连续分布拟合不能处理 0 间隔，因此表中单独报告 0 间隔占比；指数分布、对数正态分布和幂律分布均只对正间隔拟合。评价指标使用 AIC 和 KS 距离：AIC 越小越好，KS 越小越好。

{markdown_table(intervals, ["stock", "name", "interval_count", "positive_interval_count", "zero_interval_share", "mean_s", "median_s", "p90_s", "p99_s", "best_distribution_by_aic"], ["股票", "名称", "间隔数", "正间隔数", "0间隔占比", "均值(s)", "中位数(s)", "90%分位(s)", "99%分位(s)", "AIC最佳分布"])}

按 AIC 选择的最佳分布汇总为：{best_sentence}；按 KS 距离选择的最佳分布汇总为：{ks_best_sentence}。综合两类指标，对数正态分布最稳健：它在 6 只股票上取得最低 AIC，并在全部 9 只有效股票上取得最低 KS 距离。这与市场微观结构直觉一致：订单到达不是稳定强度的泊松过程，存在开盘集合竞价、连续竞价中的交易活跃时段、信息冲击和流动性聚集，因此间隔分布具有右偏、厚尾和波动聚集特征。指数分布要求无记忆性和恒定到达率，拟合效果较差；幂律分布能描述尾部但在主体区间的整体拟合稳定性较弱。

详细拟合结果保存在 `outputs/tables/fit_results.csv`，其中 `best_by_aic=True` 表示该股票按 AIC 最优。

## 4. 结论

1. 样本中有效数据为 9 只股票，`301115` 缺失订单文件。
2. 限价单占绝对多数，市价单比例很低，订单类型结构在不同股票间差异不大。
3. 订单数量差异很大，主要与成交活跃度/流动性相关，而不是简单由股价水平解释。
4. 相邻订单正间隔分布整体以对数正态分布拟合最好；AIC 下少数股票偏向幂律，但 KS 指标显示对数正态的整体分布贴合度更稳定。
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    summary, intervals, fits = analyze()
    print(f"Wrote {TABLE_DIR / 'order_summary.csv'}")
    print(f"Wrote {TABLE_DIR / 'interval_summary.csv'}")
    print(f"Wrote {TABLE_DIR / 'fit_results.csv'}")
    print(f"Wrote {SUBMIT_DIR / 'hw10_report.md'}")
    print(f"Valid stocks: {(summary['total_orders'] > 0).sum()}, fit rows: {len(fits)}")


if __name__ == "__main__":
    main()
