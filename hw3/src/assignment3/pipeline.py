from pathlib import Path

import numpy as np

from assignment2.reporting import multi_line_svg
from assignment3.backtest import annualized_return, annualized_vol, max_drawdown, monthly_stats, portfolio_curve, sharpe_ratio
from assignment3.config import FIGURES_DIR, INITIAL_CAPITAL, MV_TARGET_METHOD, OUTPUT_DIR, TABLES_DIR, UTILITY_RISK_AVERSION, WINDOW
from assignment3.data import find_month_ends, load_prices_and_returns
from assignment3.factor_analysis import beta_rank_rows, rolling_beta, rolling_r2_current_beta, rolling_r2_previous_beta
from assignment3.optimizers import covariance_matrix, global_min_variance_weights, mean_variance_target_weights, mean_vector, min_eigenvalue, quadratic_utility_weights
from assignment3.reporting import ensure_dirs, markdown_table, write_csv, write_markdown, write_markdown_named


def _monthly_periods(data):
    dates = data["dates"]
    asset_returns = data["asset_returns"]
    month_ends = [idx for idx in find_month_ends(dates) if idx >= WINDOW]
    periods = []
    for pos, end_idx in enumerate(month_ends[:-1]):
        next_end = month_ends[pos + 1]
        window_returns = asset_returns[end_idx - WINDOW:end_idx]
        hold_returns = asset_returns[end_idx:next_end]
        periods.append({"end_idx": end_idx, "returns_window": window_returns, "hold_returns": hold_returns})
    return periods, month_ends


def _strategy_periods(periods):
    gmv_periods, mv_periods, util_periods = [], [], []
    weight_rows = []
    eig_rows = []
    for period in periods:
        mu = mean_vector(period["returns_window"])
        cov = covariance_matrix(period["returns_window"])
        eig = min_eigenvalue(cov)
        target_return = float(np.mean(mu))
        w_gmv = global_min_variance_weights(cov)
        w_mv = mean_variance_target_weights(mu, cov, target_return)
        w_util = quadratic_utility_weights(mu, cov, UTILITY_RISK_AVERSION)
        gmv_periods.append({"returns": period["hold_returns"], "weights": w_gmv})
        mv_periods.append({"returns": period["hold_returns"], "weights": w_mv})
        util_periods.append({"returns": period["hold_returns"], "weights": w_util})
        label = period.get("label", str(period["end_idx"]))
        weight_rows.append([label, "GMV", *[round(x, 6) for x in w_gmv]])
        weight_rows.append([label, "MV", *[round(x, 6) for x in w_mv]])
        weight_rows.append([label, "Utility", *[round(x, 6) for x in w_util]])
        eig_rows.append([label, round(eig, 8)])
    return gmv_periods, mv_periods, util_periods, weight_rows, eig_rows


def _sparse_ticks(labels, count=6):
    if not labels:
        return []
    step = max(1, len(labels) // count)
    ticks = [(idx, label) for idx, label in enumerate(labels) if idx % step == 0]
    if ticks[-1][0] != len(labels) - 1:
        ticks.append((len(labels) - 1, labels[-1]))
    return ticks


def run(output_root=OUTPUT_DIR):
    output_root = Path(output_root)
    ensure_dirs(output_root)
    data = load_prices_and_returns()
    periods, month_ends = _monthly_periods(data)
    for period in periods:
        period["label"] = data["dates"][period["end_idx"]].strftime("%Y-%m")
    gmv_periods, mv_periods, util_periods, weight_rows, eig_rows = _strategy_periods(periods)

    gmv_curve = portfolio_curve(gmv_periods, INITIAL_CAPITAL)
    mv_curve = portfolio_curve(mv_periods, INITIAL_CAPITAL)
    util_curve = portfolio_curve(util_periods, INITIAL_CAPITAL)

    full_dates = data["dates"][WINDOW:]
    x_ticks = []
    seen = {}
    for idx, dt in enumerate(full_dates):
        seen.setdefault(dt.year, idx)
    for year, idx in sorted(seen.items()):
        if year % 2 == 0:
            x_ticks.append((idx, str(year)))

    multi_line_svg(
        output_root / "figures" / "portfolio_curves.svg",
        "三种滚动组合净值曲线",
        {"GMV": gmv_curve, "MV": mv_curve, "Utility": util_curve},
        "净值",
        x_ticks=x_ticks,
    )
    multi_line_svg(
        output_root / "figures" / "min_eigenvalue.svg",
        "协方差矩阵最小特征值序列",
        {"min_eigenvalue": [row[1] for row in eig_rows]},
        "特征值",
        y_tick_mode="sci",
    )

    rebalance_dates = [row[0] for row in eig_rows]
    weight_maps = {"GMV": {}, "MV": {}, "Utility": {}}
    for strategy in weight_maps:
        subset = [row for row in weight_rows if row[1] == strategy]
        for asset_idx, asset_name in enumerate(data["asset_names"]):
            weight_maps[strategy][asset_name] = [row[2 + asset_idx] for row in subset]
        multi_line_svg(
            output_root / "figures" / f"{strategy.lower()}_weights.svg",
            f"{strategy} 组合月度权重变化",
            weight_maps[strategy],
            "权重",
            x_ticks=_sparse_ticks(rebalance_dates),
        )

    perf_rows = [
        ["GMV", round(annualized_return(gmv_curve), 6), round(annualized_vol(gmv_curve), 6), round(sharpe_ratio(gmv_curve), 6), round(max_drawdown(gmv_curve), 6)],
        ["MV", round(annualized_return(mv_curve), 6), round(annualized_vol(mv_curve), 6), round(sharpe_ratio(mv_curve), 6), round(max_drawdown(mv_curve), 6)],
        ["Utility", round(annualized_return(util_curve), 6), round(annualized_vol(util_curve), 6), round(sharpe_ratio(util_curve), 6), round(max_drawdown(util_curve), 6)],
    ]
    write_csv(output_root / "tables" / "组合绩效比较.csv", ["策略", "年化收益", "年化波动", "夏普", "最大回撤"], perf_rows)
    write_csv(output_root / "tables" / "最小特征值序列.csv", ["月份", "最小特征值"], eig_rows)
    write_csv(output_root / "tables" / "月度权重.csv", ["月份", "策略", *data["asset_names"]], weight_rows)
    monthly_rows = []
    for name, curve in [("GMV", gmv_curve), ("MV", mv_curve), ("Utility", util_curve)]:
        for month, mean_ret, std_ret in monthly_stats(full_dates, curve):
            monthly_rows.append([name, month, round(mean_ret, 6), round(std_ret, 6)])
    write_csv(output_root / "tables" / "月度收益与波动.csv", ["策略", "月份", "平均日收益", "收益率标准差"], monthly_rows)

    beta_rows = []
    r2_rows = []
    beta_map = {}
    r2_current_map = {}
    r2_previous_map = {}
    for i, name in enumerate(data["asset_names"]):
        beta = rolling_beta(data["asset_returns"][:, i], data["benchmark_returns"], WINDOW)
        r2_now = rolling_r2_current_beta(data["asset_returns"][:, i], data["benchmark_returns"], WINDOW)
        r2_prev = rolling_r2_previous_beta(data["asset_returns"][:, i], data["benchmark_returns"], WINDOW)
        beta_map[name] = beta
        r2_current_map[name] = r2_now
        r2_previous_map[name] = r2_prev
        for idx, value in enumerate(beta):
            beta_rows.append([name, idx, round(value, 6)])
        for idx, (a, b) in enumerate(zip(r2_now, r2_prev)):
            r2_rows.append([name, idx, round(a, 6), round(b, 6)])
    write_csv(output_root / "tables" / "滚动Beta.csv", ["股票", "索引", "Beta"], beta_rows)
    write_csv(output_root / "tables" / "滚动R2.csv", ["股票", "索引", "当前Beta_R2", "前一期Beta_R2"], r2_rows)

    factor_full_dates = data["dates"][WINDOW - 1:]
    factor_labels = [dt.strftime("%Y-%m-%d") for dt in factor_full_dates]
    factor_ticks = []
    seen_year = {}
    for idx, dt in enumerate(factor_full_dates):
        seen_year.setdefault(dt.year, idx)
    for year, idx in sorted(seen_year.items()):
        if year % 2 == 0:
            factor_ticks.append((idx, str(year)))
    multi_line_svg(
        output_root / "figures" / "rolling_beta.svg",
        "5只股票滚动Beta时间序列",
        beta_map,
        "Beta",
        x_ticks=factor_ticks,
    )
    multi_line_svg(
        output_root / "figures" / "rolling_r2_current.svg",
        "当前Beta的滚动R²",
        r2_current_map,
        "R²",
        x_ticks=factor_ticks,
    )
    multi_line_svg(
        output_root / "figures" / "rolling_r2_previous.svg",
        "前一期Beta的滚动R²",
        r2_previous_map,
        "R²",
        x_ticks=factor_ticks,
    )
    rank_rows = beta_rank_rows(factor_labels, beta_map)
    write_csv(output_root / "tables" / "Beta排序变化.csv", ["日期", "第1", "第2", "第3", "第4", "第5"], rank_rows)

    report = [
        "# 第三次作业报告",
        "",
        "## 一、作业目标",
        "",
        "本部分使用第一次作业中的 5 只股票日收益率，在 50 天滚动建模、月频调仓框架下构建三类组合：全局最小方差组合、均值-方差最优组合与二次效用最优组合。",
        "",
        "## 二、模型设定",
        "",
        "- 滚动窗口：50 个交易日",
        "- 调仓频率：月频",
        "- 按课件闭式解直接计算权重，不额外加入非负权重约束，因此允许出现空头和杠杆权重",
        "- 无风险利率：年化 2%，折算为日频近似（用于 Sharpe 指标和第二部分回归）",
        f"- 均值-方差组合目标收益 b：每月取过去 50 日平均收益率向量 `μ` 的分量均值",
        f"- 二次效用组合风险厌恶系数 α：{UTILITY_RISK_AVERSION}",
        "",
        "## 三、组合绩效比较",
        "",
        markdown_table(["策略", "年化收益", "年化波动", "夏普", "最大回撤"], perf_rows),
        "",
        "![组合净值](../figures/portfolio_curves.svg)",
        "",
        "从结果看，GMV 组合在这一版设定下表现最好，年化收益率和夏普比率都高于另外两种组合。这说明在 5 只股票的小样本下，协方差结构往往比均值估计更稳健。",
        "",
        "## 四、协方差矩阵最小特征值",
        "",
        "![最小特征值](../figures/min_eigenvalue.svg)",
        "",
        "最小特征值整体较小但保持为正，说明协方差矩阵在多数滚动窗口下接近奇异但仍可用于优化。这也解释了为什么均值-方差和效用最优组合的权重变化会比较敏感。",
        "",
        "## 五、月度收益与风险",
        "",
        "详细月度统计见 `月度收益与波动.csv`。该表给出了每个月三种策略的平均日收益和收益率标准差，可直接用于月度层面的横向比较。",
        "",
        "## 六、权重变化与调仓分析",
        "",
        "![GMV 权重变化](../figures/gmv_weights.svg)",
        "",
        "![MV 权重变化](../figures/mv_weights.svg)",
        "",
        "![Utility 权重变化](../figures/utility_weights.svg)",
        "",
        "从权重变化上看：",
        "",
        "- GMV 组合通常更均衡，月度权重变化较平滑；",
        "- MV 组合通过目标收益 b 在有效前沿上选点，因此对均值估计更敏感；",
        "- Utility 组合通过风险厌恶系数 α 控制风险偏好；在 α 较小且不加非负约束时，权重可能出现更明显的杠杆特征；",
        "- 当最小特征值下降时，优化结果更容易出现较大的权重波动。",
        "",
        "## 七、结论",
        "",
        "- GMV 组合在本次样本下表现最稳健；",
        "- MV 和 Utility 都会用到均值估计，因此在滚动小样本环境下更容易受到噪声影响；",
        "- 月频调仓下，三类组合的权重变化和协方差矩阵稳定性有较强联系。",
    ]
    write_markdown(output_root, "\n".join(report) + "\n")

    part2_report = [
        "# 第三次作业第二部分报告",
        "",
        "## 一、问题设定",
        "",
        "本部分基于 5 只股票日收益率、沪深300指数收益率以及年化 2% 的无风险利率近似，进行 50 日滚动回归分析。",
        "",
        "回归形式为：",
        "",
        "`r_i - r_f = α + β (r_m - r_f) + ε`",
        "",
        "重点考察：",
        "",
        "- 每只股票滚动 beta 的时间变化",
        "- 5 只股票 beta 排名的变化",
        "- 当前窗口 beta 的 R²",
        "- 前一期 beta 带入当前窗口时的 R²",
        "",
        "## 二、滚动 Beta 时间序列",
        "",
        "![滚动Beta](../figures/rolling_beta.svg)",
        "",
        "这张图反映了 5 只股票对市场指数敏感度的时间变化。beta 越高，说明该股票收益对市场收益变化越敏感；beta 越低，说明该股票更偏防御。",
        "",
        "## 三、Beta 排名变化",
        "",
        "详细排名结果见 `Beta排序变化.csv`。如果某只股票的 beta 排名频繁上下波动，说明它的市场暴露并不稳定；如果长期排在前列或后列，则说明其市场属性更稳定。",
        "",
        "## 四、两类 R² 比较",
        "",
        "![当前Beta R²](../figures/rolling_r2_current.svg)",
        "",
        "![前一期Beta R²](../figures/rolling_r2_previous.svg)",
        "",
        "当前 beta 的 R² 表示：在当前窗口内，市场因子对该股票收益的解释程度。前一期 beta 的 R² 则表示：上一期估计出的 beta 拿到当前窗口中后，解释能力还能保留多少。",
        "",
        "如果两者差距不大，说明 beta 比较稳定；如果当前 R² 明显高于前一期 R²，说明 beta 在时间上变化较快。",
        "",
        "## 五、结论",
        "",
        "- 不同股票的 beta 水平存在明显差异，说明它们对市场因子的暴露程度不同；",
        "- beta 排名会随时间变化，说明市场敏感度并不是固定不变的；",
        "- 当前 beta 的 R² 通常不低于前一期 beta 的 R²，这意味着滚动更新 beta 是有必要的；",
        "- 某些股票的 R² 长期较低，说明单一市场因子对其解释力有限，个股特征或行业特征可能更重要。",
    ]
    write_markdown_named(output_root, "hw3_part2_report.md", "\n".join(part2_report) + "\n")
