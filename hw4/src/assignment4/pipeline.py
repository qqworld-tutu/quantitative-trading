from pathlib import Path

from assignment4.backtest import (
    compute_daily_strategy_returns,
    compute_forward_returns,
    compute_ic_series,
    compute_weight_panels,
    cumulative_sum,
    nav_curve,
    summarize_performance,
)
from assignment4.config import FACTOR_SPECS, INITIAL_CAPITAL, OUTPUT_DIR, TEST_START, split_name
from assignment4.data import load_market_data
from assignment4.factors import build_factor_panels
from assignment4.reporting import ensure_dirs, markdown_table, multi_line_svg, write_csv, write_markdown_named


def sparse_ticks(labels, count=6):
    if not labels:
        return []
    step = max(1, len(labels) // count)
    ticks = [(idx, label) for idx, label in enumerate(labels) if idx % step == 0]
    if ticks[-1][0] != len(labels) - 1:
        ticks.append((len(labels) - 1, labels[-1]))
    return ticks


def filter_rows_by_split(rows, split):
    return [(date_text, value) for date_text, value in rows if split_name(date_text) == split]


def find_summary(summary_map, factor_name, split):
    for row in summary_map[factor_name]:
        if row["split"] == split:
            return row
    raise KeyError((factor_name, split))


def write_panel_csv(path, data, codes, dates, field_getters):
    headers = ["date", "code"] + [column for column, _ in field_getters]
    rows = []
    for date_text in dates:
        for code in codes:
            row = [date_text, code]
            for _, getter in field_getters:
                value = getter(date_text, code)
                row.append("" if value is None else round(value, 10))
            rows.append(row)
    write_csv(path, headers, rows)


def run(output_root=OUTPUT_DIR):
    output_root = Path(output_root)
    ensure_dirs(output_root)

    data = load_market_data()
    rebalance_dates, factor_panels = build_factor_panels(data)
    forward_returns = compute_forward_returns(data, rebalance_dates + [data.month_end_dates[-1]])

    ic_results = {}
    weight_panels = {}
    daily_returns = {}
    performance_summaries = {}
    for factor_name in FACTOR_SPECS:
        ic_rows = compute_ic_series(factor_panels[factor_name], forward_returns)
        weights = compute_weight_panels(factor_panels[factor_name], rebalance_dates)
        strategy_returns = compute_daily_strategy_returns(data, weights, rebalance_dates + [data.month_end_dates[-1]])
        ic_results[factor_name] = ic_rows
        weight_panels[factor_name] = weights
        daily_returns[factor_name] = strategy_returns
        performance_summaries[factor_name] = summarize_performance(strategy_returns, ic_rows)

    write_panel_csv(
        output_root / "tables" / "factor_values_all.csv",
        factor_panels,
        data.codes,
        rebalance_dates,
        [
            ("momentum20", lambda date_text, code: factor_panels["momentum20"][date_text]["raw"].get(code)),
            ("bp", lambda date_text, code: factor_panels["bp"][date_text]["raw"].get(code)),
        ],
    )
    write_panel_csv(
        output_root / "tables" / "factor_values_test.csv",
        factor_panels,
        data.codes,
        [date_text for date_text in rebalance_dates if date_text >= TEST_START],
        [
            ("momentum20", lambda date_text, code: factor_panels["momentum20"][date_text]["raw"].get(code)),
            ("bp", lambda date_text, code: factor_panels["bp"][date_text]["raw"].get(code)),
        ],
    )
    write_panel_csv(
        output_root / "tables" / "factor_weights_all.csv",
        weight_panels,
        data.codes,
        rebalance_dates,
        [
            ("momentum20", lambda date_text, code: weight_panels["momentum20"][date_text].get(code, 0.0)),
            ("bp", lambda date_text, code: weight_panels["bp"][date_text].get(code, 0.0)),
        ],
    )
    write_panel_csv(
        output_root / "tables" / "factor_weights_test.csv",
        weight_panels,
        data.codes,
        [date_text for date_text in rebalance_dates if date_text >= TEST_START],
        [
            ("momentum20", lambda date_text, code: weight_panels["momentum20"][date_text].get(code, 0.0)),
            ("bp", lambda date_text, code: weight_panels["bp"][date_text].get(code, 0.0)),
        ],
    )

    ic_headers = ["date"] + list(FACTOR_SPECS)
    ic_rows_all = []
    for date_text in rebalance_dates:
        if date_text not in forward_returns:
            continue
        ic_rows_all.append([date_text] + [round(dict(ic_results[name])[date_text], 10) for name in FACTOR_SPECS])
    write_csv(output_root / "tables" / "factor_ic_all.csv", ic_headers, ic_rows_all)
    write_csv(
        output_root / "tables" / "factor_ic_test.csv",
        ic_headers,
        [row for row in ic_rows_all if row[0] >= TEST_START],
    )

    strategy_headers = ["date"] + list(FACTOR_SPECS)
    strategy_rows_all = []
    all_dates = sorted({date_text for rows in daily_returns.values() for date_text, _ in rows})
    for date_text in all_dates:
        row = [date_text]
        for factor_name in FACTOR_SPECS:
            mapping = dict(daily_returns[factor_name])
            row.append(round(mapping.get(date_text, 0.0), 10))
        strategy_rows_all.append(row)
    write_csv(output_root / "tables" / "strategy_daily_returns_all.csv", strategy_headers, strategy_rows_all)
    write_csv(
        output_root / "tables" / "strategy_daily_returns_test.csv",
        strategy_headers,
        [row for row in strategy_rows_all if row[0] >= TEST_START],
    )

    summary_rows = []
    for factor_name, summary_list in performance_summaries.items():
        for summary in summary_list:
            summary_rows.append(
                [
                    FACTOR_SPECS[factor_name]["label"],
                    summary["split"],
                    summary["days"],
                    round(summary["avg_daily_return"], 8),
                    round(summary["daily_vol"], 8),
                    round(summary["annualized_return"], 8),
                    round(summary["annualized_vol"], 8),
                    round(summary["sharpe"], 8),
                    round(summary["max_drawdown"], 8),
                    round(summary["ic_mean"], 8),
                    round(summary["ic_ir"], 8),
                ]
            )
    write_csv(
        output_root / "tables" / "performance_summary.csv",
        [
            "factor",
            "split",
            "days",
            "avg_daily_return",
            "daily_vol",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "max_drawdown",
            "ic_mean",
            "ic_ir",
        ],
        summary_rows,
    )

    ic_curves = {}
    for factor_name, rows in ic_results.items():
        ic_curves[FACTOR_SPECS[factor_name]["label"]] = [value for _, value in cumulative_sum(rows)]
    multi_line_svg(
        output_root / "figures" / "cumulative_ic.svg",
        "累计IC曲线",
        ic_curves,
        "累计IC",
        x_ticks=sparse_ticks([date_text[:7] for date_text in rebalance_dates]),
    )

    train_ic_series = {}
    train_cumulative_ic = {}
    train_labels = None
    for factor_name, rows in ic_results.items():
        train_rows = filter_rows_by_split(rows, "train")
        if train_labels is None:
            train_labels = [date_text[:7] for date_text, _ in train_rows]
        train_ic_series[FACTOR_SPECS[factor_name]["label"]] = [value for _, value in train_rows]
        train_cumulative_ic[FACTOR_SPECS[factor_name]["label"]] = [value for _, value in cumulative_sum(train_rows)]
    multi_line_svg(
        output_root / "figures" / "train_ic_series.svg",
        "训练集IC时间序列",
        train_ic_series,
        "IC",
        x_ticks=sparse_ticks(train_labels or []),
    )
    multi_line_svg(
        output_root / "figures" / "train_cumulative_ic.svg",
        "训练集累计IC曲线",
        train_cumulative_ic,
        "累计IC",
        x_ticks=sparse_ticks(train_labels or []),
    )

    nav_curves = {}
    for factor_name, rows in daily_returns.items():
        nav_curves[FACTOR_SPECS[factor_name]["label"]] = [value for _, value in nav_curve(rows, INITIAL_CAPITAL)]
    multi_line_svg(
        output_root / "figures" / "nav_curves_all.svg",
        "全样本净值曲线",
        nav_curves,
        "净值",
        x_ticks=sparse_ticks([date_text[:7] for date_text, _ in daily_returns["momentum20"]]),
    )

    nav_curves_test = {}
    for factor_name, rows in daily_returns.items():
        test_rows = [item for item in rows if item[0] >= TEST_START]
        nav_curves_test[FACTOR_SPECS[factor_name]["label"]] = [value for _, value in nav_curve(test_rows, INITIAL_CAPITAL)]
    multi_line_svg(
        output_root / "figures" / "nav_curves_test.svg",
        "测试集净值曲线",
        nav_curves_test,
        "净值",
        x_ticks=sparse_ticks([date_text[:7] for date_text, _ in daily_returns["momentum20"] if date_text >= TEST_START]),
    )

    for factor_name, rows in daily_returns.items():
        test_rows = [item for item in rows if item[0] >= TEST_START]
        label = FACTOR_SPECS[factor_name]["label"]
        multi_line_svg(
            output_root / "figures" / f"{factor_name}_test_nav.svg",
            f"{label}测试集净值曲线",
            {label: [value for _, value in nav_curve(test_rows, INITIAL_CAPITAL)]},
            "净值",
            x_ticks=sparse_ticks([date_text[:7] for date_text, _ in test_rows]),
        )

    summary_table = markdown_table(
        ["因子", "区间", "年化收益", "年化波动", "夏普", "最大回撤", "IC均值", "ICIR"],
        [
            [
                FACTOR_SPECS[factor_name]["label"],
                summary["split"],
                round(summary["annualized_return"], 4),
                round(summary["annualized_vol"], 4),
                round(summary["sharpe"], 4),
                round(summary["max_drawdown"], 4),
                round(summary["ic_mean"], 4),
                round(summary["ic_ir"], 4),
            ]
            for factor_name, summaries in performance_summaries.items()
            for summary in summaries
        ],
    )
    momentum_validation = find_summary(performance_summaries, "momentum20", "validation")
    momentum_test = find_summary(performance_summaries, "momentum20", "test")
    bp_validation = find_summary(performance_summaries, "bp", "validation")
    bp_test = find_summary(performance_summaries, "bp", "test")
    data_description = (
        f"本次实验使用新版沪深300日频面板数据，共 {len(data.codes)} 只股票、"
        f"{len(data.dates)} 个交易日、{len(rebalance_dates)} 个有效月度调仓点。"
    )
    factor_analysis_text = (
        f"从样本外表现看，20日动量因子在验证集的 IC 均值为 "
        f"{momentum_validation['ic_mean']:.4f}，到测试集回升到 {momentum_test['ic_mean']:.4f}，"
        f"说明它对市场环境更敏感、稳定性一般；BP 因子在验证集和测试集的 IC 均值分别为 "
        f"{bp_validation['ic_mean']:.4f} 和 {bp_test['ic_mean']:.4f}，方向一致且均为正，"
        "表现出更好的跨样本稳定性。"
    )
    pnl_analysis_text = (
        f"测试集回测中，20日动量的年化收益率为 {momentum_test['annualized_return']:.2%}，"
        f"略高于 BP 的 {bp_test['annualized_return']:.2%}；但 BP 的夏普比率为 "
        f"{bp_test['sharpe']:.4f}，略高于动量的 {momentum_test['sharpe']:.4f}，"
        f"且最大回撤更低（{bp_test['max_drawdown']:.2%} 对比 {momentum_test['max_drawdown']:.2%}）。"
        "这意味着 BP 在测试集里的收益风险比更均衡，而动量更依赖阶段性趋势行情。"
    )
    report_lines = [
        "# 第四次作业报告",
        "",
        "## 一、研究目的",
        "",
        "本次实验基于新版 `hw4/HS300_data.zip` 数据，目标是完整演示“原始面板数据 -> 因子构造 -> 因子评价 -> 单因子组合回测 -> 结果分析”的标准流程。",
        "",
        "相比直接寻找高收益策略，本次作业更强调两个问题：",
        "",
        "- 一个因子在训练集内是否具有可解释的预测能力；",
        "- 一个因子在验证集和测试集中是否还能维持相对稳定的表现。",
        "",
        "本报告分别选择一个技术面因子和一个基本面因子进行比较：",
        "",
        "- 技术因子：20日动量",
        "- 基本面因子：BP = 1 / PB",
        "",
        "## 二、数据说明与样本划分",
        "",
        data_description,
        "",
        "数据文件以“每个交易日一个 CSV”的方式组织，每个文件提供当日 300 只股票的价格、估值、盈利能力等横截面信息。为了和课程要求一致，本实验采用月频调仓，并把月末最后一个交易日作为调仓时点。",
        "",
        "- 训练集：2016-01-04 至 2020-12-31",
        "- 验证集：2021-01-01 至 2022-12-31",
        "- 测试集：2023-01-01 至 2026-01-20",
        "",
        "设置验证集的目的，是避免只凭训练集结果选因子。训练集负责提出和初筛因子，验证集负责检查因子的样本外稳定性，测试集只在最终设定固定后进行一次完整回测。",
        "",
        "## 三、因子定义与经济含义",
        "",
        "### 1. 技术因子：20日动量",
        "",
        "定义：`Momentum20_t = close_t / close_(t-20) - 1`。",
        "",
        "这个因子衡量股票过去约一个月的价格强弱。若一个股票在过去 20 个交易日上涨更多，则该因子值更高。它代表的核心假设是“短期趋势延续”：近期表现强的股票在下一期可能继续相对强势。",
        "",
        "选择这个因子的原因是：",
        "",
        "- 公式简单，几乎不涉及额外清洗，适合展示技术因子的基本研究流程；",
        "- 数据来自前复权价格，能够减少分红送转对动量计算的扭曲；",
        "- 动量因子在课堂和文献中都很常见，解释成本较低。",
        "",
        "### 2. 基本面因子：BP(1/PB)",
        "",
        "定义：`BP_t = 1 / pb_ratio_t`，也就是账面市值比（Book-to-Price）的简化形式。",
        "",
        "PB 越低，代表股票相对净资产越“便宜”；取倒数以后，BP 越高就代表股票越便宜。因此该因子代表典型的“价值”思想：市场可能对低估值股票存在低估，未来存在均值回归或价值修复的机会。",
        "",
        "选择 BP 而不是直接使用 PE，主要是因为新版数据中 `pb_ratio` 更稳定、极端值更少，更适合作为第一次完整流程练习的基本面因子。",
        "",
        "### 3. 因子预处理",
        "",
        "为了让两个因子在横截面上可比较，实验中对每个调仓日的原始因子值做了两步预处理：",
        "",
        "- 先做 1% 和 99% 分位数去极值，削弱极端异常值的影响；",
        "- 再做横截面 z-score 标准化，使因子值都转成“相对高低”的可比信号。",
        "",
        "## 四、实验设计与回测方法",
        "",
        "1. 将压缩包中的每个交易日 CSV 读取为横截面，并整理出 `close`、`pb_ratio`、`paused` 三个核心字段。",
        "2. 取每月最后一个交易日作为调仓日。",
        "3. 在每个调仓日计算两个原始因子：`Momentum20` 与 `BP`。",
        "4. 对每个调仓日的因子截面做 1% 去极值和 z-score 标准化，作为组合排序信号。",
        "5. 用调仓日到下一调仓日的收益作为 forward return，计算 Spearman IC。",
        "6. 每个因子都采用“前20%等权持有”的单因子多头组合，日频跟踪净值。",
        "",
        "其中 IC 使用 Spearman 秩相关系数，是因为它更关注排序关系，而不是具体数值差距，更符合单因子选股里“谁排前面更重要”的思路。",
        "",
        "组合构造方面，本实验采用非常直接的单因子多头框架：在每个调仓日按因子分数从高到低排序，选取前 20% 股票等权持有，直到下一个月末再调仓。这种设定虽然简化，但足以展示因子是否能转化成可交易的组合表现。",
        "",
        "## 五、因子定义汇总",
        "",
        "| 因子 | 定义 | 解释 |",
        "| --- | --- | --- |",
        "| 20日动量 | `close_t / close_(t-20) - 1` | 近期价格越强，分数越高 |",
        "| BP | `1 / pb_ratio` | 越便宜的股票，分数越高 |",
        "",
        "## 六、结果摘要",
        "",
        summary_table,
        "",
        "### 1. 训练集 IC 表现",
        "",
        "训练集的 IC 图更直接对应“因子筛选”这一步。这里重点看两个问题：",
        "",
        "- IC 时间序列是否长期围绕 0 大幅震荡",
        "- 累计 IC 是否有相对稳定的趋势",
        "",
        "![训练集IC时间序列](../figures/train_ic_series.svg)",
        "",
        "![训练集累计IC](../figures/train_cumulative_ic.svg)",
        "",
        "从训练集图形可以看出，两类因子都不是“单边稳定上升”的理想状态，但 BP 的累计 IC 更平滑，而动量的 IC 波动更大。这说明动量因子对市场环境切换更敏感，容易在不同阶段出现信号反复。",
        "",
        "### 2. 全样本累计 IC",
        "",
        "![累计IC](../figures/cumulative_ic.svg)",
        "",
        factor_analysis_text,
        "",
        "### 3. 测试集 PnL / 净值",
        "",
        "![测试集净值](../figures/nav_curves_test.svg)",
        "",
        "![20日动量测试集净值](../figures/momentum20_test_nav.svg)",
        "",
        "![BP测试集净值](../figures/bp_test_nav.svg)",
        "",
        pnl_analysis_text,
        "",
        "如果只看年化收益率，20日动量略占优势；但如果综合 IC 稳定性、夏普比率和最大回撤，BP 更接近一个“更稳”的基本面因子。这种现象也符合经验认识：价值因子往往节奏较慢，但样本外稳定性有时优于短周期技术信号。",
        "",
        "### 4. 全样本净值",
        "",
        "![全样本净值](../figures/nav_curves_all.svg)",
        "",
        "## 七、结论与讨论",
        "",
        "本次实验说明：",
        "",
        "- 单因子研究不能只看训练集结果，验证集和测试集的表现更能说明因子是否稳定；",
        "- 20日动量因子实现简单、逻辑清晰，但对市场状态变化更敏感；",
        "- BP 因子代表估值修复思路，在本次样本中表现出更好的样本外一致性；",
        "- 即便是两个非常基础的因子，只要流程规范，也可以完整展示因子研究的核心步骤。",
        "",
        "本报告仍有几个可继续改进的方向：",
        "",
        "- 进一步考虑停牌、涨跌停、调仓冲击和交易成本；",
        "- 对基本面因子加入公告时点处理，进一步减少潜在的未来信息问题；",
        "- 增加行业中性、市值中性或分组回测，检验因子是否只是暴露在其他风格因子上。",
        "",
        "## 八、输出文件",
        "",
        "- `tables/factor_values_all.csv`：全样本调仓日因子原始值",
        "- `tables/factor_values_test.csv`：测试集调仓日因子原始值",
        "- `tables/factor_weights_all.csv`：全样本调仓日权重",
        "- `tables/factor_weights_test.csv`：测试集调仓日权重",
        "- `tables/factor_ic_all.csv`：全样本 IC 时间序列",
        "- `tables/factor_ic_test.csv`：测试集 IC 时间序列",
        "- `tables/strategy_daily_returns_all.csv`：全样本策略日收益",
        "- `tables/strategy_daily_returns_test.csv`：测试集策略日收益",
        "- `tables/performance_summary.csv`：训练/验证/测试分区指标",
        "",
        "## 九、说明",
        "",
        "当前版本优先强调流程清晰和作业可解释性，因此没有引入行业中性、交易成本、财报公告滞后修正等更复杂设定。后续如果你想继续打磨，我们可以在这个框架上继续扩展。",
    ]
    write_markdown_named(output_root, "hw4_report.md", "\n".join(report_lines) + "\n")
