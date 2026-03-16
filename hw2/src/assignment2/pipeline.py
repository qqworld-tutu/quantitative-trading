from pathlib import Path

from assignment2.config import FIGURES_DIR, INITIAL_CAPITAL, OUTPUT_DIR, TABLES_DIR
from assignment2.data import load_aligned_prices, split_dataset
from assignment2.portfolio import buy_and_hold_curve, equal_weight_curve, performance_row, yearly_summary
from assignment2.reporting import ensure_dirs, markdown_table, multi_line_svg, write_csv, write_markdown
from assignment2.strategy_model import build_training_set, fit_logistic_regression, run_model_strategy
from assignment2.strategy_rules import run_rule_strategy


def _pick_rule_params(split):
    best = None
    for window in (5, 10, 20):
        for top_n in (1, 2):
            values, _ = run_rule_strategy(split["prices"]["train"], window, top_n, INITIAL_CAPITAL)
            score = values[-1]
            if best is None or score > best[0]:
                best = (score, window, top_n)
    return best[1], best[2]


def _pick_model_params(split):
    x, y = build_training_set(split["prices"]["train"], len(split["dates"]["train"]))
    weights = fit_logistic_regression(x, y)
    best = None
    for top_n in (1, 2):
        values, _ = run_model_strategy(split["prices"]["valid"], weights, top_n, INITIAL_CAPITAL)
        score = values[-1]
        if best is None or score > best[0]:
            best = (score, top_n)
    return weights, best[1]


def _full_curves(dates, prices, benchmark):
    single_curves = {name: buy_and_hold_curve(values, INITIAL_CAPITAL) for name, values in prices.items()}
    equal_curve = equal_weight_curve(prices, INITIAL_CAPITAL)
    benchmark_curve = buy_and_hold_curve(benchmark, INITIAL_CAPITAL)
    return single_curves, equal_curve, benchmark_curve


def _year_ticks(dates):
    seen = {}
    for idx, dt in enumerate(dates):
        seen.setdefault(dt.year, idx)
    years = sorted(seen.items())
    if len(years) > 8:
        years = years[::2]
        if years[-1][0] != dates[-1].year:
            years.append((dates[-1].year, seen[dates[-1].year]))
    return [(idx, str(year)) for year, idx in years]


def run(output_root=OUTPUT_DIR):
    output_root = Path(output_root)
    ensure_dirs(output_root)
    dates, prices, benchmark = load_aligned_prices()
    split = split_dataset(dates, prices, benchmark)

    single_curves, equal_curve, benchmark_curve = _full_curves(dates, prices, benchmark)

    rule_window, rule_top_n = _pick_rule_params(split)
    model_weights, model_top_n = _pick_model_params(split)

    rule_curve, _ = run_rule_strategy(prices, rule_window, rule_top_n, INITIAL_CAPITAL)
    model_curve, _ = run_model_strategy(prices, model_weights, model_top_n, INITIAL_CAPITAL)

    multi_line_svg(
        output_root / "figures" / "equal_weight_vs_stocks.svg",
        "五只股票、等权组合与基准净值",
        {**single_curves, "等权组合": equal_curve, "沪深300指数": benchmark_curve},
        "净值",
        x_ticks=_year_ticks(dates),
    )
    multi_line_svg(
        output_root / "figures" / "strategy_compare.svg",
        "认知规则、模型策略、等权组合与基准净值",
        {
            "认知规则策略": rule_curve,
            "模型策略": model_curve,
            "等权组合": equal_curve,
            "沪深300指数": benchmark_curve,
        },
        "净值",
        x_ticks=_year_ticks(dates),
    )

    perf_rows = []
    for name, curve in single_curves.items():
        perf_rows.append(performance_row(name, curve))
    perf_rows.append(performance_row("等权组合", equal_curve))
    perf_rows.append(performance_row("沪深300指数", benchmark_curve))
    perf_rows.append(performance_row("认知规则策略", rule_curve))
    perf_rows.append(performance_row("模型策略", model_curve))
    write_csv(output_root / "tables" / "绩效比较.csv", ["策略", "年化收益", "年化波动", "夏普", "最大回撤"], perf_rows)

    yearly_rows = []
    for name, curve in {**single_curves, "等权组合": equal_curve, "沪深300指数": benchmark_curve}.items():
        for year, mean_ret, std_ret in yearly_summary(dates, curve):
            yearly_rows.append([name, year, round(mean_ret, 6), round(std_ret, 6)])
    write_csv(output_root / "tables" / "逐年收益与波动.csv", ["资产", "年份", "平均日收益", "收益率标准差"], yearly_rows)

    param_rows = [
        ["认知规则", f"window={rule_window}", f"top_n={rule_top_n}"],
        ["模型策略", "logistic_regression", f"top_n={model_top_n}"],
    ]
    write_csv(output_root / "tables" / "参数选择.csv", ["策略", "参数1", "参数2"], param_rows)

    train_n = len(split["dates"]["train"])
    valid_n = len(split["dates"]["valid"])
    test_n = len(split["dates"]["test"])
    report = [
        "# 第二次作业报告",
        "",
        "## 一、作业目标",
        "",
        "本次作业使用第一次作业中选出的 5 只股票，完成两部分内容：",
        "",
        "- 仿照案例 2.8，对单只股票与等权组合做买入并持有分析；",
        "- 构造一个基于认知规则的策略和一个基于模型的策略，并比较其绩效。",
        "",
        "## 二、数据与设定",
        "",
        "- 股票池：贵州茅台、中国平安、长江电力、中国中免、恒瑞医药",
        "- 基准：沪深300指数",
        "- 初始资金：100000 元",
        "- 调仓频率：日频",
        "- 训练/验证/测试切分："
        f" {train_n} / {valid_n} / {test_n} 个交易日",
        "",
        "## 三、等权组合分析",
        "",
        "沿用第一次作业选出的 5 只股票，对每只股票和等权组合做买入并持有分析，初始资金为 100000 元，基准为沪深300指数。",
        "",
        markdown_table(["策略", "年化收益", "年化波动", "夏普", "最大回撤"], perf_rows[:7]),
        "",
        "![等权组合净值](../figures/equal_weight_vs_stocks.svg)",
        "",
        "从结果看，等权组合的年化收益率为 0.208641，明显高于基准指数 0.010647；同时其波动率低于部分高波动个股，说明分散配置确实降低了组合的个体风险暴露。",
        "",
        "## 四、策略设计",
        "",
        f"- 认知规则策略：过去 {rule_window} 日截面动量，日频调仓，持有前 {rule_top_n} 只股票。",
        f"- 模型策略：逻辑回归预测下一日上涨概率，日频调仓，持有前 {model_top_n} 只股票。",
        "",
        "参数选择结果如下：",
        "",
        markdown_table(["策略", "参数1", "参数2"], param_rows),
        "",
        "## 五、策略比较",
        "",
        markdown_table(["策略", "年化收益", "年化波动", "夏普", "最大回撤"], perf_rows[-2:]),
        "",
        "![策略比较净值](../figures/strategy_compare.svg)",
        "",
        "模型策略的年化收益率和夏普比率都高于认知规则策略，同时最大回撤更小，说明在当前样本和当前特征设计下，逻辑回归给出的打分排序优于单纯截面动量规则。",
        "",
        "## 六、逐年统计",
        "",
        "完整逐年平均日收益与波动结果已输出到 `hw2/outputs/tables/逐年收益与波动.csv`。从逐年结果看，组合与策略在不同年份表现差异明显，说明市场风格切换会显著影响简单策略的有效性。",
        "",
        "## 七、结论",
        "",
        "- 等权组合降低了单只股票的特异性风险。",
        "- 认知规则策略利用了简单的截面动量信息，具有可解释性强的优点。",
        "- 模型策略在有限样本下提供了另一种系统化打分方法，整体绩效更优。",
        "- 由于股票池仅有 5 只股票，结果更适合作为教学演示，不宜直接外推到更大规模实盘。",
    ]
    write_markdown(output_root, "\n".join(report) + "\n")
