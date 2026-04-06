# HW4

第四次作业实现目录，使用 `hw4/HS300_data.zip` 这套新版日频沪深300面板数据。

说明：

- 仓库默认不提交体积较大的原始数据压缩包；
- 代码会优先读取 `quantitative-trading/hw4/HS300_data.zip`；
- 如果该文件不存在，则自动回退到工作区根目录下的 `hw4/HS300_data.zip`。

当前实现包含：

- `run_assignment4.py`：作业入口脚本
- `src/assignment4/`：数据读取、因子构造、IC、回测、报告生成
- `outputs/`：运行后生成的表格、图和 Markdown 报告

当前默认策略设定：

- 调仓频率：月频
- 训练集：2016-2020
- 验证集：2021-2022
- 测试集：2023-01-01 之后
- 技术因子：20日动量
- 基本面因子：BP = 1 / PB
- 组合构造：每期选信号最高的前 20% 股票，等权持有

运行方式：

```bash
source /Users/quanchen/miniconda3/bin/activate quant
python3 run_assignment4.py
```

运行后会在 `outputs/` 下生成：

- `tables/`：因子值、权重、IC、策略日收益率、绩效汇总
- `figures/`：累计 IC 和净值曲线
- `report/hw4_report.md`：自动生成的作业报告草稿
