from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ZIP = ROOT / "量化交易2026春-第1次作业数据.zip"
FALLBACK_DATA_ZIP = ROOT.parent / "hw1" / "量化交易2026春-第1次作业数据.zip"
DATA_ZIP = LOCAL_DATA_ZIP if LOCAL_DATA_ZIP.exists() else FALLBACK_DATA_ZIP
OUTPUT_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"
WINDOW = 50
ANNUAL_RF = 0.02
DAILY_RF = ANNUAL_RF / 252
INITIAL_CAPITAL = 100000.0
UTILITY_RISK_AVERSION = 10.0
MV_TARGET_METHOD = "mean_of_mu"
STOCKS = [
    ("600519.SH", "贵州茅台"),
    ("601318.SH", "中国平安"),
    ("600900.SH", "长江电力"),
    ("601888.SH", "中国中免"),
    ("600276.SH", "恒瑞医药"),
]
