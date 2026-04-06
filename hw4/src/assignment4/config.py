from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_ZIP = ROOT / "HS300_data.zip"
FALLBACK_DATA_ZIP = ROOT.parents[1] / "hw4" / "HS300_data.zip"
DATA_ZIP = LOCAL_DATA_ZIP if LOCAL_DATA_ZIP.exists() else FALLBACK_DATA_ZIP
OUTPUT_DIR = ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "report"

TRAIN_END = "2020-12-31"
VALIDATION_END = "2022-12-31"
TEST_START = "2023-01-01"

MOMENTUM_WINDOW = 20
TOP_SELECTION_RATIO = 0.20
TRADING_DAYS_PER_YEAR = 252
INITIAL_CAPITAL = 1.0
WINSORIZE_PCT = 0.01

FACTOR_SPECS = {
    "momentum20": {
        "label": "20日动量",
        "description": "Momentum20 = Close_t / Close_(t-20) - 1",
        "direction": "数值越高，代表近期价格越强",
    },
    "bp": {
        "label": "BP(1/PB)",
        "description": "BP = 1 / pb_ratio",
        "direction": "数值越高，代表相对更便宜",
    },
}


def split_name(date_text):
    if date_text <= TRAIN_END:
        return "train"
    if date_text <= VALIDATION_END:
        return "validation"
    return "test"
