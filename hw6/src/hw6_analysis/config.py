from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ZIP = ROOT / "future_data.zip"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

DEFAULT_SYMBOL = "MA9999.XZCE"
PLOT_POINTS = 1500
PREDICTION_HORIZONS = [5, 15, 30, 60]
MAIN_HORIZON = 5


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
