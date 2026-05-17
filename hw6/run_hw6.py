from __future__ import annotations

import argparse
import sys

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from hw6_analysis.config import (  # noqa: E402
    DATA_ZIP,
    DEFAULT_SYMBOL,
    FIGURES_DIR,
    MAIN_HORIZON,
    PLOT_POINTS,
    PREDICTION_HORIZONS,
    TABLES_DIR,
    ensure_output_dirs,
)
from hw6_analysis.data import load_contract, sample_info  # noqa: E402
from hw6_analysis.evaluation import evaluate_signals, horizon_best_summary  # noqa: E402
from hw6_analysis.features import build_feature_frame  # noqa: E402
from hw6_analysis.plots import plot_figure_3_1_to_3_9  # noqa: E402
from hw6_analysis.signals import method_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HW6 timing-signal experiment.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Contract key in future_data.pkl.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    raw_df = load_contract(DATA_ZIP, args.symbol)
    features = build_feature_frame(raw_df)
    summary, horizon_summary = evaluate_signals(features, PREDICTION_HORIZONS, MAIN_HORIZON)
    best_by_horizon = horizon_best_summary(horizon_summary)
    figure_path = plot_figure_3_1_to_3_9(args.symbol, features, summary, FIGURES_DIR, PLOT_POINTS)

    sample_info(raw_df, args.symbol).to_csv(TABLES_DIR / "sample_info.csv", index=False)
    method_catalog().to_csv(TABLES_DIR / "method_catalog.csv", index=False)
    summary.to_csv(TABLES_DIR / "predictive_summary.csv", index=False)
    horizon_summary.to_csv(TABLES_DIR / "horizon_ic_summary.csv", index=False)
    best_by_horizon.to_csv(TABLES_DIR / "horizon_best_signals.csv", index=False)

    print(f"symbol: {args.symbol}")
    print(f"figure: {figure_path}")
    print(f"tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
