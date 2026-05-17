from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from hw6_analysis.filters import hp_trend


def plot_figure_3_1_to_3_9(symbol: str, features, summary, output_dir: Path, plot_points: int) -> Path:
    sample = features.tail(plot_points).copy()
    close = sample["close"]
    sample["hp_trend"] = hp_trend(close)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(sample.index, close, label="Close", linewidth=1.0)
    axes[0].plot(sample.index, sample["sma20"], label="SMA20", linewidth=1.0)
    axes[0].plot(sample.index, sample["sma60"], label="SMA60", linewidth=1.0)
    axes[0].set_title("1. Moving average")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(sample.index, sample["macd"], label="MACD", linewidth=1.0)
    axes[1].plot(sample.index, sample["macd_signal"], label="Signal", linewidth=1.0)
    axes[1].bar(sample.index, sample["macd_hist"], width=0.01, alpha=0.35, label="Hist")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_title("2. MACD")
    axes[1].legend(loc="upper left", fontsize=8)

    axes[2].plot(sample.index, sample["rsi14"], color="tab:green", linewidth=1.0)
    axes[2].axhline(70, color="tab:red", linestyle="--", linewidth=0.8)
    axes[2].axhline(30, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[2].axhline(50, color="gray", linestyle=":", linewidth=0.8)
    axes[2].set_ylim(0, 100)
    axes[2].set_title("3. RSI")

    axes[3].plot(sample.index, sample["kdj_k"], label="K", linewidth=1.0)
    axes[3].plot(sample.index, sample["kdj_d"], label="D", linewidth=1.0)
    axes[3].plot(sample.index, sample["kdj_j"], label="J", linewidth=0.9)
    axes[3].axhline(80, color="tab:red", linestyle="--", linewidth=0.8)
    axes[3].axhline(20, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[3].set_title("4. KDJ")
    axes[3].legend(loc="upper left", fontsize=8)

    axes[4].plot(sample.index, close, label="Close", linewidth=0.9, alpha=0.45)
    axes[4].plot(sample.index, sample["kalman"], label="Kalman", linewidth=1.0)
    axes[4].plot(sample.index, sample["hp_trend"], label="HP trend", linewidth=1.0)
    axes[4].set_title("5. Kalman and HP filtering")
    axes[4].legend(loc="upper left", fontsize=8)

    plot_summary = summary.copy()
    plot_summary["abs_ic"] = plot_summary["pearson_ic"].abs()
    plot_summary = plot_summary.sort_values("abs_ic", ascending=False)
    axes[5].barh(
        plot_summary["signal"],
        plot_summary["pearson_ic"],
        color=np.where(plot_summary["pearson_ic"] >= 0, "#4c72b0", "#c44e52"),
    )
    axes[5].axvline(0, color="gray", linewidth=0.8)
    axes[5].set_title("6. Pearson IC for 5-minute return")

    for ax in axes[:5]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.tick_params(axis="x", labelsize=8)
    axes[5].tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Core timing indicators and filters for {symbol}", fontsize=16)
    path = output_dir / "figure_3_1_to_3_9_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
