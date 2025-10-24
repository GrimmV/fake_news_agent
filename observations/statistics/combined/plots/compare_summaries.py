import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================
# Configuration (edit here)
# =============================

# Paths to an arbitrary number of CSV summary files.
# Each must have columns: metric, label, mean, median, std
# FILES: List[str] = [
#     os.path.join(os.path.dirname(__file__), "set-30b-initial-prompt_summary.csv"),
#     os.path.join(os.path.dirname(__file__), "set-30b-improved-prompt_summary.csv"),
# ]

METRICS_MAPPING = {
    "label_correlation": "Label Correlation",
    "short_assessment_truthfulness": "Short Assessment Truthfulness",
    "layman_xai_truthfulness": "Layman XAI Truthfulness",
    "technical_assessment_clarity": "Technical Assessment Clarity",
    "technical_xai_clarity": "Technical XAI Clarity",
    "xai_description_truthfulness": "XAI Description Truthfulness"
}


FILES: List[str] = [
    os.path.join(os.path.dirname(__file__), "set-30b-initial-prompt_summary.csv"),
    os.path.join(os.path.dirname(__file__), "set-30b-improved-prompt_summary.csv"),
]



# Labels for settings (x-axis). Must align with FILES. If empty or length differs,
# labels are inferred from basenames without extension.
SETTING_LABELS: List[str] = [
    "Base Prompt",
    "Improved Prompt",
]

# Output path for the figure
OUTPUT_FIGURE: str = os.path.join(
    os.path.dirname(__file__),
    "compare_summaries.png",
)

# Optional explicit order of metrics to plot. If None, inferred from data (sorted).
METRICS_ORDER: Optional[List[str]] = None

# Figure layout
# For a 2 x 3 layout, use 3 columns. Additional rows will be added if needed.
N_COLS: int = 3
FIGSIZE_PER_ROW: Tuple[float, float] = (10.0, 4.8)  # (width, height) per row

# Bar and line styling (align with example style)
# Mean bars: warm/beige; Std bars: light blue
MEAN_BAR_COLOR: str = "#f0d9a8"
STD_BAR_COLOR: str = "#bcd7f5"
MEAN_LINE_COLOR: str = "#d79e00"
STD_LINE_COLOR: str = "#2d6ea3"
BAR_WIDTH: float = 0.36
LINE_WIDTH: float = 2.6
ANNOTATION_FONTSIZE: int = 18

# Typography
TITLE_FONTSIZE: int = 20
TICK_FONTSIZE: int = 18
LEGEND_FONTSIZE: int = 20


def read_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Safety: ensure numeric types
    for col in ["mean", "std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Aggregate over labels per metric: simple average of the provided values
    grouped = (
        df.groupby("metric", as_index=False)
        .agg({"mean": "mean", "std": "mean"})
        .rename(columns={"mean": "mean", "std": "std"})
    )
    return grouped


def determine_metrics(aggregated: List[pd.DataFrame]) -> List[str]:
    if not aggregated:
        return []
    if METRICS_ORDER is not None:
        present: set = set()
        for df in aggregated:
            present |= set(df["metric"])  # union of available metrics
        return [m for m in METRICS_ORDER if m in present]
    # Try intersection of all first
    inter: Optional[set] = None
    union: set = set()
    for df in aggregated:
        mset = set(df["metric"])
        union |= mset
        inter = mset if inter is None else inter.intersection(mset)
    metrics = sorted(inter) if inter else []
    if not metrics:
        metrics = sorted(union)
    return metrics


def safe_percent_change(new_value: float, old_value: float) -> Optional[float]:
    if old_value is None or np.isnan(old_value) or old_value == 0:
        return None
    if new_value is None or np.isnan(new_value):
        return None
    return (new_value - old_value) / abs(old_value) * 100.0


def add_connection_annotation(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, text: str, line_color: str, text_color: str, linestyle: str = "--", above_pad: float = 0.04, fontweight: str = "bold") -> None:
    ax.plot([x0, x1], [y0, y1], linestyle=linestyle, color=line_color, linewidth=LINE_WIDTH)
    xt = (x0 + x1) / 2.0
    base = max(y0, y1)
    yt = base + (abs(base) + 1e-8) * above_pad
    ax.text(xt, yt, text, ha="center", va="bottom", fontsize=ANNOTATION_FONTSIZE, fontweight=fontweight, color=text_color)


def plot_metric_subplot(ax: plt.Axes, metric: str, aggregated: List[pd.DataFrame], setting_labels: List[str]) -> None:
    # Gather values for each setting (file)
    means: List[float] = []
    stds: List[float] = []
    for agg in aggregated:
        row = agg[agg["metric"] == metric]
        means.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
        stds.append(float(row["std"].iloc[0]) if not row.empty else np.nan)

    n = len(setting_labels)
    x = np.arange(n)

    # Bars per setting: mean and std side-by-side
    mean_centers = x - BAR_WIDTH / 2.0
    std_centers = x + BAR_WIDTH / 2.0
    ax.bar(mean_centers, means, width=BAR_WIDTH, color=MEAN_BAR_COLOR, label="Mean")
    ax.bar(std_centers, stds, width=BAR_WIDTH, color=STD_BAR_COLOR, label="Variance")

    # Lines across settings, connected at the center of corresponding bars
    ax.plot(mean_centers, means, color=MEAN_LINE_COLOR, linewidth=LINE_WIDTH)
    ax.plot(std_centers, stds, color=STD_LINE_COLOR, linestyle="--", linewidth=LINE_WIDTH)

    # Annotations for deltas between consecutive settings
    for i in range(1, n):
        # Mean change
        pct_m = safe_percent_change(means[i], means[i - 1])
        if pct_m is not None and not (np.isnan(means[i]) or np.isnan(means[i - 1])):
            text_color = "green" if pct_m >= 0 else "red"
            text = f"{pct_m:+.0f}%"
            add_connection_annotation(
                ax,
                mean_centers[i - 1],
                means[i - 1],
                mean_centers[i],
                means[i],
                text,
                line_color=MEAN_LINE_COLOR,
                text_color=text_color,
                linestyle="-",
            )
        # Std change
        pct_s = safe_percent_change(stds[i], stds[i - 1])
        if pct_s is not None and not (np.isnan(stds[i]) or np.isnan(stds[i - 1])):
            text_color = "red" if pct_s >= 0 else "green"
            text = f"{pct_s:+.0f}%"
            add_connection_annotation(
                ax,
                std_centers[i - 1],
                stds[i - 1],
                std_centers[i],
                stds[i],
                text,
                line_color=STD_LINE_COLOR,
                text_color=text_color,
                linestyle="--",
            )

    # Cosmetics
    ax.set_title(METRICS_MAPPING[metric], fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main() -> None:
    # Load and aggregate per file
    aggregated: List[pd.DataFrame] = [read_and_aggregate(p) for p in FILES]

    metrics = determine_metrics(aggregated)
    if not metrics:
        raise ValueError("No metrics found to plot. Check input files and columns.")

    # Setting labels
    if not SETTING_LABELS or len(SETTING_LABELS) != len(FILES):
        setting_labels = [os.path.splitext(os.path.basename(p))[0] for p in FILES]
    else:
        setting_labels = SETTING_LABELS

    n_metrics = len(metrics)
    n_cols = max(1, int(N_COLS))
    n_rows = max(1, (n_metrics + n_cols - 1) // n_cols)
    figsize = (FIGSIZE_PER_ROW[0] * n_cols, FIGSIZE_PER_ROW[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for idx, metric in enumerate(metrics):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]
        plot_metric_subplot(ax, metric, aggregated, setting_labels)

    # Remove any unused subplots
    for j in range(n_metrics, n_rows * n_cols):
        r = j // n_cols
        c = j % n_cols
        fig.delaxes(axes[r][c])

    # Add main title
    fig.suptitle("Improvement Iteration", fontsize=24, fontweight='bold')

    # Shared legend for Mean/Std
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:2], labels[:2], loc="lower center", ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE)
        plt.subplots_adjust(top=0.88, hspace=0.4)
    else:
        plt.subplots_adjust(bottom=0.2)

    os.makedirs(os.path.dirname(OUTPUT_FIGURE), exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()


