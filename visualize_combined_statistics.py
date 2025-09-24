import argparse
import math
import os
from importlib.machinery import SourceFileLoader
from typing import Dict, List, Tuple

import numpy as np

import matplotlib.pyplot as plt

version = "_v1-1"

def load_combined_variable(file_path: str, var_name: str) -> Dict[str, List[List[float]]]:
    """Load a python file from an arbitrary path and return the variable named var_name.

    The variable is expected to be a dict: {label: [[...], [...], ...]} where inner lists
    are numeric sequences for the given metric.
    """
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    module = SourceFileLoader(module_name, file_path).load_module()
    if not hasattr(module, var_name):
        raise AttributeError(
            f"Variable '{var_name}' not found in '{file_path}'."
        )
    value = getattr(module, var_name)
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected '{var_name}' to be a dict, got {type(value).__name__}."
        )
    return value


def load_metric_from_file(file_path: str) -> Tuple[str, Dict[str, List[List[float]]]]:
    """Load a metric variable from file assuming variable name equals filename stem."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    values = load_combined_variable(file_path, stem)
    return stem, values


def flatten(values_by_label: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    """Flatten {label: [[...], [...]]} to {label: [...]} by concatenation."""
    flattened: Dict[str, List[float]] = {}
    for label, rows in values_by_label.items():
        all_values: List[float] = []
        for row in rows:
            all_values.extend(row)
        flattened[label] = all_values
    return flattened


def plot_boxplot(values_by_label: Dict[str, List[float]], title: str, metric: str, save_path: str | None) -> None:
    labels = sorted(values_by_label.keys())
    data = [values_by_label[label] for label in labels]

    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot(data, patch_artist=True, labels=labels)

    # Simple styling
    colors = ["#69b3a2", "#4374B3", "#E6842A", "#66A61E", "#E7298A", "#A6761D", "#7570B3"]
    for i, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.6)

    plt.ylabel(metric)
    plt.xlabel("labels")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_subplots_for_directory(dir_path: str, title: str | None, save_path: str | None) -> None:
    """Create a grid of subplots, one per metric file in the directory."""
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".py") and not f.startswith("__")
    ]
    if not files:
        raise FileNotFoundError(f"No metric files found in directory: {dir_path}")

    metrics_data: List[Tuple[str, Dict[str, List[float]]]] = []
    for file_path in sorted(files):
        metric_name, nested = load_metric_from_file(file_path)
        metrics_data.append((metric_name, flatten(nested)))

    n = len(metrics_data)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4 * rows + 1))
    # Normalize axes to 2D array
    if isinstance(axes, plt.Axes):  # single subplot fallback
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]  # list of axes in one row
    elif cols == 1:
        axes = [[ax] for ax in axes]

    color_palette = ["#69b3a2", "#4374B3", "#E6842A", "#66A61E", "#E7298A", "#A6761D", "#7570B3"]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx >= n:
                ax.axis('off')
                continue
            metric_name, values_by_label = metrics_data[idx]
            labels = sorted(values_by_label.keys())
            data = [values_by_label[label] for label in labels]
            bplot = ax.boxplot(data, patch_artist=True, labels=labels)
            for i, patch in enumerate(bplot['boxes']):
                patch.set_facecolor(color_palette[i % len(color_palette)])
                patch.set_alpha(0.6)
            ax.set_title(metric_name)
            ax.set_ylabel("value")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            idx += 1

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved subplot grid to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def compute_label_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
    }


def write_summary_csv(rows: List[Dict[str, str | int | float]], output_path: str) -> None:
    import csv

    fieldnames = [
        "metric",
        "label",
        "mean",
        "median",
        "std",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved summary to: {output_path}")


def summarize_directory(dir_path: str, output_path: str | None = None) -> str:
    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".py") and not f.startswith("__")
    ]
    if not files:
        raise FileNotFoundError(f"No metric files found in directory: {dir_path}")

    rows: List[Dict[str, str | int | float]] = []
    for file_path in sorted(files):
        metric_name, nested = load_metric_from_file(file_path)
        flat = flatten(nested)
        for label, values in flat.items():
            stats = compute_label_stats(values)
            rows.append({
                "metric": metric_name,
                "label": label,
                **stats,
            })

    if output_path is None:
        plots_dir = os.path.join("observations", f"statistics{version}", "combined", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        dir_name = os.path.basename(os.path.normpath(dir_path))
        output_path = os.path.join(plots_dir, f"{dir_name}_summary.csv")

    write_summary_csv(rows, output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a combined statistics file. Example:\n\n"
            f"  python visualize_combined_statistics.py --combined observations/statistics{version}/combined/label_correlation_combined.py --metric label_correlation\n\n"
            "Produces a boxplot of metric values (y) grouped by labels (x).\n\n"
            "Directory mode to plot all metrics in a combined set folder as subplots and also write a CSV summary:\n\n"
            f"  python visualize_combined_statistics.py --combined-dir observations/statistics{version}/combined/set-4b\n"
        )
    )
    parser.add_argument(
        "--combined",
        required=False,
        help=f"Path to the combined python file (e.g., observations/statistics{version}/combined/label_correlation_combined.py)",
    )
    parser.add_argument(
        "--combined-dir",
        required=False,
        help=f"Path to a combined set directory (e.g., observations/statistics{version}/combined/set-4b)",
    )
    parser.add_argument(
        "--metric",
        required=False,
        help="Variable name inside the combined file (e.g., label_correlation)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            f"Optional path to save the plot (e.g., observations/statistics{version}/combined/plots/label_correlation.png). "
            "If omitted, the plot will be shown."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title for the plot. Defaults to '<metric> by label'",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional path for the CSV summary in directory mode. Defaults to '.../plots/<dir>_summary.csv'"
        ),
    )
    args = parser.parse_args()
    if not args.combined and not args.combined_dir:
        parser.error("Either --combined or --combined-dir must be provided.")
    if args.combined and not args.metric:
        parser.error("--metric is required when using --combined (single-file mode).")
    return args


def main() -> None:
    args = parse_args()

    if args.combined_dir:
        dir_path = args.combined_dir
        # Default output path for directory mode
        output_path = args.output
        if output_path is None:
            plots_dir = os.path.join("observations", f"statistics{version}", "combined", "plots")
            os.makedirs(plots_dir, exist_ok=True)
            dir_name = os.path.basename(os.path.normpath(dir_path))
            output_path = os.path.join(plots_dir, f"{dir_name}.png")
        plot_subplots_for_directory(dir_path, args.title, output_path)
        # Always produce a summary CSV for directory mode
        summarize_directory(dir_path, args.summary_output)
        return

    # Single-file mode
    combined_path = args.combined
    metric = args.metric
    output_path = args.output
    title = args.title or f"{metric} by label"

    values_nested = load_combined_variable(combined_path, metric)
    values_flat = flatten(values_nested)

    # Default output path if not provided
    if output_path is None:
        base_dir = os.path.join("observations", f"statistics{version}", "combined", "plots")
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(combined_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_{metric}.png")

    plot_boxplot(values_flat, title, metric, output_path)


if __name__ == "__main__":
    main()


