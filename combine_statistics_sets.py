import argparse
import importlib
import os
import sys
from typing import Dict, List, Any, Set


def ensure_project_on_path() -> None:
    """Ensure the project root (this file's directory) is on sys.path for imports."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_metric_from_set(set_number: int, metric: str) -> Dict[str, List[float]]:
    """Dynamically import a metric module from observations.statistics.set_{N} and return its variable.

    Example: set_number=5, metric='label_correlation' → imports
    observations.statistics.set_5.label_correlation and returns the value of
    variable `label_correlation`.
    """
    module_path = f"observations.statistics.set_{set_number}.{metric}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import module '{module_path}'. Ensure file 'observations/statistics/set_{set_number}/{metric}.py' exists."
        ) from exc

    if not hasattr(module, metric):
        raise AttributeError(
            f"Module '{module_path}' does not define variable '{metric}'."
        )

    value = getattr(module, metric)
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected '{module_path}.{metric}' to be a dict, got {type(value).__name__}."
        )
    return value


def combine_metric(sets: List[int], metric: str) -> Dict[str, List[List[float]]]:
    """Combine the given metric across sets into {label: [[...], [...], ...]} format."""
    combined: Dict[str, List[List[float]]] = {}
    for set_num in sets:
        data = load_metric_from_set(set_num, metric)
        for label, values in data.items():
            if not isinstance(values, list):
                raise TypeError(
                    f"Expected list for label '{label}' in set {set_num}, got {type(values).__name__}."
                )
            combined.setdefault(label, []).append(values)
    return combined


def render_python_assignment(var_name: str, data: Dict[str, Any]) -> str:
    """Render a Python assignment for the combined dict with readable formatting."""
    lines: List[str] = []
    lines.append(f"{var_name} = {{")
    # Sort keys for deterministic output
    for idx, label in enumerate(sorted(data.keys())):
        values = data[label]
        lines.append(f"    \"{label}\": [")
        for row in values:
            # format row as [a, b, c] with 2 decimals if numeric, preserve zeros
            formatted_row = ", ".join(
                f"{x:.2f}" if isinstance(x, (int, float)) else repr(x) for x in row
            )
            lines.append(f"        [{formatted_row}],")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-set statistics into a single file. Example:\n\n"
            "  python combine_statistics_sets.py --metric label_correlation --sets 5 8 11 \n\n"
            "This will import from 'observations/statistics/set_{N}/label_correlation.py' and write a combined assignment.\n\n"
            "Automated mode to combine ALL metrics common to the sets and save under a group id:\n\n"
            "  python combine_statistics_sets.py --auto-all --id 4b --sets 5 8 11\n"
        )
    )
    parser.add_argument(
        "--metric",
        required=False,
        help="Metric/module name to combine (e.g., 'label_correlation', 'focus_quality').",
    )
    parser.add_argument(
        "--sets",
        required=True,
        nargs="+",
        type=int,
        help="List of set numbers to combine, e.g., 5 8 11",
    )
    parser.add_argument(
        "--auto-all",
        action="store_true",
        help="If set, combine all metrics common to the specified sets (requires --id).",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Identifier for group output directory under combined/set-<id>/ (used with --auto-all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output file path. Defaults to 'observations/statistics/combined/<metric>_combined.py'. "
            "Use '-' to print to stdout."
        ),
    )
    args = parser.parse_args()
    if not args.auto_all and not args.metric:
        parser.error("--metric is required unless --auto-all is specified.")
    return args


def list_metric_files_for_set(set_number: int) -> Set[str]:
    """List metric module base names available in a given set directory."""
    set_dir = os.path.join("observations", "statistics", f"set_{set_number}")
    if not os.path.isdir(set_dir):
        raise FileNotFoundError(f"Set directory not found: {set_dir}")
    metrics: Set[str] = set()
    for entry in os.listdir(set_dir):
        if not entry.endswith(".py"):
            continue
        if entry.startswith("__"):
            continue
        metrics.add(entry[:-3])
    return metrics


def find_common_metrics(sets: List[int]) -> List[str]:
    """Return sorted list of metric module names common to all specified sets."""
    common: Set[str] = None  # type: ignore
    for set_num in sets:
        metrics = list_metric_files_for_set(set_num)
        if common is None:
            common = set(metrics)
        else:
            common &= metrics
    return sorted(common or [])


def write_combined_metric_file(metric: str, combined: Dict[str, List[List[float]]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{metric}.py")
    rendered = render_python_assignment(metric, combined)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered + "\n")
    return output_path


def aggregate_all_metrics(sets: List[int], group_id: str) -> List[str]:
    """Aggregate all metrics common to the given sets and write under combined/set-<group_id>/.

    Returns a list of written file paths.
    """
    metrics = find_common_metrics(sets)
    if not metrics:
        raise RuntimeError("No common metrics found across specified sets.")
    out_dir = os.path.join("observations", "statistics", "combined", f"set-{group_id}")
    written: List[str] = []
    for metric in metrics:
        combined = combine_metric(sets, metric)
        path = write_combined_metric_file(metric, combined, out_dir)
        written.append(path)
    return written


def main() -> None:
    ensure_project_on_path()
    args = parse_args()

    # Simple feature detection for automated mode via optional args set by user
    auto_all = hasattr(args, "auto_all") and getattr(args, "auto_all")  # Backward-safe
    # If the parser hasn't defined auto_all/id (older version), default to manual path
    if auto_all:
        group_id = getattr(args, "id")
        if not group_id:
            raise SystemExit("--auto-all requires --id to specify the output group folder, e.g., --id 4b")
        sets: List[int] = args.sets
        written = aggregate_all_metrics(sets, group_id)
        print("Wrote:")
        for path in written:
            print(f"  - {path}")
        return

    metric: str = args.metric
    sets: List[int] = args.sets
    output_path: str = args.output

    combined = combine_metric(sets, metric)
    rendered = render_python_assignment(metric, combined)

    if output_path is None:
        output_dir = os.path.join("observations", "statistics", "combined")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{metric}_combined.py")

    if output_path == "-":
        print(rendered)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered + "\n")

    print(f"Wrote combined metric to: {output_path}")


if __name__ == "__main__":
    main()


