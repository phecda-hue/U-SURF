import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


plt.rcParams["figure.max_open_warning"] = 0

ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_ROOT = ROOT / "analysis_results"
DEFAULT_OUTPUT_DIR = ROOT / "analysis_results" /  "vascular_analysis_plots"


SUMMARY_GROUPS = {
    "vascular_density": [
        "vessel_area_density",
        "skeleton_length_density_per_px",
    ],
    "diameter": [
        "diameter_median_px",
        "diameter_iqr_px",
        "diameter_p90_px",
        "length_weighted_mean_diameter_px",
        "microvessel_length_fraction",
    ],
    "tortuosity_curvature": [
        "tortuosity_length_weighted_mean",
        "tortuosity_p90",
        "median_curvature_per_px",
        "high_curvature_length_density_per_px",
    ],
    "branching_connectivity": [
        "branch_density_per_px",
        "largest_component_length_fraction",
        "internal_endpoint_density_per_px",
    ],
    "orientation": [
        "orientation_anisotropy",
    ],
}


CYCLE_METRICS = [
    "vascular_cycle_density_per_px",
    "cycle_length_median_px",
    "cycle_length_p90_px",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect vascular analysis CSV outputs and generate grouped plots."
        )
    )
    parser.add_argument(
        "--analysis-dir",
        action="append",
        type=Path,
        default=[],
        help=(
            "One analysis output directory. May be repeated. If omitted, "
            "directories under --search-root containing vascular_metrics_summary.csv "
            "or csv/vascular_metrics_summary.csv are used."
        ),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="Root searched for analysis directories when --analysis-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plot PNGs, combined CSV tables, and PDF report.",
    )
    parser.add_argument(
        "--max-dirs",
        type=int,
        default=20,
        help="Maximum automatically discovered analysis directories.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=40,
        help="Histogram bin count for distribution plots.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def to_float(value) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def numeric_column(rows: list[dict], field: str) -> np.ndarray:
    return np.asarray([to_float(row.get(field)) for row in rows], dtype=np.float64)


def dataset_unit(dataset: dict) -> str | None:
    for source in ("metrics", "network", "graph"):
        unit = dataset.get(source, {}).get("physical_unit")
        if unit:
            return unit
    return None


def physical_metric_name(metric: str, unit: str | None) -> str | None:
    if not unit:
        return None
    explicit = {
        "roi_area_px": f"roi_area_{unit}2",
        "vessel_area_px": f"vessel_area_{unit}2",
        "vessel_area_density": f"vessel_area_density_{unit}2_per_{unit}2",
        "density_roi_area_px": f"density_roi_area_{unit}2",
        "skeleton_length_px": f"skeleton_length_{unit}",
        "skeleton_length_density_per_px": f"skeleton_length_density_{unit}_per_{unit}2",
        "diameter_iqr_px": f"diameter_iqr_{unit}",
        "diameter_p90_px": f"diameter_p90_{unit}",
        "median_curvature_per_px": f"median_curvature_per_{unit}",
        "high_curvature_length_density_per_px": (
            f"high_curvature_length_density_{unit}_per_{unit}2"
        ),
        "branch_density_per_px": f"branch_density_per_{unit}2",
        "endpoint_density_per_px": f"endpoint_density_per_{unit}2",
        "internal_endpoint_density_per_px": f"internal_endpoint_density_per_{unit}2",
        "component_density_per_px": f"component_density_per_{unit}2",
        "vascular_cycle_density_per_px": f"vascular_cycle_density_per_{unit}2",
        "networkx_dfs_total_length_px": f"networkx_dfs_total_length_{unit}",
        "sknw_total_edge_length_px": f"sknw_total_edge_length_{unit}",
        "vascular_graph_total_length_px": f"vascular_graph_total_length_{unit}",
        "total_length_px": f"total_length_{unit}",
        "total_component_edge_length_px": f"total_component_edge_length_{unit}",
    }
    if metric in explicit:
        return explicit[metric]
    if metric.endswith("_per_px"):
        return f"{metric[:-len('_per_px')]}_per_{unit}"
    if metric.endswith("_px"):
        return f"{metric[:-len('_px')]}_{unit}"
    return None


def preferred_metric(metric: str, summary: dict, unit: str | None) -> str:
    physical = physical_metric_name(metric, unit)
    if physical and summary.get(physical) not in (None, ""):
        return physical
    return metric


def physical_scale(summary: dict, unit: str | None) -> tuple[float, float, float]:
    if not unit:
        return np.nan, np.nan, np.nan
    x_scale = to_float(summary.get(f"physical_pixel_width_{unit}_per_px"))
    y_scale = to_float(summary.get(f"physical_pixel_height_{unit}_per_px"))
    if not np.isfinite(x_scale) or not np.isfinite(y_scale):
        return np.nan, np.nan, np.nan
    mean_scale = (x_scale + y_scale) * 0.5
    area_scale = x_scale * y_scale
    return mean_scale, area_scale, x_scale


def converted_metric_value(
    metric: str,
    summary: dict,
    unit: str | None,
) -> tuple[str, float, str]:
    if metric == "vessel_area_density":
        physical = physical_metric_name(metric, unit)
        if physical and summary.get(physical) not in (None, ""):
            return physical, to_float(summary.get(physical)), physical
        if summary.get(metric) not in (None, ""):
            return metric, to_float(summary.get(metric)), metric
        return metric, to_float(summary.get("vessel_area_fraction")), "vessel_area_fraction"
    source_metric = preferred_metric(metric, summary, unit)
    value = to_float(summary.get(source_metric))
    if source_metric != metric or not unit or not np.isfinite(value):
        return source_metric, value, source_metric

    physical = physical_metric_name(metric, unit)
    mean_scale, area_scale, _ = physical_scale(summary, unit)
    if not physical or not np.isfinite(mean_scale) or not np.isfinite(area_scale):
        return source_metric, value, source_metric

    area_metrics = {"roi_area_px", "vessel_area_px", "density_roi_area_px"}
    area_density_metrics = {
        "branch_density_per_px",
        "endpoint_density_per_px",
        "component_density_per_px",
        "vascular_cycle_density_per_px",
    }
    if metric in area_metrics:
        return physical, float(value * area_scale), source_metric
    if metric == "skeleton_length_density_per_px":
        return physical, float(value * mean_scale / area_scale), source_metric
    if metric in area_density_metrics:
        return physical, float(value / area_scale), source_metric
    if metric.endswith("_per_px"):
        return physical, float(value / mean_scale), source_metric
    if metric.endswith("_px"):
        return physical, float(value * mean_scale), source_metric
    return source_metric, value, source_metric


def field_unit(field: str) -> str:
    for unit in ("um", "mm", "cm"):
        if field.endswith(f"_{unit}2_per_{unit}2"):
            return f"{unit}^2/{unit}^2"
        if field.endswith(f"_per_length_{unit}"):
            return f"1/{unit}"
        if field.endswith(f"_{unit}_per_{unit}2"):
            return f"{unit}/{unit}^2"
        if field.endswith(f"_per_{unit}2"):
            return f"1/{unit}^2"
        if field.endswith(f"_per_{unit}"):
            return f"1/{unit}"
        if field.endswith(f"_{unit}2"):
            return f"{unit}^2"
        if field.endswith(f"_{unit}"):
            return unit
    if field.endswith("_per_px"):
        return "1/px"
    if field.endswith("_px"):
        return "px"
    if field.endswith("_deg"):
        return "deg"
    return ""


def metric_label(field: str) -> str:
    label = field
    unit_text = field_unit(field)
    for unit in ("um", "mm", "cm"):
        suffixes = (
            f"_{unit}2_per_{unit}2",
            f"_per_length_{unit}",
            f"_{unit}_per_{unit}2",
            f"_per_{unit}2",
            f"_per_{unit}",
            f"_{unit}2",
            f"_{unit}",
        )
        for suffix in suffixes:
            if label.endswith(suffix):
                label = label[: -len(suffix)]
                break
    for suffix in ("_per_px", "_px", "_deg"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    pretty = label.replace("_", " ")
    return f"{pretty} [{unit_text}]" if unit_text else pretty


def preferred_csv_field(dataset: dict, csv_key: str, field: str) -> str:
    unit = dataset_unit(dataset)
    physical = physical_metric_name(field, unit)
    rows = dataset.get(csv_key, [])
    if physical and rows and physical in rows[0]:
        return physical
    return field


def first_row(path: Path) -> dict:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def analysis_root_from_metrics_path(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "csv" else path.parent


def analysis_csv_dir(path: Path) -> Path:
    csv_dir = path / "csv"
    if (csv_dir / "vascular_metrics_summary.csv").is_file():
        return csv_dir
    if path.name == "csv" and (path / "vascular_metrics_summary.csv").is_file():
        return path
    return path


def discover_analysis_dirs(search_root: Path, max_dirs: int) -> list[Path]:
    dirs = [
        analysis_root_from_metrics_path(path)
        for path in search_root.rglob("vascular_metrics_summary.csv")
        if path.is_file()
    ]
    dirs = sorted(set(dirs), key=lambda path: path.stat().st_mtime, reverse=True)
    return dirs[:max_dirs]


def sample_label(path: Path) -> str:
    name = path.name
    prefixes = ("vessel_analysis_", "vessel_geometry_")
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def load_dataset(path: Path) -> dict:
    csv_dir = analysis_csv_dir(path)
    return {
        "path": path,
        "label": sample_label(path),
        "csv_dir": csv_dir,
        "metrics": first_row(csv_dir / "vascular_metrics_summary.csv"),
        "network": first_row(csv_dir / "network_summary.csv"),
        "graph": first_row(csv_dir / "graph_summary.csv"),
        "segments": read_csv_rows(csv_dir / "vessel_segments.csv"),
        "cycles": read_csv_rows(csv_dir / "vascular_cycles.csv"),
        "junction_clusters": read_csv_rows(csv_dir / "junction_cluster_curvatures.csv"),
        "components": read_csv_rows(csv_dir / "vessel_components.csv"),
    }


def merged_summary(dataset: dict) -> dict:
    merged = {}
    merged.update(dataset["metrics"])
    merged.update(dataset["network"])
    merged.update(dataset["graph"])
    return merged


def write_selected_metrics_table(datasets: list[dict], path: Path) -> None:
    fields = ["sample", "category", "metric", "source_metric", "unit", "value"]
    rows = []
    for dataset in datasets:
        summary = merged_summary(dataset)
        unit = dataset_unit(dataset)
        for category, metrics in SUMMARY_GROUPS.items():
            for metric in metrics:
                display_metric, value, source_metric = converted_metric_value(
                    metric,
                    summary,
                    unit,
                )
                rows.append(
                    {
                        "sample": dataset["label"],
                        "category": category,
                        "metric": metric_label(display_metric),
                        "source_metric": source_metric,
                        "unit": field_unit(display_metric),
                        "value": value,
                    }
                )
        for metric in CYCLE_METRICS:
            display_metric, value, source_metric = converted_metric_value(
                metric,
                summary,
                unit,
            )
            rows.append(
                {
                    "sample": dataset["label"],
                    "category": "cycle_structure",
                    "metric": metric_label(display_metric),
                    "source_metric": source_metric,
                    "unit": field_unit(display_metric),
                    "value": value,
                }
            )

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_figure(fig, figures: list[tuple[str, plt.Figure]], name: str, output_dir: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    figures.append((name, fig))


def plot_summary_group(
    datasets: list[dict],
    group_name: str,
    metrics: list[str],
    output_dir: Path,
    figures: list[tuple[str, plt.Figure]],
) -> None:
    labels = [dataset["label"] for dataset in datasets]
    summaries = [merged_summary(dataset) for dataset in datasets]
    display_metrics = []
    for metric in metrics:
        fields = [
            converted_metric_value(metric, summary, dataset_unit(dataset))[0]
            for dataset, summary in zip(datasets, summaries)
        ]
        display_metrics.append(metric_label(next((field for field in fields if field), metric)))
    values = np.asarray(
        [
            [
                converted_metric_value(metric, summary, dataset_unit(dataset))[1]
                for metric in metrics
            ]
            for dataset, summary in zip(datasets, summaries)
        ],
        dtype=np.float64,
    )

    fig_height = max(4.0, 0.38 * len(metrics) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    y = np.arange(len(metrics))
    width = 0.8 / max(1, len(datasets))
    for index, label in enumerate(labels):
        ax.barh(
            y + (index - (len(datasets) - 1) / 2) * width,
            values[index],
            height=width,
            label=label,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(display_metrics)
    ax.invert_yaxis()
    ax.set_title(group_name.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    add_figure(fig, figures, f"summary_{group_name}", output_dir)


def plot_distribution_boxplot(
    datasets: list[dict],
    csv_key: str,
    field: str,
    title: str,
    output_dir: Path,
    figures: list[tuple[str, plt.Figure]],
) -> None:
    labels = []
    arrays = []
    plotted_fields = []
    for dataset in datasets:
        source_field = preferred_csv_field(dataset, csv_key, field)
        values = numeric_column(dataset[csv_key], source_field)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        labels.append(dataset["label"])
        arrays.append(values)
        plotted_fields.append(source_field)
    if not arrays:
        return

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(arrays)), 5))
    ax.boxplot(arrays, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(metric_label(plotted_fields[0]))
    ax.grid(axis="y", alpha=0.25)
    add_figure(fig, figures, f"box_{csv_key}_{field}", output_dir)


def plot_histograms(
    datasets: list[dict],
    csv_key: str,
    field: str,
    title: str,
    output_dir: Path,
    figures: list[tuple[str, plt.Figure]],
    bins: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    plotted_field = field
    for dataset in datasets:
        source_field = preferred_csv_field(dataset, csv_key, field)
        values = numeric_column(dataset[csv_key], source_field)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=dataset["label"],
        )
        plotted = True
        plotted_field = source_field
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel(metric_label(plotted_field))
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    add_figure(fig, figures, f"hist_{csv_key}_{field}", output_dir)


def plot_cycle_scatter(
    datasets: list[dict],
    output_dir: Path,
    figures: list[tuple[str, plt.Figure]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    plotted_field = "length_px"
    for dataset in datasets:
        rows = dataset["cycles"]
        length_field = preferred_csv_field(dataset, "cycles", "length_px")
        length = numeric_column(rows, length_field)
        node_count = numeric_column(rows, "node_count")
        valid = np.isfinite(length) & np.isfinite(node_count)
        if not valid.any():
            continue
        ax.scatter(length[valid], node_count[valid], s=10, alpha=0.55, label=dataset["label"])
        plotted = True
        plotted_field = length_field
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Cycle Size Distribution")
    ax.set_xlabel(metric_label(plotted_field))
    ax.set_ylabel("Cycle node count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    add_figure(fig, figures, "scatter_cycle_length_node_count", output_dir)


def plot_all(datasets: list[dict], output_dir: Path, bins: int) -> list[tuple[str, plt.Figure]]:
    figures: list[tuple[str, plt.Figure]] = []
    for group_name, metrics in SUMMARY_GROUPS.items():
        plot_summary_group(datasets, group_name, metrics, output_dir, figures)

    plot_summary_group(
        datasets,
        "cycle_structure",
        CYCLE_METRICS,
        output_dir,
        figures,
    )

    distributions = [
        ("segments", "median_normal_diameter_px", "Segment Diameter Distribution"),
        ("segments", "tortuosity", "Segment Tortuosity Distribution"),
        ("segments", "median_curvature_per_px", "Segment Curvature Distribution"),
        ("cycles", "length_px", "Cycle Length Distribution"),
    ]
    for csv_key, field, title in distributions:
        plot_distribution_boxplot(datasets, csv_key, field, title, output_dir, figures)
        plot_histograms(datasets, csv_key, field, title, output_dir, figures, bins)

    return figures


def write_pdf_report(figures: list[tuple[str, plt.Figure]], path: Path) -> None:
    with PdfPages(path) as pdf:
        for _, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir = output_dir / "csv"
    image_output_dir = output_dir / "images"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    if args.analysis_dir:
        analysis_dirs = [
            path.parent if (path := resolve_path(raw_path)).name == "csv" else path
            for raw_path in args.analysis_dir
        ]
    else:
        analysis_dirs = discover_analysis_dirs(
            resolve_path(args.search_root),
            args.max_dirs,
        )

    analysis_dirs = [
        path
        for path in analysis_dirs
        if (analysis_csv_dir(path) / "vascular_metrics_summary.csv").is_file()
    ]
    if not analysis_dirs:
        raise FileNotFoundError(
            "No analysis directories containing vascular_metrics_summary.csv were found."
        )

    datasets = [load_dataset(path) for path in analysis_dirs]
    selected_metrics_path = csv_output_dir / "selected_vascular_metrics.csv"
    report_path = image_output_dir / "vascular_analysis_report.pdf"
    write_selected_metrics_table(datasets, selected_metrics_path)
    figures = plot_all(datasets, image_output_dir, bins=args.hist_bins)
    write_pdf_report(figures, report_path)

    for _, fig in figures:
        plt.close(fig)

    print(f"Analysis directories: {len(datasets)}")
    for dataset in datasets:
        print(f"- {dataset['label']}: {dataset['path']} (csv: {dataset['csv_dir']})")
    print(f"Selected metrics CSV: {selected_metrics_path}")
    print(f"PDF report: {report_path}")
    print(f"PNG plots: {image_output_dir}")


if __name__ == "__main__":
    main()
