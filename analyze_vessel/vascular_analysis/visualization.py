from .common import Path, csv, np, plt

def save_overlay(
    probability: np.ndarray,
    skeleton: np.ndarray,
    path: Path,
) -> None:
    base = np.dstack([probability, probability, probability])
    overlay = base.copy()
    overlay[skeleton] = [1.0, 0.0, 0.0]
    blended = 0.55 * base + 0.45 * overlay
    blended[skeleton] = [1.0, 0.0, 0.0]
    plt.imsave(path, np.clip(blended, 0, 1))

def save_pruning_overlay(
    probability: np.ndarray,
    kept_skeleton: np.ndarray,
    removed_mask: np.ndarray,
    path: Path,
) -> None:
    image = np.dstack([probability, probability, probability])
    image[kept_skeleton] = [1.0, 0.0, 0.0]
    image[removed_mask] = [0.0, 1.0, 1.0]
    plt.imsave(path, np.clip(image, 0, 1))

def save_density_roi_overlay(
    probability: np.ndarray,
    vessel_mask: np.ndarray,
    roi_mask: np.ndarray,
    path: Path,
) -> None:
    base = np.dstack([probability, probability, probability])
    image = base.copy()
    removed_roi = ~roi_mask
    kept_vessels = vessel_mask & roi_mask
    removed_vessels = vessel_mask & removed_roi

    image[removed_roi] = 0.35 * image[removed_roi] + 0.65 * np.array([1.0, 0.0, 0.75])
    image[kept_vessels] = [1.0, 0.0, 0.0]
    image[removed_vessels] = [0.0, 1.0, 1.0]
    plt.imsave(path, np.clip(image, 0, 1))

def save_scalar_map(
    probability: np.ndarray,
    rows: list[dict],
    value_key: str,
    colorbar_label: str,
    path: Path,
) -> None:
    height, width = probability.shape
    fig_width = 12
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height),
        dpi=180,
    )
    ax.imshow(probability, cmap="gray")

    values = np.asarray(
        [row[value_key] for row in rows],
        dtype=np.float64,
    )
    valid = np.isfinite(values)

    if valid.any():
        xs = np.asarray([row["x"] for row in rows])[valid]
        ys = np.asarray([row["y"] for row in rows])[valid]
        values = values[valid]
        vmax = float(np.percentile(values, 95))
        if vmax <= 0:
            vmax = float(values.max()) if values.size else 1.0
        scatter = ax.scatter(
            xs,
            ys,
            c=values,
            s=4.0,
            cmap="turbo",
            vmin=0,
            vmax=vmax,
            alpha=0.9,
            linewidths=0,
        )
        colorbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label(colorbar_label)

    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def save_combined_curvature_overlay(
    probability: np.ndarray,
    vessel_point_rows: list[dict],
    junction_branch_point_rows: list[dict],
    path: Path,
    value_key: str = "curvature_per_px",
    colorbar_label: str = "Curvature (1/pixel)",
) -> None:
    height, width = probability.shape
    fig_width = 12
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.imshow(probability, cmap="gray")

    vessel_values = np.asarray(
        [row.get(value_key, np.nan) for row in vessel_point_rows],
        dtype=np.float64,
    )
    branch_values = np.asarray(
        [row.get(value_key, np.nan) for row in junction_branch_point_rows],
        dtype=np.float64,
    )
    all_values = np.concatenate([vessel_values, branch_values])
    valid_all = all_values[np.isfinite(all_values)]

    if valid_all.size:
        vmax = float(np.percentile(valid_all, 95))
        if vmax <= 0:
            vmax = float(valid_all.max()) if valid_all.size else 1.0

        vessel_valid = np.isfinite(vessel_values)
        if vessel_valid.any():
            ax.scatter(
                np.asarray([row["x"] for row in vessel_point_rows])[vessel_valid],
                np.asarray([row["y"] for row in vessel_point_rows])[vessel_valid],
                c=vessel_values[vessel_valid],
                s=3.0,
                cmap="turbo",
                vmin=0,
                vmax=vmax,
                alpha=0.75,
                linewidths=0,
            )

        branch_valid = np.isfinite(branch_values)
        if branch_valid.any():
            scatter = ax.scatter(
                np.asarray([row["x"] for row in junction_branch_point_rows])[
                    branch_valid
                ],
                np.asarray([row["y"] for row in junction_branch_point_rows])[
                    branch_valid
                ],
                c=branch_values[branch_valid],
                s=5.0,
                cmap="turbo",
                vmin=0,
                vmax=vmax,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.12,
            )
        else:
            scatter = ax.scatter([], [], c=[], cmap="turbo", vmin=0, vmax=vmax)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(colorbar_label)

    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def parse_id_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(";") if part.strip()]

def save_thick_vessel_overlay(
    probability: np.ndarray,
    segment_geometries: dict[int, np.ndarray],
    thick_vessel_rows: list[dict],
    path: Path,
    diameter_key: str = "median_normal_diameter_px",
    length_key: str = "longest_path_px",
    unit_label: str = "px",
) -> None:
    height, width = probability.shape
    fig_width = 14
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.imshow(probability, cmap="gray")

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for index, row in enumerate(thick_vessel_rows):
        color = colors[index % len(colors)]
        segment_ids = parse_id_list(row.get("segment_ids", ""))
        label_points = []

        for segment_id in segment_ids:
            segment = segment_geometries.get(segment_id)
            if segment is None or len(segment) < 2:
                continue
            line_width = float(
                np.clip(row["median_normal_diameter_px"] * 0.35, 2.5, 7.0)
            )
            ax.plot(
                segment[:, 1],
                segment[:, 0],
                color="black",
                linewidth=line_width + 1.8,
                alpha=0.9,
                solid_capstyle="round",
            )
            ax.plot(
                segment[:, 1],
                segment[:, 0],
                color=color,
                linewidth=line_width,
                solid_capstyle="round",
            )
            label_points.append(segment)

        if not label_points:
            continue

        points = np.vstack(label_points)
        label_y, label_x = np.median(points, axis=0)
        diameter = float(row.get(diameter_key, row["median_normal_diameter_px"]))
        length = float(row.get(length_key, row["longest_path_px"]))
        label = (
            f"T{int(row['thick_vessel_id'])} "
            f"d={diameter:.2f}{unit_label} "
            f"L={length:.2f}{unit_label}"
        )
        ax.text(
            label_x,
            label_y,
            label,
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": (0, 0, 0, 0.65),
                "edgecolor": color,
                "linewidth": 0.6,
            },
        )

    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def safe_stat(values: np.ndarray, function, default=np.nan) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(function(finite))
