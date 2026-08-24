from .common import (
    cv2,
    gaussian_filter,
    np,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
    threshold_otsu,
)

def smooth_probability(
    probability: np.ndarray,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    if sigma_y == 0 and sigma_x == 0:
        return probability.copy()
    return gaussian_filter(
        probability,
        sigma=(sigma_y, sigma_x),
        mode="nearest",
    ).astype(np.float32)

def automatic_threshold(probability: np.ndarray) -> float:
    nonzero = probability[np.isfinite(probability) & (probability > 0)]
    if nonzero.size == 0:
        return 0.5
    if np.allclose(nonzero, nonzero[0]):
        return float(nonzero[0] * 0.5)
    return float(threshold_otsu(nonzero))

def make_binary_mask(
    probability: np.ndarray,
    threshold: float | None,
    min_object_size: int,
    min_hole_size: int,
    closing_radius: int = 0,
) -> tuple[np.ndarray, float]:
    used_threshold = automatic_threshold(probability) if threshold is None else float(threshold)
    if not 0 <= used_threshold <= 1:
        raise ValueError("Threshold values must be in [0, 1].")

    binary = probability >= used_threshold
    if closing_radius > 0:
        kernel_size = int(closing_radius) * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        binary = cv2.morphologyEx(
            binary.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        ).astype(bool)
    if min_object_size > 0:
        binary = remove_small_objects(
            binary,
            max_size=max(0, int(min_object_size) - 1),
            connectivity=2,
        )
    if min_hole_size > 0:
        binary = remove_small_holes(
            binary,
            max_size=max(0, int(min_hole_size) - 1),
            connectivity=2,
        )
    return binary.astype(bool), used_threshold

def fill_small_skeleton_loops(
    skeleton: np.ndarray,
    max_hole_size: int,
) -> tuple[np.ndarray, int]:
    if max_hole_size <= 0:
        return skeleton.astype(bool), 0

    filled = remove_small_holes(
        skeleton.astype(bool),
        max_size=max(0, int(max_hole_size) - 1),
        connectivity=2,
    )
    filled_pixels = int(np.sum(filled & ~skeleton))
    if filled_pixels == 0:
        return skeleton.astype(bool), 0
    return skeletonize(filled).astype(bool), filled_pixels

def boolean_runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    runs: list[tuple[bool, int, int]] = []
    if mask.size == 0:
        return runs
    start = 0
    current = bool(mask[0])
    for index in range(1, len(mask)):
        value = bool(mask[index])
        if value == current:
            continue
        runs.append((current, start, index))
        start = index
        current = value
    runs.append((current, start, len(mask)))
    return runs

def bridge_low_density_gaps(low_density: np.ndarray, max_gap: int) -> np.ndarray:
    bridged = low_density.astype(bool).copy()
    if max_gap <= 0:
        return bridged

    runs = boolean_runs(bridged)
    for run_index, (value, start, stop) in enumerate(runs):
        if value:
            continue
        if stop - start > max_gap:
            continue
        has_low_before = run_index > 0 and runs[run_index - 1][0]
        has_low_after = run_index + 1 < len(runs) and runs[run_index + 1][0]
        if has_low_before and has_low_after:
            bridged[start:stop] = True
    return bridged

def density_roi_dimensions(roi_mask: np.ndarray) -> dict:
    kept_rows = np.any(roi_mask, axis=1)
    kept_cols = np.any(roi_mask, axis=0)
    width_px = int(np.count_nonzero(kept_cols))
    height_px = int(np.count_nonzero(kept_rows))
    return {
        "density_roi_width_px": width_px,
        "density_roi_height_px": height_px,
        "density_roi_area_from_dimensions_px": int(width_px * height_px),
    }

def make_density_roi(
    vessel_mask: np.ndarray,
    axis: str,
    threshold: float,
    min_run: int,
    bridge_gap: int,
) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    if axis == "both":
        y_roi, y_profile_rows, y_removed_run_rows, y_summary = make_density_roi(
            vessel_mask,
            axis="y",
            threshold=threshold,
            min_run=min_run,
            bridge_gap=bridge_gap,
        )
        x_roi, x_profile_rows, x_removed_run_rows, x_summary = make_density_roi(
            vessel_mask,
            axis="x",
            threshold=threshold,
            min_run=min_run,
            bridge_gap=bridge_gap,
        )
        roi_mask = y_roi & x_roi
        roi_dimensions = density_roi_dimensions(roi_mask)
        summary = {
            "density_roi_enabled": True,
            "density_roi_axis": axis,
            "density_roi_threshold": float(threshold),
            "density_roi_min_run": int(min_run),
            "density_roi_bridge_gap": int(bridge_gap),
            "density_roi_removed_coordinate_count": int(
                y_summary["density_roi_removed_coordinate_count"]
                + x_summary["density_roi_removed_coordinate_count"]
            ),
            "density_roi_removed_y_coordinate_count": int(
                y_summary["density_roi_removed_coordinate_count"]
            ),
            "density_roi_removed_x_coordinate_count": int(
                x_summary["density_roi_removed_coordinate_count"]
            ),
            "density_roi_removed_area_px": int(
                np.size(roi_mask) - roi_dimensions["density_roi_area_from_dimensions_px"]
            ),
            "density_roi_area_px": int(
                roi_dimensions["density_roi_area_from_dimensions_px"]
            ),
            "density_roi_removed_area_fraction": float(
                1.0
                - (
                    roi_dimensions["density_roi_area_from_dimensions_px"]
                    / float(np.size(roi_mask))
                )
            ),
        }
        summary.update(roi_dimensions)
        return (
            roi_mask,
            y_profile_rows + x_profile_rows,
            y_removed_run_rows + x_removed_run_rows,
            summary,
        )

    if axis == "y":
        density = np.mean(vessel_mask, axis=1)
        roi_mask = np.ones_like(vessel_mask, dtype=bool)
    elif axis == "x":
        density = np.mean(vessel_mask, axis=0)
        roi_mask = np.ones_like(vessel_mask, dtype=bool)
    else:
        raise ValueError(f"Unsupported ROI density axis: {axis}")

    low_density = density <= threshold
    bridged_low_density = bridge_low_density_gaps(low_density, bridge_gap)
    remove_coordinates = np.zeros_like(bridged_low_density, dtype=bool)
    for value, start, stop in boolean_runs(bridged_low_density):
        if value and stop - start >= min_run:
            remove_coordinates[start:stop] = True

    if axis == "y":
        roi_mask[remove_coordinates, :] = False
    else:
        roi_mask[:, remove_coordinates] = False

    roi_dimensions = density_roi_dimensions(roi_mask)
    profile_rows = [
        {
            "axis": axis,
            "coordinate": int(index),
            "vessel_density": float(value),
            "low_density_initial": bool(low_density[index]),
            "low_density_after_gap_bridge": bool(bridged_low_density[index]),
            "removed_from_roi": bool(remove_coordinates[index]),
        }
        for index, value in enumerate(density)
    ]
    removed_run_rows = [
        {
            "axis": axis,
            "start_coordinate": int(start),
            "end_coordinate_exclusive": int(stop),
            "length": int(stop - start),
            "mean_vessel_density": float(np.mean(density[start:stop])),
            "max_vessel_density": float(np.max(density[start:stop])),
        }
        for value, start, stop in boolean_runs(remove_coordinates)
        if value
    ]
    summary = {
        "density_roi_enabled": True,
        "density_roi_axis": axis,
        "density_roi_threshold": float(threshold),
        "density_roi_min_run": int(min_run),
        "density_roi_bridge_gap": int(bridge_gap),
        "density_roi_removed_coordinate_count": int(np.sum(remove_coordinates)),
        "density_roi_removed_area_px": int(
            np.size(roi_mask) - roi_dimensions["density_roi_area_from_dimensions_px"]
        ),
        "density_roi_area_px": int(
            roi_dimensions["density_roi_area_from_dimensions_px"]
        ),
        "density_roi_removed_area_fraction": float(
            1.0
            - (
                roi_dimensions["density_roi_area_from_dimensions_px"]
                / float(np.size(roi_mask))
            )
        ),
    }
    summary.update(roi_dimensions)
    return roi_mask, profile_rows, removed_run_rows, summary

def full_density_roi(shape: tuple[int, int]) -> tuple[np.ndarray, list[dict], list[dict], dict]:
    roi_mask = np.ones(shape, dtype=bool)
    roi_dimensions = density_roi_dimensions(roi_mask)
    summary = {
        "density_roi_enabled": False,
        "density_roi_axis": "",
        "density_roi_threshold": np.nan,
        "density_roi_min_run": 0,
        "density_roi_bridge_gap": 0,
        "density_roi_removed_coordinate_count": 0,
        "density_roi_removed_area_px": 0,
        "density_roi_area_px": int(
            roi_dimensions["density_roi_area_from_dimensions_px"]
        ),
        "density_roi_removed_area_fraction": 0.0,
    }
    summary.update(roi_dimensions)
    return roi_mask, [], [], summary
