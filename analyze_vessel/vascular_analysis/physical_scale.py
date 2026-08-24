from .common import argparse, np

def path_length(points: np.ndarray | list[tuple[int, int]]) -> float:
    array = np.asarray(points, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())

def path_length_scaled(
    points: np.ndarray | list[tuple[int, int]],
    scale_y: float,
    scale_x: float,
) -> float:
    array = np.asarray(points, dtype=np.float64)
    if len(array) < 2:
        return 0.0
    diff = np.diff(array, axis=0)
    dy = diff[:, 0] * scale_y
    dx = diff[:, 1] * scale_x
    return float(np.sqrt(dy * dy + dx * dx).sum())

def local_orientation_metrics(
    segments: list[np.ndarray],
    scale_y: float = 1.0,
    scale_x: float = 1.0,
) -> dict:
    vector_sum = 0.0 + 0.0j
    total_length = 0.0
    horizontal_length = 0.0
    vertical_length = 0.0
    diagonal_length = 0.0

    for segment in segments:
        points = np.asarray(segment, dtype=np.float64)
        if len(points) < 2:
            continue

        delta = np.diff(points, axis=0)
        dy = delta[:, 0] * scale_y
        dx = delta[:, 1] * scale_x
        local_length = np.hypot(dy, dx)
        valid = np.isfinite(local_length) & (local_length > 1e-8)
        if not np.any(valid):
            continue

        dy = dy[valid]
        dx = dx[valid]
        local_length = local_length[valid]
        local_angle = np.arctan2(dy, dx)

        vector_sum += np.sum(local_length * np.exp(2j * local_angle))
        total_length += float(np.sum(local_length))

        folded_deg = np.abs(np.degrees(local_angle)) % 180.0
        folded_deg = np.minimum(folded_deg, 180.0 - folded_deg)
        horizontal_length += float(np.sum(local_length[folded_deg <= 22.5]))
        vertical_length += float(np.sum(local_length[folded_deg >= 67.5]))
        diagonal_length += float(
            np.sum(local_length[(folded_deg > 22.5) & (folded_deg < 67.5)])
        )

    if total_length <= 0:
        return {
            "orientation_anisotropy": np.nan,
            "principal_orientation_deg": np.nan,
            "horizontal_length_fraction": np.nan,
            "vertical_length_fraction": np.nan,
            "diagonal_length_fraction": np.nan,
        }

    mean_vector = vector_sum / total_length
    return {
        "orientation_anisotropy": float(np.abs(mean_vector)),
        "principal_orientation_deg": float(
            (np.degrees(np.angle(mean_vector)) / 2.0) % 180.0
        ),
        "horizontal_length_fraction": horizontal_length / total_length,
        "vertical_length_fraction": vertical_length / total_length,
        "diagonal_length_fraction": diagonal_length / total_length,
    }

def configure_physical_scale(args: argparse.Namespace, shape: tuple[int, int]) -> dict:
    height_px, width_px = shape
    if args.physical_width is not None and args.physical_height is not None:
        scale_x = float(args.physical_width) / float(width_px)
        scale_y = float(args.physical_height) / float(height_px)
        unit = args.physical_unit
        source = "physical_image_size"
    elif args.pixel_size_um is not None:
        scale_x = float(args.pixel_size_um)
        scale_y = float(args.pixel_size_um)
        unit = "um"
        source = "pixel_size_um"
    else:
        scale_x = np.nan
        scale_y = np.nan
        unit = ""
        source = ""

    enabled = bool(np.isfinite(scale_x) and np.isfinite(scale_y))
    mean_scale = float((scale_x + scale_y) * 0.5) if enabled else np.nan
    scale = {
        "enabled": enabled,
        "unit": unit,
        "source": source,
        "x_per_px": float(scale_x),
        "y_per_px": float(scale_y),
        "mean_per_px": mean_scale,
    }
    setattr(args, "_physical_scale", scale)
    return scale

def physical_scale_from_args(args: argparse.Namespace) -> dict:
    return getattr(
        args,
        "_physical_scale",
        {
            "enabled": False,
            "unit": "",
            "source": "",
            "x_per_px": np.nan,
            "y_per_px": np.nan,
            "mean_per_px": np.nan,
        },
    )

def apply_physical_parameter_overrides(args: argparse.Namespace, scale: dict) -> None:
    overrides = {
        "prune_branch_length_physical": "prune_branch_length",
        "parent_sample_length_physical": "parent_sample_length",
        "prune_junction_link_length_physical": "prune_junction_link_length",
        "resample_spacing_physical": "resample_spacing",
        "min_segment_length_physical": "min_segment_length",
        "curvature_end_margin_physical": "curvature_end_margin",
        "junction_branch_min_length_physical": "junction_branch_min_length",
    }
    requested = {
        physical_name: getattr(args, physical_name, None)
        for physical_name in overrides
        if getattr(args, physical_name, None) is not None
    }
    if not requested:
        return
    if not scale.get("enabled"):
        names = ", ".join(f"--{name.replace('_', '-')}" for name in requested)
        raise ValueError(f"{names} require physical scaling.")

    mean_scale = float(scale["mean_per_px"])
    if mean_scale <= 0:
        raise ValueError("Physical scale must be positive.")
    for physical_name, pixel_name in overrides.items():
        value = getattr(args, physical_name, None)
        if value is not None:
            setattr(args, pixel_name, float(value) / mean_scale)

def cycle_length_filter_values(args: argparse.Namespace, scale: dict) -> dict:
    min_value = float(args.min_cycle_length)
    max_value = (
        float(args.max_cycle_length)
        if args.max_cycle_length is not None
        else np.nan
    )
    if args.cycle_length_unit == "physical":
        if not scale.get("enabled"):
            raise ValueError(
                "--cycle-length-unit physical requires --physical-width and "
                "--physical-height, or --pixel-size-um."
            )
        mean_scale = float(scale["mean_per_px"])
        return {
            "unit_mode": "physical",
            "filter_unit": scale["unit"],
            "min_px": min_value / mean_scale,
            "max_px": max_value / mean_scale if np.isfinite(max_value) else np.nan,
            "min_physical": min_value,
            "max_physical": max_value,
        }

    mean_scale = float(scale["mean_per_px"]) if scale.get("enabled") else np.nan
    return {
        "unit_mode": "px",
        "filter_unit": "px",
        "min_px": min_value,
        "max_px": max_value,
        "min_physical": min_value * mean_scale if np.isfinite(mean_scale) else np.nan,
        "max_physical": (
            max_value * mean_scale
            if np.isfinite(max_value) and np.isfinite(mean_scale)
            else np.nan
        ),
    }

def add_physical_length_fields(
    row: dict,
    length_fields: list[str],
    args: argparse.Namespace,
) -> None:
    scale = physical_scale_from_args(args)
    if not scale["enabled"]:
        return
    unit = scale["unit"]
    mean_scale = scale["mean_per_px"]
    for field in length_fields:
        if field in row and np.isfinite(float(row[field])):
            output_field = field.removesuffix("_px") + f"_{unit}"
            row[output_field] = float(row[field]) * mean_scale
