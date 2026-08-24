from .common import (
    Path,
    argparse,
    cv2,
    np,
    nx,
    plt,
    sknw,
)

from .preprocessing import density_roi_dimensions

from .physical_scale import local_orientation_metrics, path_length, path_length_scaled, physical_scale_from_args

from .graph_analysis import weighted_percentile

from .visualization import safe_stat

def vascular_metrics_summary(
    measurement_binary: np.ndarray,
    roi_mask: np.ndarray,
    endpoint_count: int,
    internal_endpoint_count: int,
    component_count: int,
    component_lengths: dict[int, float],
    prepared_segments: list[np.ndarray],
    segment_rows: list[dict],
    curvature_values: np.ndarray,
    curvature_values_physical: np.ndarray | None,
    graph_summary: dict,
    junction_cluster_rows: list[dict],
    args: argparse.Namespace,
) -> dict:
    roi_dimensions = density_roi_dimensions(roi_mask)
    roi_width_px = float(roi_dimensions["density_roi_width_px"])
    roi_height_px = float(roi_dimensions["density_roi_height_px"])
    roi_area_px = float(roi_dimensions["density_roi_area_from_dimensions_px"])
    vessel_area_px = float(np.count_nonzero(measurement_binary))
    scale = physical_scale_from_args(args)
    physical_enabled = bool(scale["enabled"])
    unit = scale["unit"] if physical_enabled else ""
    scale_x = float(scale["x_per_px"]) if physical_enabled else np.nan
    scale_y = float(scale["y_per_px"]) if physical_enabled else np.nan
    mean_scale = float(scale["mean_per_px"]) if physical_enabled else np.nan
    roi_area_unit2 = roi_area_px * scale_x * scale_y if physical_enabled else np.nan
    total_graph_length_px = float(
        graph_summary.get("vascular_graph_total_length_px")
        or graph_summary.get("sknw_total_edge_length_px")
        or graph_summary.get("networkx_dfs_total_length_px")
        or np.nan
    )
    if not np.isfinite(total_graph_length_px):
        total_graph_length_px = float(graph_summary.get("networkx_dfs_total_length_px", 0.0))

    segment_lengths = np.asarray(
        [row["length_px"] for row in segment_rows],
        dtype=np.float64,
    )
    if physical_enabled:
        segment_lengths_physical = np.asarray(
            [row.get(f"length_{unit}", row["length_px"] * mean_scale) for row in segment_rows],
            dtype=np.float64,
        )
    else:
        segment_lengths_physical = np.asarray([], dtype=np.float64)
    segment_diameters = np.asarray(
        [row["median_normal_diameter_px"] for row in segment_rows],
        dtype=np.float64,
    )
    if physical_enabled:
        segment_diameters_physical = np.asarray(
            [
                row.get(
                    f"median_normal_diameter_{unit}",
                    row["median_normal_diameter_px"] * mean_scale,
                )
                for row in segment_rows
            ],
            dtype=np.float64,
        )
    else:
        segment_diameters_physical = np.asarray([], dtype=np.float64)
    segment_tortuosity = np.asarray(
        [row["tortuosity"] for row in segment_rows],
        dtype=np.float64,
    )
    finite_length = np.isfinite(segment_lengths) & (segment_lengths > 0)
    total_segment_length = float(np.sum(segment_lengths[finite_length]))
    finite_length_physical = (
        np.isfinite(segment_lengths_physical) & (segment_lengths_physical > 0)
        if physical_enabled
        else np.zeros_like(finite_length)
    )
    total_segment_length_physical = (
        float(np.sum(segment_lengths_physical[finite_length_physical]))
        if physical_enabled
        else np.nan
    )

    diameter_percentiles = weighted_percentile(
        segment_diameters,
        segment_lengths,
        [10, 25, 50, 75, 90],
    )
    diameter_percentiles_physical = (
        weighted_percentile(
            segment_diameters_physical,
            segment_lengths_physical,
            [10, 25, 50, 75, 90],
        )
        if physical_enabled
        else {}
    )
    tortuosity_percentiles = weighted_percentile(
        segment_tortuosity,
        segment_lengths,
        [50, 90],
    )

    finite_diameter = np.isfinite(segment_diameters) & finite_length
    finite_diameter_physical = (
        np.isfinite(segment_diameters_physical) & finite_length_physical
        if physical_enabled
        else np.zeros_like(finite_diameter)
    )
    micro_length = float(
        np.sum(
            segment_lengths[
                finite_diameter
                & (segment_diameters <= args.microvessel_max_diameter)
            ]
        )
    )
    micro_length_physical = (
        float(
            np.sum(
                segment_lengths_physical[
                    finite_diameter
                    & (segment_diameters <= args.microvessel_max_diameter)
                ]
            )
        )
        if physical_enabled
        else np.nan
    )
    tortuous_11_length = float(
        np.sum(
            segment_lengths[
                finite_length
                & np.isfinite(segment_tortuosity)
                & (segment_tortuosity > 1.1)
            ]
        )
    )
    tortuous_12_length = float(
        np.sum(
            segment_lengths[
                finite_length
                & np.isfinite(segment_tortuosity)
                & (segment_tortuosity > 1.2)
            ]
        )
    )

    orientation_metrics = local_orientation_metrics(
        prepared_segments,
        scale_y=scale_y if physical_enabled else 1.0,
        scale_x=scale_x if physical_enabled else 1.0,
    )

    finite_curvature = curvature_values[np.isfinite(curvature_values)]
    finite_curvature_physical = (
        curvature_values_physical[np.isfinite(curvature_values_physical)]
        if curvature_values_physical is not None
        else np.asarray([], dtype=np.float32)
    )
    high_curvature_threshold = (
        float(np.percentile(finite_curvature, args.high_curvature_percentile))
        if finite_curvature.size
        else np.nan
    )
    high_curvature_threshold_physical = (
        float(np.percentile(finite_curvature_physical, args.high_curvature_percentile))
        if finite_curvature_physical.size
        else np.nan
    )
    high_curvature_point_fraction = (
        float(np.mean(finite_curvature >= high_curvature_threshold))
        if finite_curvature.size and np.isfinite(high_curvature_threshold)
        else np.nan
    )
    high_curvature_length = 0.0
    high_curvature_length_physical = 0.0
    for row in segment_rows:
        length = float(row.get("length_px", np.nan))
        curvature = float(row.get("median_curvature_per_px", np.nan))
        if (
            np.isfinite(length)
            and length > 0
            and np.isfinite(curvature)
            and np.isfinite(high_curvature_threshold)
            and curvature >= high_curvature_threshold
        ):
            high_curvature_length += length
        if physical_enabled:
            length_physical = float(row.get(f"length_{unit}", np.nan))
            curvature_physical = float(row.get(f"median_curvature_per_{unit}", np.nan))
            if (
                np.isfinite(length_physical)
                and length_physical > 0
                and np.isfinite(curvature_physical)
                and np.isfinite(high_curvature_threshold_physical)
                and curvature_physical >= high_curvature_threshold_physical
            ):
                high_curvature_length_physical += length_physical

    component_length_values = np.asarray(
        list(component_lengths.values()),
        dtype=np.float64,
    )
    total_component_length = float(np.sum(component_length_values))
    largest_component_length = (
        float(np.max(component_length_values))
        if component_length_values.size
        else 0.0
    )
    small_component_length = float(
        np.sum(component_length_values[component_length_values < args.small_component_max_length])
    )

    branch_count = len(junction_cluster_rows)
    vascular_cycle_count = float(graph_summary.get("vascular_cycle_count", np.nan))
    loop_area_total_px2 = float(
        graph_summary.get("enclosed_loop_area_total_px2", np.nan)
    )
    metrics = {
        "roi_width_px": roi_width_px,
        "roi_height_px": roi_height_px,
        "roi_area_px": roi_area_px,
        "vessel_area_px": vessel_area_px,
        "vessel_area_density": vessel_area_px / roi_area_px if roi_area_px else np.nan,
        "vessel_area_fraction": vessel_area_px / roi_area_px if roi_area_px else np.nan,
        "skeleton_length_px": total_graph_length_px,
        "skeleton_length_density_per_px": (
            total_graph_length_px / roi_area_px if roi_area_px else np.nan
        ),
        "length_weighted_mean_diameter_px": (
            float(np.average(segment_diameters[finite_diameter], weights=segment_lengths[finite_diameter]))
            if np.any(finite_diameter)
            else np.nan
        ),
        "diameter_p10_px": diameter_percentiles[10],
        "diameter_p25_px": diameter_percentiles[25],
        "diameter_median_px": diameter_percentiles[50],
        "diameter_iqr_px": diameter_percentiles[75] - diameter_percentiles[25],
        "diameter_p75_px": diameter_percentiles[75],
        "diameter_p90_px": diameter_percentiles[90],
        "microvessel_max_diameter_px": float(args.microvessel_max_diameter),
        "microvessel_length_fraction": (
            micro_length / total_segment_length if total_segment_length > 0 else np.nan
        ),
        "tortuosity_length_weighted_mean": (
            float(
                np.average(
                    segment_tortuosity[np.isfinite(segment_tortuosity) & finite_length],
                    weights=segment_lengths[np.isfinite(segment_tortuosity) & finite_length],
                )
            )
            if np.any(np.isfinite(segment_tortuosity) & finite_length)
            else np.nan
        ),
        "tortuosity_median": tortuosity_percentiles[50],
        "tortuosity_p90": tortuosity_percentiles[90],
        "tortuosity_gt_1_1_length_fraction": (
            tortuous_11_length / total_segment_length if total_segment_length > 0 else np.nan
        ),
        "tortuosity_gt_1_2_length_fraction": (
            tortuous_12_length / total_segment_length if total_segment_length > 0 else np.nan
        ),
        "high_curvature_percentile": float(args.high_curvature_percentile),
        "high_curvature_threshold_per_px": high_curvature_threshold,
        "high_curvature_point_fraction": high_curvature_point_fraction,
        "median_curvature_per_px": (
            float(np.median(finite_curvature)) if finite_curvature.size else np.nan
        ),
        "high_curvature_length_density_per_px": (
            high_curvature_length / roi_area_px if roi_area_px else np.nan
        ),
        "branch_density_per_px": branch_count / roi_area_px if roi_area_px else np.nan,
        "endpoint_density_per_px": endpoint_count / roi_area_px if roi_area_px else np.nan,
        "internal_endpoint_density_per_px": (
            internal_endpoint_count / roi_area_px if roi_area_px else np.nan
        ),
        "component_density_per_px": component_count / roi_area_px if roi_area_px else np.nan,
        "largest_component_length_fraction": (
            largest_component_length / total_component_length
            if total_component_length > 0
            else np.nan
        ),
        "small_component_max_length_px": float(args.small_component_max_length),
        "small_component_length_fraction": (
            small_component_length / total_component_length
            if total_component_length > 0
            else np.nan
        ),
        "vascular_cycle_density_per_px": (
            vascular_cycle_count / roi_area_px
            if roi_area_px and np.isfinite(vascular_cycle_count)
            else np.nan
        ),
        "vascular_cycle_density_per_vessel_length_px": (
            vascular_cycle_count / total_graph_length_px
            if total_graph_length_px > 0 and np.isfinite(vascular_cycle_count)
            else np.nan
        ),
        "loop_area_fraction": (
            loop_area_total_px2 / roi_area_px
            if roi_area_px and np.isfinite(loop_area_total_px2)
            else np.nan
        ),
        "loop_area_median_px2": float(
            graph_summary.get("enclosed_loop_area_median_px2", np.nan)
        ),
        "loop_perimeter_median_px": float(
            graph_summary.get("enclosed_loop_perimeter_median_px", np.nan)
        ),
        "loop_equivalent_diameter_median_px": float(
            graph_summary.get("enclosed_loop_equivalent_diameter_median_px", np.nan)
        ),
        **orientation_metrics,
    }

    if physical_enabled:
        skeleton_length_unit = float(
            graph_summary.get(f"sknw_total_edge_length_{unit}")
            or graph_summary.get(f"networkx_dfs_total_length_{unit}")
            or (total_graph_length_px * mean_scale)
        )
        metrics.update(
            {
                f"physical_unit": unit,
                f"physical_pixel_width_{unit}_per_px": scale_x,
                f"physical_pixel_height_{unit}_per_px": scale_y,
                f"roi_area_{unit}2": roi_area_unit2,
                f"vessel_area_{unit}2": vessel_area_px * scale_x * scale_y,
                f"vessel_area_density_{unit}2_per_{unit}2": (
                    (vessel_area_px * scale_x * scale_y) / roi_area_unit2
                    if roi_area_unit2 > 0
                    else np.nan
                ),
                f"skeleton_length_{unit}": skeleton_length_unit,
                f"skeleton_length_density_{unit}_per_{unit}2": (
                    skeleton_length_unit / roi_area_unit2
                    if roi_area_unit2 > 0
                    else np.nan
                ),
                f"branch_density_per_{unit}2": (
                    branch_count / roi_area_unit2 if roi_area_unit2 > 0 else np.nan
                ),
                f"endpoint_density_per_{unit}2": (
                    endpoint_count / roi_area_unit2 if roi_area_unit2 > 0 else np.nan
                ),
                f"internal_endpoint_density_per_{unit}2": (
                    internal_endpoint_count / roi_area_unit2
                    if roi_area_unit2 > 0
                    else np.nan
                ),
                f"component_density_per_{unit}2": (
                    component_count / roi_area_unit2 if roi_area_unit2 > 0 else np.nan
                ),
                f"vascular_cycle_density_per_{unit}2": (
                    vascular_cycle_count / roi_area_unit2
                    if roi_area_unit2 > 0 and np.isfinite(vascular_cycle_count)
                    else np.nan
                ),
                f"vascular_cycle_density_per_{unit}_vessel": (
                    vascular_cycle_count / skeleton_length_unit
                    if skeleton_length_unit > 0 and np.isfinite(vascular_cycle_count)
                    else np.nan
                ),
                f"loop_area_median_{unit}2": float(
                    graph_summary.get(f"enclosed_loop_area_median_{unit}2", np.nan)
                ),
                f"loop_perimeter_median_{unit}": float(
                    graph_summary.get(f"enclosed_loop_perimeter_median_{unit}", np.nan)
                ),
                f"loop_equivalent_diameter_median_{unit}": float(
                    graph_summary.get(
                        f"enclosed_loop_equivalent_diameter_median_{unit}",
                        np.nan,
                    )
                ),
                f"length_weighted_mean_diameter_{unit}": (
                    float(
                        np.average(
                            segment_diameters_physical[finite_diameter_physical],
                            weights=segment_lengths_physical[finite_diameter_physical],
                        )
                    )
                    if np.any(finite_diameter_physical)
                    else np.nan
                ),
                f"diameter_median_{unit}": diameter_percentiles_physical[50],
                f"diameter_iqr_{unit}": (
                    diameter_percentiles_physical[75]
                    - diameter_percentiles_physical[25]
                ),
                f"diameter_p90_{unit}": diameter_percentiles_physical[90],
                f"microvessel_length_{unit}": micro_length_physical,
                f"microvessel_length_fraction_{unit}_weighted": (
                    micro_length_physical / total_segment_length_physical
                    if total_segment_length_physical > 0
                    else np.nan
                ),
                f"high_curvature_threshold_per_{unit}": high_curvature_threshold_physical,
                f"median_curvature_per_{unit}": (
                    float(np.median(finite_curvature_physical))
                    if finite_curvature_physical.size
                    else np.nan
                ),
                f"high_curvature_length_density_{unit}_per_{unit}2": (
                    high_curvature_length_physical / roi_area_unit2
                    if roi_area_unit2 > 0
                    else np.nan
                ),
            }
        )
    return metrics

def sknw_cycle_rows_and_geometries(
    skeleton: np.ndarray,
    min_cycle_length_px: float = 0.0,
    max_cycle_length_px: float | None = None,
    cycle_length_unit: str = "px",
    scale: dict | None = None,
) -> tuple[list[dict], dict[int, list[np.ndarray]]]:
    if sknw is None:
        return [], {}
    scale = scale or {"enabled": False, "x_per_px": 1.0, "y_per_px": 1.0, "unit": ""}
    graph = sknw.build_sknw(skeleton.astype(np.uint16), multi=False)
    cycles = nx.cycle_basis(graph)
    rows: list[dict] = []
    geometries: dict[int, list[np.ndarray]] = {}
    cycle_id = 0
    for cycle in cycles:
        edge_paths = []
        length_px = 0.0
        for index, node in enumerate(cycle):
            neighbor = cycle[(index + 1) % len(cycle)]
            edge_data = graph.get_edge_data(node, neighbor, default={})
            points = edge_data.get("pts")
            if points is not None and len(points) >= 2:
                points = np.asarray(points, dtype=np.float32)
                edge_paths.append(points)
                length_px += path_length(points)
            else:
                source = np.asarray(graph.nodes[node].get("o"), dtype=np.float32)
                target = np.asarray(graph.nodes[neighbor].get("o"), dtype=np.float32)
                points = np.vstack([source, target]).astype(np.float32)
                edge_paths.append(points)
                length_px += path_length(points)
        length_physical = (
            sum(
                path_length_scaled(
                    points,
                    scale_y=float(scale["y_per_px"]),
                    scale_x=float(scale["x_per_px"]),
                )
                for points in edge_paths
            )
            if scale.get("enabled")
            else np.nan
        )
        filter_length = (
            length_physical
            if cycle_length_unit == "physical" and scale.get("enabled")
            else length_px
        )
        if filter_length < min_cycle_length_px:
            continue
        if max_cycle_length_px is not None and filter_length > max_cycle_length_px:
            continue
        cycle_id += 1
        all_points = np.vstack(edge_paths)
        min_y, min_x = np.min(all_points, axis=0)
        max_y, max_x = np.max(all_points, axis=0)
        row = {
            "cycle_id": cycle_id,
            "node_count": len(cycle),
            "edge_count": len(edge_paths),
            "length_px": float(length_px),
            "centroid_y": float(np.mean(all_points[:, 0])),
            "centroid_x": float(np.mean(all_points[:, 1])),
            "min_y": float(min_y),
            "min_x": float(min_x),
            "max_y": float(max_y),
            "max_x": float(max_x),
        }
        if scale.get("enabled"):
            row[f"length_{scale['unit']}"] = float(length_physical)
        rows.append(row)
        geometries[cycle_id] = edge_paths
    return rows, geometries

def contour_perimeter(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    closed = np.vstack([points_xy, points_xy[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())

def contour_perimeter_scaled(
    points_xy: np.ndarray,
    scale_y: float,
    scale_x: float,
) -> float:
    if len(points_xy) < 2:
        return 0.0
    yx = np.column_stack([points_xy[:, 1], points_xy[:, 0]]).astype(np.float64)
    closed = np.vstack([yx, yx[0]])
    diff = np.diff(closed, axis=0)
    dy = diff[:, 0] * scale_y
    dx = diff[:, 1] * scale_x
    return float(np.hypot(dy, dx).sum())

def enclosed_loop_rows_and_geometries(
    skeleton: np.ndarray,
    roi_mask: np.ndarray,
    scale: dict | None = None,
    min_perimeter: float = 0.0,
    max_perimeter: float | None = None,
    perimeter_unit: str = "px",
    min_equivalent_diameter: float = 0.0,
    max_aspect_ratio: float | None = None,
) -> tuple[list[dict], dict[int, list[np.ndarray]], dict]:
    scale = scale or {"enabled": False, "x_per_px": 1.0, "y_per_px": 1.0, "unit": ""}
    analysis_mask = roi_mask.astype(bool)
    barrier = skeleton.astype(bool) | ~analysis_mask
    background = (~barrier) & analysis_mask
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        background.astype(np.uint8),
        connectivity=4,
    )

    rows: list[dict] = []
    geometries: dict[int, list[np.ndarray]] = {}
    loop_id = 0
    height, width = skeleton.shape
    unit = scale.get("unit", "")
    scale_enabled = bool(scale.get("enabled"))
    scale_x = float(scale.get("x_per_px", 1.0))
    scale_y = float(scale.get("y_per_px", 1.0))
    area_scale = scale_x * scale_y

    for label in range(1, count):
        component = labels == label
        if not np.any(component):
            continue

        touches_image_boundary = (
            np.any(component[0, :])
            or np.any(component[-1, :])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )
        touches_roi_boundary = np.any(cv2.dilate(component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) & (~analysis_mask))
        if touches_image_boundary or touches_roi_boundary:
            continue

        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
        if len(contour) < 3:
            continue

        area_px = float(stats[label, cv2.CC_STAT_AREA])
        perimeter_px = contour_perimeter(contour)
        equivalent_diameter_px = float(2.0 * np.sqrt(area_px / np.pi))
        physical_values = {}
        filter_perimeter = perimeter_px
        filter_equivalent_diameter = equivalent_diameter_px
        if scale_enabled:
            area_physical = area_px * area_scale
            perimeter_physical = contour_perimeter_scaled(
                contour,
                scale_y=scale_y,
                scale_x=scale_x,
            )
            equivalent_diameter_physical = float(2.0 * np.sqrt(area_physical / np.pi))
            physical_values = {
                f"area_{unit}2": float(area_physical),
                f"perimeter_{unit}": float(perimeter_physical),
                f"length_{unit}": float(perimeter_physical),
                f"equivalent_diameter_{unit}": equivalent_diameter_physical,
            }
            if perimeter_unit == "physical":
                filter_perimeter = perimeter_physical
                filter_equivalent_diameter = equivalent_diameter_physical

        if filter_perimeter < min_perimeter:
            continue
        if max_perimeter is not None and filter_perimeter > max_perimeter:
            continue
        if filter_equivalent_diameter < min_equivalent_diameter:
            continue
        if max_aspect_ratio is not None and area_px > 0:
            aspect_ratio = (perimeter_px * perimeter_px) / (4.0 * np.pi * area_px)
            if aspect_ratio > max_aspect_ratio:
                continue

        loop_id += 1
        centroid_x, centroid_y = centroids[label]
        yx_contour = np.column_stack([contour[:, 1], contour[:, 0]]).astype(np.float32)
        row = {
            "cycle_id": loop_id,
            "loop_id": loop_id,
            "cycle_type": "enclosed_face",
            "area_px2": area_px,
            "perimeter_px": perimeter_px,
            "length_px": perimeter_px,
            "equivalent_diameter_px": equivalent_diameter_px,
            "perimeter_area_shape_index": (
                float((perimeter_px * perimeter_px) / (4.0 * np.pi * area_px))
                if area_px > 0
                else np.nan
            ),
            "centroid_y": float(centroid_y),
            "centroid_x": float(centroid_x),
            "min_y": float(stats[label, cv2.CC_STAT_TOP]),
            "min_x": float(stats[label, cv2.CC_STAT_LEFT]),
            "max_y": float(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] - 1),
            "max_x": float(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] - 1),
            "touches_image_boundary": bool(touches_image_boundary),
            "touches_roi_boundary": bool(touches_roi_boundary),
        }
        row.update(physical_values)
        rows.append(row)
        geometries[loop_id] = [yx_contour]

    loop_areas = np.asarray([row["area_px2"] for row in rows], dtype=np.float64)
    loop_perimeters = np.asarray([row["perimeter_px"] for row in rows], dtype=np.float64)
    summary = {
        "enclosed_loop_count": len(rows),
        "enclosed_loop_area_total_px2": safe_stat(loop_areas, np.sum),
        "enclosed_loop_area_median_px2": safe_stat(loop_areas, np.median),
        "enclosed_loop_perimeter_median_px": safe_stat(loop_perimeters, np.median),
        "enclosed_loop_equivalent_diameter_median_px": (
            float(2.0 * np.sqrt(np.median(loop_areas) / np.pi))
            if loop_areas.size
            else np.nan
        ),
    }
    if scale_enabled:
        loop_areas_physical = np.asarray(
            [row[f"area_{unit}2"] for row in rows],
            dtype=np.float64,
        )
        loop_perimeters_physical = np.asarray(
            [row[f"perimeter_{unit}"] for row in rows],
            dtype=np.float64,
        )
        summary.update(
            {
                f"enclosed_loop_area_total_{unit}2": safe_stat(
                    loop_areas_physical,
                    np.sum,
                ),
                f"enclosed_loop_area_median_{unit}2": safe_stat(
                    loop_areas_physical,
                    np.median,
                ),
                f"enclosed_loop_perimeter_median_{unit}": safe_stat(
                    loop_perimeters_physical,
                    np.median,
                ),
                f"enclosed_loop_equivalent_diameter_median_{unit}": (
                    float(2.0 * np.sqrt(np.median(loop_areas_physical) / np.pi))
                    if loop_areas_physical.size
                    else np.nan
                ),
            }
        )
    return rows, geometries, summary

def loop_perimeter_distribution_rows(
    cycle_rows: list[dict],
    scale: dict,
) -> list[dict]:
    if scale.get("enabled"):
        unit = scale["unit"]
        value_key = f"perimeter_{unit}"
        unit_label = unit
        bins = [
            ("<1", 0.0, 1.0),
            ("1-2", 1.0, 2.0),
            ("2-5", 2.0, 5.0),
            ("5-10", 5.0, 10.0),
            (">10", 10.0, np.inf),
        ]
    else:
        value_key = "perimeter_px"
        unit_label = "px"
        bins = [
            ("<25", 0.0, 25.0),
            ("25-50", 25.0, 50.0),
            ("50-100", 50.0, 100.0),
            ("100-250", 100.0, 250.0),
            (">250", 250.0, np.inf),
        ]

    values = np.asarray(
        [row.get(value_key, np.nan) for row in cycle_rows],
        dtype=np.float64,
    )
    rows = []
    for label, low, high in bins:
        if np.isinf(high):
            in_bin = values >= low
        else:
            in_bin = (values >= low) & (values < high)
        rows.append(
            {
                "category": f"loop_perimeter_{label}_{unit_label}",
                "method": "enclosed_background_faces_perimeter_distribution",
                "count": int(np.count_nonzero(in_bin)),
                "total_length_px": np.nan,
                "note": f"Perimeter bin [{low}, {high}) in {unit_label}.",
            }
        )
    return rows

def oriented_line_kernel(
    theta: float,
    sigma_parallel: float,
    sigma_perp: float,
) -> np.ndarray:
    radius = int(np.ceil(3.0 * max(sigma_parallel, sigma_perp)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float32)
    along = x * np.cos(theta) + y * np.sin(theta)
    across = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(
        -0.5
        * (
            (along / float(sigma_parallel)) ** 2
            + (across / float(sigma_perp)) ** 2
        )
    ).astype(np.float32)
    kernel -= float(np.mean(kernel))
    norm = float(np.sum(np.abs(kernel)))
    if norm > 0:
        kernel /= norm
    return kernel

def orientation_score_bank(
    image: np.ndarray,
    orientation_count: int,
    sigma_parallel: float,
    sigma_perp: float,
) -> np.ndarray:
    source = image.astype(np.float32)
    finite = source[np.isfinite(source)]
    if finite.size and float(finite.max()) > float(finite.min()):
        source = (source - float(finite.min())) / (
            float(finite.max()) - float(finite.min())
        )
    source = np.nan_to_num(source, nan=0.0, posinf=1.0, neginf=0.0)

    responses = []
    for theta in np.linspace(0.0, np.pi, int(orientation_count), endpoint=False):
        kernel = oriented_line_kernel(theta, sigma_parallel, sigma_perp)
        response = cv2.filter2D(
            source,
            cv2.CV_32F,
            kernel,
            borderType=cv2.BORDER_REFLECT,
        )
        responses.append(np.maximum(response, 0.0))
    return np.stack(responses, axis=0).astype(np.float32)

def dominant_orientation_counts(
    responses: np.ndarray,
    threshold_sigma: float,
) -> np.ndarray:
    threshold = float(threshold_sigma) * float(np.std(responses))
    prev_response = np.roll(responses, 1, axis=0)
    next_response = np.roll(responses, -1, axis=0)
    maxima = (
        (responses >= prev_response)
        & (responses >= next_response)
        & (responses > threshold)
    )
    return np.sum(maxima, axis=0).astype(np.uint8)

def dilated_cycle_sample_mask(
    geometries: dict[int, list[np.ndarray]],
    cycle_id: int,
    shape: tuple[int, int],
    radius: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for points in geometries.get(cycle_id, []):
        rounded = np.rint(points).astype(np.int32)
        valid = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < shape[0])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < shape[1])
        )
        rounded = rounded[valid]
        if rounded.size:
            mask[rounded[:, 0], rounded[:, 1]] = 1
    if radius > 0 and np.any(mask):
        size = 2 * int(radius) + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    return mask.astype(bool)

def filter_cycles_by_orientation_score(
    cycle_rows: list[dict],
    cycle_geometries: dict[int, list[np.ndarray]],
    source_image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[int, list[np.ndarray]], dict]:
    summary = {
        "cycle_orientation_filter_enabled": bool(args.enable_cycle_orientation_filter),
        "cycle_orientation_filter_input_count": len(cycle_rows),
        "cycle_orientation_filter_removed_count": 0,
        "cycle_orientation_filter_kept_count": len(cycle_rows),
    }
    if not args.enable_cycle_orientation_filter or not cycle_rows:
        return cycle_rows, cycle_geometries, summary

    responses = orientation_score_bank(
        source_image,
        orientation_count=args.cycle_orientation_count,
        sigma_parallel=args.cycle_orientation_sigma_parallel,
        sigma_perp=args.cycle_orientation_sigma_perp,
    )
    dominant_counts = dominant_orientation_counts(
        responses,
        threshold_sigma=args.cycle_orientation_response_threshold,
    )

    kept_rows: list[dict] = []
    kept_geometries: dict[int, list[np.ndarray]] = {}
    removed_count = 0
    for row in cycle_rows:
        cycle_id = int(row["cycle_id"])
        sample_mask = dilated_cycle_sample_mask(
            cycle_geometries,
            cycle_id,
            dominant_counts.shape,
            radius=args.cycle_orientation_sample_radius,
        )
        sample_counts = dominant_counts[sample_mask]
        sample_count = int(sample_counts.size)
        junction_like = sample_counts >= int(args.cycle_orientation_min_dominant)
        junction_point_count = int(np.count_nonzero(junction_like))
        junction_fraction = (
            junction_point_count / float(sample_count) if sample_count else 0.0
        )
        max_dominant = int(np.max(sample_counts)) if sample_count else 0
        keep = (
            junction_point_count >= int(args.cycle_orientation_min_junction_points)
            or junction_fraction >= float(args.cycle_orientation_min_junction_fraction)
        )
        row.update(
            {
                "orientation_filter_keep": bool(keep),
                "orientation_sample_points": sample_count,
                "orientation_junction_like_points": junction_point_count,
                "orientation_junction_like_fraction": float(junction_fraction),
                "orientation_max_dominant_count": max_dominant,
            }
        )
        if keep:
            kept_rows.append(row)
            kept_geometries[cycle_id] = cycle_geometries.get(cycle_id, [])
        else:
            removed_count += 1

    summary.update(
        {
            "cycle_orientation_filter_removed_count": removed_count,
            "cycle_orientation_filter_kept_count": len(kept_rows),
            "cycle_orientation_filter_orientation_count": int(
                args.cycle_orientation_count
            ),
            "cycle_orientation_filter_min_dominant": int(
                args.cycle_orientation_min_dominant
            ),
            "cycle_orientation_filter_min_junction_points": int(
                args.cycle_orientation_min_junction_points
            ),
            "cycle_orientation_filter_min_junction_fraction": float(
                args.cycle_orientation_min_junction_fraction
            ),
        }
    )
    return kept_rows, kept_geometries, summary

def enclosed_loop_summary_from_rows(
    rows: list[dict],
    scale: dict,
) -> dict:
    loop_areas = np.asarray([row["area_px2"] for row in rows], dtype=np.float64)
    loop_perimeters = np.asarray(
        [row["perimeter_px"] for row in rows],
        dtype=np.float64,
    )
    summary = {
        "enclosed_loop_count": len(rows),
        "enclosed_loop_area_total_px2": safe_stat(loop_areas, np.sum),
        "enclosed_loop_area_median_px2": safe_stat(loop_areas, np.median),
        "enclosed_loop_perimeter_median_px": safe_stat(loop_perimeters, np.median),
        "enclosed_loop_equivalent_diameter_median_px": (
            float(2.0 * np.sqrt(np.median(loop_areas) / np.pi))
            if loop_areas.size
            else np.nan
        ),
    }
    if scale.get("enabled"):
        unit = scale["unit"]
        loop_areas_physical = np.asarray(
            [row[f"area_{unit}2"] for row in rows],
            dtype=np.float64,
        )
        loop_perimeters_physical = np.asarray(
            [row[f"perimeter_{unit}"] for row in rows],
            dtype=np.float64,
        )
        summary.update(
            {
                f"enclosed_loop_area_total_{unit}2": safe_stat(
                    loop_areas_physical,
                    np.sum,
                ),
                f"enclosed_loop_area_median_{unit}2": safe_stat(
                    loop_areas_physical,
                    np.median,
                ),
                f"enclosed_loop_perimeter_median_{unit}": safe_stat(
                    loop_perimeters_physical,
                    np.median,
                ),
                f"enclosed_loop_equivalent_diameter_median_{unit}": (
                    float(2.0 * np.sqrt(np.median(loop_areas_physical) / np.pi))
                    if loop_areas_physical.size
                    else np.nan
                ),
            }
        )
    return summary

def save_cycle_overlay(
    probability: np.ndarray,
    cycle_rows: list[dict],
    cycle_geometries: dict[int, list[np.ndarray]],
    path: Path,
) -> None:
    height, width = probability.shape
    fig_width = 14
    fig_height = fig_width * height / width
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    ax.imshow(probability, cmap="gray")
    colors = plt.cm.turbo(np.linspace(0, 1, max(2, min(len(cycle_rows), 256))))

    for index, row in enumerate(cycle_rows):
        cycle_id = int(row["cycle_id"])
        color = colors[index % len(colors)]
        for points in cycle_geometries.get(cycle_id, []):
            if len(points) < 2:
                continue
            ax.plot(
                points[:, 1],
                points[:, 0],
                color="black",
                linewidth=3.0,
                alpha=0.8,
                solid_capstyle="round",
            )
            ax.plot(
                points[:, 1],
                points[:, 0],
                color=color,
                linewidth=1.5,
                alpha=0.95,
                solid_capstyle="round",
            )

    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def unified_cycle_summary_rows(
    graph_summary: dict,
    split_merge_count: int,
) -> tuple[dict, list[dict]]:
    sknw_cycle_rank = graph_summary.get("sknw_cycle_rank", np.nan)
    sknw_length = graph_summary.get("sknw_total_edge_length_px", np.nan)
    sknw_available = bool(graph_summary.get("sknw_available", False))
    if sknw_available and np.isfinite(sknw_cycle_rank):
        primary_method = "sknw_condensed_graph"
        primary_total_length_px = float(sknw_length)
    else:
        primary_method = "networkx_pixel_graph"
        primary_total_length_px = float(
            graph_summary.get("networkx_dfs_total_length_px", np.nan)
        )

    unified = {
        "vascular_graph_total_length_method": primary_method,
        "vascular_graph_total_length_px": primary_total_length_px,
        "direct_parallel_split_merge_count": int(split_merge_count),
    }
    rows = [
        {
            "category": "direct_parallel_split_merge",
            "method": "same_endpoint_pair_parallel_segments",
            "count": int(split_merge_count),
            "total_length_px": np.nan,
            "note": "Specific split-merge subtype, not the total cycle count.",
        },
    ]
    return unified, rows
