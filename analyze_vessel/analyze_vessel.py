from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from vascular_analysis.common import cv2, distance_transform_edt, np, skeletonize, time
    from vascular_analysis.cli import parse_args, validate_args
    from vascular_analysis.io_utils import load_grayscale, require_same_shape, resolve_multi_output_maps
    from vascular_analysis.multicontrast import make_multicontrast_input
    from vascular_analysis.preprocessing import fill_small_skeleton_loops, full_density_roi, make_binary_mask, make_density_roi, smooth_probability
    from vascular_analysis.summary_rows import public_metrics_summary_row, public_summary_row
    from vascular_analysis.physical_scale import apply_physical_parameter_overrides, configure_physical_scale, cycle_length_filter_values, path_length, path_length_scaled, physical_scale_from_args
    from vascular_analysis.graph_analysis import networkx_graph_analysis, skeleton_component_edge_lengths, skeleton_component_labels, sknw_graph_analysis
    from vascular_analysis.cycle_analysis import enclosed_loop_rows_and_geometries, enclosed_loop_summary_from_rows, filter_cycles_by_orientation_score, loop_perimeter_distribution_rows, save_cycle_overlay, sknw_cycle_rows_and_geometries, unified_cycle_summary_rows, vascular_metrics_summary
    from vascular_analysis.skeleton_geometry import neighbor_count, normal_diameters_for_segment, prepare_segments, prune_short_junction_links, prune_terminal_branches, rasterize_segments, sample_image, segment_component_id, segment_curvature, trace_segments_with_nodes
    from vascular_analysis.visualization import parse_id_list, safe_stat, save_combined_curvature_overlay, save_density_roi_overlay, save_overlay, save_pruning_overlay, save_scalar_map, save_thick_vessel_overlay, write_csv
    from vascular_analysis.advanced_analysis import junction_branch_curvature_analysis, node_key, save_split_merge_overlay, split_merge_analysis, thick_vessel_analysis
else:
    from .vascular_analysis.common import cv2, distance_transform_edt, np, skeletonize, time
    from .vascular_analysis.cli import parse_args, validate_args
    from .vascular_analysis.io_utils import load_grayscale, require_same_shape, resolve_multi_output_maps
    from .vascular_analysis.multicontrast import make_multicontrast_input
    from .vascular_analysis.preprocessing import fill_small_skeleton_loops, full_density_roi, make_binary_mask, make_density_roi, smooth_probability
    from .vascular_analysis.summary_rows import public_metrics_summary_row, public_summary_row
    from .vascular_analysis.physical_scale import apply_physical_parameter_overrides, configure_physical_scale, cycle_length_filter_values, path_length, path_length_scaled, physical_scale_from_args
    from .vascular_analysis.graph_analysis import networkx_graph_analysis, skeleton_component_edge_lengths, skeleton_component_labels, sknw_graph_analysis
    from .vascular_analysis.cycle_analysis import enclosed_loop_rows_and_geometries, enclosed_loop_summary_from_rows, filter_cycles_by_orientation_score, loop_perimeter_distribution_rows, save_cycle_overlay, sknw_cycle_rows_and_geometries, unified_cycle_summary_rows, vascular_metrics_summary
    from .vascular_analysis.skeleton_geometry import neighbor_count, normal_diameters_for_segment, prepare_segments, prune_short_junction_links, prune_terminal_branches, rasterize_segments, sample_image, segment_component_id, segment_curvature, trace_segments_with_nodes
    from .vascular_analysis.visualization import parse_id_list, safe_stat, save_combined_curvature_overlay, save_density_roi_overlay, save_overlay, save_pruning_overlay, save_scalar_map, save_thick_vessel_overlay, write_csv
    from .vascular_analysis.advanced_analysis import junction_branch_curvature_analysis, node_key, save_split_merge_overlay, split_merge_analysis, thick_vessel_analysis


def main() -> None:
    started_at = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - started_at
        print(f"[{elapsed:7.1f}s] {message}", flush=True)

    args = parse_args()
    validate_args(args)

    input_path = args.input
    if args.multicontrast_image is not None:
        output_dir = (
            args.output_dir
            or Path("predictions")
            / f"{args.multicontrast_image.stem}_multicontrast_vessel_analysis"
        )
    elif args.multi_output_input is not None:
        output_dir = (
            args.output_dir
            or args.multi_output_input.parent / "vessel_geometry_analysis"
        )
    else:
        output_dir = (
            args.output_dir
            or input_path.parent / "vessel_geometry_analysis"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    images_dir = output_dir / "images"
    if not args.density_roi_visual_only or args.multicontrast_image is not None:
        csv_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.multicontrast_image is not None:
        if args.threshold is None:
            args.threshold = args.multicontrast_final_threshold
        if args.measurement_threshold is None:
            args.measurement_threshold = args.multicontrast_final_threshold
        input_path, _ = make_multicontrast_input(args, output_dir, progress)

    (
        multi_output_mask,
        centerline_probability,
        predicted_distance_map,
    ) = resolve_multi_output_maps(args)
    if multi_output_mask is not None:
        probability = multi_output_mask
        input_path = args.multi_output_input
    else:
        probability = load_grayscale(input_path)
    require_same_shape(probability, centerline_probability, "centerline map")
    require_same_shape(probability, predicted_distance_map, "distance map")
    physical_scale = configure_physical_scale(args, probability.shape)
    apply_physical_parameter_overrides(args, physical_scale)
    if physical_scale["enabled"] and args.cycle_length_unit == "px":
        args.cycle_length_unit = "physical"
    cycle_filter = cycle_length_filter_values(args, physical_scale)
    progress("loaded input")

    skeleton_source = (
        centerline_probability
        if centerline_probability is not None
        else probability
    )
    skeleton_threshold_arg = (
        args.centerline_threshold
        if centerline_probability is not None
        else args.threshold
    )
    # The blurred map is used only for a stable skeleton.
    skeleton_probability = smooth_probability(
        skeleton_source,
        args.sigma_y,
        args.sigma_x,
    )
    skeleton_hole_size = (
        args.min_hole_size
        if args.skeleton_min_hole_size is None
        else args.skeleton_min_hole_size
    )
    measurement_hole_size = (
        args.min_hole_size
        if args.measurement_min_hole_size is None
        else args.measurement_min_hole_size
    )
    skeleton_binary, skeleton_threshold = make_binary_mask(
        skeleton_probability,
        skeleton_threshold_arg,
        args.min_object_size,
        skeleton_hole_size,
        args.mask_closing_radius,
    )
    skeleton_binary_before_density_roi = skeleton_binary.copy()
    progress("made skeleton mask")

    # Diameter is measured from an unblurred mask so Gaussian smoothing does
    # not artificially widen vessels.
    measurement_binary, measurement_threshold = make_binary_mask(
        probability,
        args.measurement_threshold,
        args.min_object_size,
        measurement_hole_size,
        args.mask_closing_radius,
    )
    measurement_binary_before_density_roi = measurement_binary.copy()
    density_roi_requested = (
        args.auto_density_roi
        or args.density_roi_visual_only
        or args.density_roi_axis != "y"
        or args.density_roi_threshold != 0.005
        or args.density_roi_min_run != 25
        or args.density_roi_bridge_gap != 15
    )
    if density_roi_requested:
        (
            density_roi_visual_mask,
            density_roi_profile_rows,
            density_roi_removed_run_rows,
            detected_density_roi_summary,
        ) = make_density_roi(
            measurement_binary,
            axis=args.density_roi_axis,
            threshold=args.density_roi_threshold,
            min_run=args.density_roi_min_run,
            bridge_gap=args.density_roi_bridge_gap,
        )
        if args.density_roi_visual_only:
            roi_mask, _, _, density_roi_summary = full_density_roi(
                measurement_binary.shape
            )
            progress(
                "detected density ROI for visualization only "
                f"(would remove {detected_density_roi_summary['density_roi_removed_area_fraction']:.3%}; "
                f"remaining {detected_density_roi_summary['density_roi_width_px']}x"
                f"{detected_density_roi_summary['density_roi_height_px']} px)"
            )
        else:
            roi_mask = density_roi_visual_mask
            density_roi_summary = detected_density_roi_summary
            skeleton_binary &= roi_mask
            measurement_binary &= roi_mask
            progress(
                "applied density ROI "
                f"(removed {density_roi_summary['density_roi_removed_area_fraction']:.3%})"
            )
    else:
        (
            roi_mask,
            density_roi_profile_rows,
            density_roi_removed_run_rows,
            density_roi_summary,
        ) = full_density_roi(measurement_binary.shape)
        density_roi_visual_mask = roi_mask
    if args.density_roi_visual_only:
        save_density_roi_overlay(
            probability,
            measurement_binary_before_density_roi,
            density_roi_visual_mask,
            images_dir / "density_roi_overlay.png",
        )
        progress("saved density ROI overlay")
        return
    if predicted_distance_map is not None:
        diameter_map = np.where(
            measurement_binary,
            np.clip(predicted_distance_map, 0.0, None)
            * float(args.distance_to_diameter_scale),
            0.0,
        ).astype(np.float32)
    else:
        diameter_map = (
            2.0 * distance_transform_edt(measurement_binary)
        ).astype(np.float32)
    scale_for_diameter = physical_scale_from_args(args)
    if scale_for_diameter["enabled"]:
        if predicted_distance_map is None:
            diameter_map_physical = (
                2.0
                * distance_transform_edt(
                    measurement_binary,
                    sampling=(
                        float(scale_for_diameter["y_per_px"]),
                        float(scale_for_diameter["x_per_px"]),
                    ),
                )
            ).astype(np.float32)
        else:
            diameter_map_physical = (
                diameter_map * float(scale_for_diameter["mean_per_px"])
            ).astype(np.float32)
    else:
        diameter_map_physical = None
    progress("made measurement mask and diameter map")

    skeleton_initial = skeletonize(skeleton_binary)
    progress("skeletonized")
    skeleton_initial, skeleton_loop_hole_pixels_filled = fill_small_skeleton_loops(
        skeleton_initial,
        args.skeleton_loop_hole_size,
    )
    if args.skeleton_loop_hole_size > 0:
        progress(
            "filled small skeleton loops "
            f"({skeleton_loop_hole_pixels_filled} enclosed pixels)"
        )

    skeleton_terminal_pruned, terminal_removed_mask, terminal_records = (
        prune_terminal_branches(
            skeleton_initial,
            probability,
            diameter_map,
            args,
        )
    )
    progress("pruned terminal branches")

    skeleton_clean, junction_removed_mask, junction_records = (
        prune_short_junction_links(
            skeleton_terminal_pruned,
            diameter_map,
            args.prune_junction_link_length,
            args.prune_junction_link_diameter_factor,
        )
    )
    removed_mask = terminal_removed_mask | junction_removed_mask
    progress("pruned junction links")

    degrees = neighbor_count(skeleton_clean)
    endpoint_count = int(np.sum(skeleton_clean & (degrees == 1)))
    endpoint_mask = skeleton_clean & (degrees == 1)
    internal_endpoint_mask = endpoint_mask.copy()
    internal_endpoint_mask[:2, :] = False
    internal_endpoint_mask[-2:, :] = False
    internal_endpoint_mask[:, :2] = False
    internal_endpoint_mask[:, -2:] = False
    internal_endpoint_count = int(np.sum(internal_endpoint_mask & roi_mask))
    junction_pixel_count = int(np.sum(skeleton_clean & (degrees >= 3)))
    component_count, component_labels = skeleton_component_labels(skeleton_clean)
    component_lengths = skeleton_component_edge_lengths(
        skeleton_clean,
        component_labels,
        component_count,
    )
    networkx_summary = networkx_graph_analysis(
        skeleton_clean,
        args.networkx_cycle_basis_limit,
        physical_scale,
    )
    sknw_summary = sknw_graph_analysis(skeleton_clean, physical_scale)
    graph_summary = {}
    graph_summary.update(networkx_summary)
    graph_summary.update(sknw_summary)
    progress("analyzed skeleton graph with networkx/sknw")

    raw_segments = trace_segments_with_nodes(skeleton_clean)
    progress(f"traced {len(raw_segments)} raw segments")
    prepared_segments = prepare_segments(
        raw_segments,
        spacing=args.resample_spacing,
        smooth_window=args.smooth_skeleton_window,
    )
    progress(f"prepared {len(prepared_segments)} segments")

    display_skeleton = rasterize_segments(
        prepared_segments,
        skeleton_clean.shape,
        args.smooth_skeleton_line_width,
    )

    point_rows: list[dict] = []
    segment_rows: list[dict] = []
    segment_geometries: dict[int, np.ndarray] = {}
    segment_normal_diameter_profiles: dict[int, np.ndarray] = {}
    scale = physical_scale_from_args(args)
    physical_curvature_key = (
        f"curvature_per_{scale['unit']}" if scale["enabled"] else ""
    )
    component_accumulators = {
        component_id: {
            "segment_count": 0,
            "point_count": 0,
            "length_physical": 0.0,
            "normal_diameters": [],
            "distance_transform_diameters": [],
            "curvatures": [],
            "curvatures_physical": [],
        }
        for component_id in range(1, component_count + 1)
    }

    total_length_px = 0.0
    total_length_physical = 0.0
    all_diameters_weighted: list[float] = []
    all_diameters_physical_weighted: list[float] = []
    all_normal_diameters_weighted: list[float] = []
    all_curvatures: list[float] = []
    all_curvatures_physical: list[float] = []

    for segment_id, segment in enumerate(prepared_segments, start=1):
        length_px = path_length(segment)
        if length_px < args.min_segment_length:
            continue

        segment_geometries[segment_id] = segment

        component_id = segment_component_id(segment, component_labels)

        diameters_px = sample_image(
            diameter_map,
            segment,
            order=1,
        )
        diameters_physical = (
            sample_image(
                diameter_map_physical,
                segment,
                order=1,
            )
            if diameter_map_physical is not None
            else np.asarray([], dtype=np.float32)
        )
        normal_diameters_px = normal_diameters_for_segment(
            segment,
            measurement_binary,
            step=args.normal_measure_step,
            max_half_width=args.normal_max_half_width,
            diameter_hints_px=diameters_px,
        )
        segment_normal_diameter_profiles[segment_id] = normal_diameters_px.copy()
        probabilities = sample_image(
            probability,
            segment,
            order=1,
        )
        positive_diameters = diameters_px[diameters_px > 0]
        positive_diameters_physical = diameters_physical[diameters_physical > 0]
        positive_normal_diameters = normal_diameters_px[
            np.isfinite(normal_diameters_px) & (normal_diameters_px > 0)
        ]
        median_diameter_px = (
            float(np.median(positive_diameters))
            if positive_diameters.size
            else 0.0
        )

        raw_curvature_margin_px = max(
            args.curvature_end_margin,
            args.curvature_diameter_margin_factor * median_diameter_px,
        )
        curvature_margin_px = min(
            raw_curvature_margin_px,
            args.curvature_margin_max_fraction * length_px,
        )
        curvature_per_px = segment_curvature(
            segment,
            spacing=args.resample_spacing,
            requested_window=args.curvature_window,
            end_margin_px=curvature_margin_px,
        )
        curvature_per_physical = None
        if scale["enabled"]:
            mean_scale = float(scale["mean_per_px"])
            curvature_per_physical = segment_curvature(
                segment,
                spacing=args.resample_spacing * mean_scale,
                requested_window=args.curvature_window,
                end_margin_px=curvature_margin_px * mean_scale,
                scale_y=float(scale["y_per_px"]),
                scale_x=float(scale["x_per_px"]),
            )

        endpoint_distance_px = float(
            np.linalg.norm(segment[-1] - segment[0])
        )
        length_physical_for_metrics = np.nan
        endpoint_distance_physical_for_metrics = np.nan
        if scale["enabled"]:
            scale_y = float(scale["y_per_px"])
            scale_x = float(scale["x_per_px"])
            endpoint_dy_physical = (segment[-1, 0] - segment[0, 0]) * scale_y
            endpoint_dx_physical = (segment[-1, 1] - segment[0, 1]) * scale_x
            length_physical_for_metrics = path_length_scaled(
                segment,
                scale_y,
                scale_x,
            )
            endpoint_distance_physical_for_metrics = float(
                np.hypot(endpoint_dy_physical, endpoint_dx_physical)
            )
            tortuosity = (
                length_physical_for_metrics / endpoint_distance_physical_for_metrics
                if endpoint_distance_physical_for_metrics > 1e-8
                else np.nan
            )
        else:
            tortuosity = (
                length_px / endpoint_distance_px
                if endpoint_distance_px > 1e-8
                else np.nan
            )

        valid_curvature = curvature_per_px[
            np.isfinite(curvature_per_px)
        ]
        valid_curvature_physical = (
            curvature_per_physical[np.isfinite(curvature_per_physical)]
            if curvature_per_physical is not None
            else np.asarray([], dtype=np.float32)
        )
        total_length_px += length_px
        if scale["enabled"] and np.isfinite(length_physical_for_metrics):
            total_length_physical += float(length_physical_for_metrics)
        all_diameters_weighted.extend(
            diameters_px[diameters_px > 0].tolist()
        )
        if positive_diameters_physical.size:
            all_diameters_physical_weighted.extend(
                positive_diameters_physical.tolist()
            )
        all_normal_diameters_weighted.extend(
            positive_normal_diameters.tolist()
        )
        all_curvatures.extend(valid_curvature.tolist())
        all_curvatures_physical.extend(valid_curvature_physical.tolist())
        if component_id > 0:
            accumulator = component_accumulators[component_id]
            accumulator["segment_count"] += 1
            accumulator["point_count"] += int(len(segment))
            if scale["enabled"] and np.isfinite(length_physical_for_metrics):
                accumulator["length_physical"] += float(length_physical_for_metrics)
            accumulator["distance_transform_diameters"].extend(
                diameters_px[diameters_px > 0].tolist()
            )
            accumulator["normal_diameters"].extend(
                positive_normal_diameters.tolist()
            )
            accumulator["curvatures"].extend(valid_curvature.tolist())
            accumulator["curvatures_physical"].extend(
                valid_curvature_physical.tolist()
            )

        segment_row = {
            "segment_id": segment_id,
            "component_id": component_id,
            "start_node": node_key(segment[0]),
            "end_node": node_key(segment[-1]),
            "start_y": float(segment[0, 0]),
            "start_x": float(segment[0, 1]),
            "end_y": float(segment[-1, 0]),
            "end_x": float(segment[-1, 1]),
            "min_x": float(np.min(segment[:, 1])),
            "max_x": float(np.max(segment[:, 1])),
            "min_y": float(np.min(segment[:, 0])),
            "max_y": float(np.max(segment[:, 0])),
            "n_points": int(len(segment)),
            "length_px": float(length_px),
            "endpoint_distance_px": endpoint_distance_px,
            "tortuosity": float(tortuosity),
            "mean_probability": safe_stat(
                probabilities,
                np.mean,
            ),
            "mean_diameter_px": safe_stat(
                positive_diameters,
                np.mean,
            ),
            "median_diameter_px": safe_stat(
                positive_diameters,
                np.median,
            ),
            "p95_diameter_px": safe_stat(
                positive_diameters,
                lambda values: np.percentile(values, 95),
            ),
            "mean_normal_diameter_px": safe_stat(
                positive_normal_diameters,
                np.mean,
            ),
            "median_normal_diameter_px": safe_stat(
                positive_normal_diameters,
                np.median,
            ),
            "p95_normal_diameter_px": safe_stat(
                positive_normal_diameters,
                lambda values: np.percentile(values, 95),
            ),
            "mean_curvature_per_px": safe_stat(
                valid_curvature,
                np.mean,
            ),
            "median_curvature_per_px": safe_stat(
                valid_curvature,
                np.median,
            ),
            "p95_curvature_per_px": safe_stat(
                valid_curvature,
                lambda values: np.percentile(values, 95),
            ),
            "max_curvature_per_px": safe_stat(
                valid_curvature,
                np.max,
            ),
            "curvature_end_margin_px": float(curvature_margin_px),
        }
        if scale["enabled"]:
            unit = scale["unit"]
            mean_scale = float(scale["mean_per_px"])
            segment_row.update(
                {
                    f"length_{unit}": length_physical_for_metrics,
                    f"endpoint_distance_{unit}": endpoint_distance_physical_for_metrics,
                    f"mean_diameter_{unit}": safe_stat(
                        positive_diameters_physical,
                        np.mean,
                    ),
                    f"median_diameter_{unit}": safe_stat(
                        positive_diameters_physical,
                        np.median,
                    ),
                    f"p95_diameter_{unit}": safe_stat(
                        positive_diameters_physical,
                        lambda values: np.percentile(values, 95),
                    ),
                    f"mean_normal_diameter_{unit}": segment_row["mean_normal_diameter_px"] * mean_scale,
                    f"median_normal_diameter_{unit}": segment_row["median_normal_diameter_px"] * mean_scale,
                    f"p95_normal_diameter_{unit}": segment_row["p95_normal_diameter_px"] * mean_scale,
                    f"mean_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        np.mean,
                    ),
                    f"median_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        np.median,
                    ),
                    f"p95_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        lambda values: np.percentile(values, 95),
                    ),
                    f"max_curvature_per_{unit}": safe_stat(
                        valid_curvature_physical,
                        np.max,
                    ),
                }
            )

        if args.pixel_size_um is not None:
            pixel_size_um = args.pixel_size_um
            segment_row.update(
                {
                    "length_um": float(length_px * pixel_size_um),
                    "endpoint_distance_um": float(
                        endpoint_distance_px * pixel_size_um
                    ),
                    "mean_diameter_um": float(
                        segment_row["mean_diameter_px"] * pixel_size_um
                    ),
                    "median_diameter_um": float(
                        segment_row["median_diameter_px"] * pixel_size_um
                    ),
                    "p95_diameter_um": float(
                        segment_row["p95_diameter_px"] * pixel_size_um
                    ),
                    "mean_normal_diameter_um": float(
                        segment_row["mean_normal_diameter_px"] * pixel_size_um
                    ),
                    "median_normal_diameter_um": float(
                        segment_row["median_normal_diameter_px"] * pixel_size_um
                    ),
                    "p95_normal_diameter_um": float(
                        segment_row["p95_normal_diameter_px"] * pixel_size_um
                    ),
                    "mean_curvature_per_um": float(
                        segment_row["mean_curvature_per_px"] / pixel_size_um
                    ),
                    "median_curvature_per_um": float(
                        segment_row["median_curvature_per_px"] / pixel_size_um
                    ),
                    "p95_curvature_per_um": float(
                        segment_row["p95_curvature_per_px"] / pixel_size_um
                    ),
                    "max_curvature_per_um": float(
                        segment_row["max_curvature_per_px"] / pixel_size_um
                    ),
                }
            )

        segment_rows.append(segment_row)

        for point_index, (
            (y, x),
            probability_value,
            diameter_px,
            diameter_physical,
            normal_diameter_px,
            curvature_px,
            curvature_physical,
        ) in enumerate(
            zip(
                segment,
                probabilities,
                diameters_px,
                diameters_physical
                if diameter_map_physical is not None
                else np.full_like(diameters_px, np.nan),
                normal_diameters_px,
                curvature_per_px,
                curvature_per_physical
                if curvature_per_physical is not None
                else np.full_like(curvature_per_px, np.nan),
            )
        ):
            row = {
                "segment_id": segment_id,
                "component_id": component_id,
                "point_index": point_index,
                "y": float(y),
                "x": float(x),
                "probability": float(probability_value),
                "diameter_px": float(diameter_px),
                "normal_diameter_px": float(normal_diameter_px),
                "curvature_per_px": float(curvature_px),
            }
            if args.pixel_size_um is not None:
                row["diameter_um"] = float(
                    diameter_px * args.pixel_size_um
                )
                row["normal_diameter_um"] = float(
                    normal_diameter_px * args.pixel_size_um
                )
                row["curvature_per_um"] = float(
                    curvature_px / args.pixel_size_um
                )
            if scale["enabled"]:
                unit = scale["unit"]
                mean_scale = float(scale["mean_per_px"])
                row[f"diameter_{unit}"] = float(diameter_physical)
                row[f"normal_diameter_{unit}"] = float(normal_diameter_px * mean_scale)
                row[f"curvature_per_{unit}"] = float(curvature_physical)
            point_rows.append(row)
    progress(f"measured {len(segment_rows)} segments")

    diameter_values = np.asarray(
        all_diameters_weighted,
        dtype=np.float32,
    )
    curvature_values = np.asarray(
        all_curvatures,
        dtype=np.float32,
    )
    curvature_values_physical = np.asarray(
        all_curvatures_physical,
        dtype=np.float32,
    )
    normal_diameter_values = np.asarray(
        all_normal_diameters_weighted,
        dtype=np.float32,
    )
    diameter_values_physical = np.asarray(
        all_diameters_physical_weighted,
        dtype=np.float32,
    )
    component_rows: list[dict] = []
    for component_id, accumulator in component_accumulators.items():
        normal_values = np.asarray(
            accumulator["normal_diameters"],
            dtype=np.float32,
        )
        dt_values = np.asarray(
            accumulator["distance_transform_diameters"],
            dtype=np.float32,
        )
        component_curvatures = np.asarray(
            accumulator["curvatures"],
            dtype=np.float32,
        )
        component_curvatures_physical = np.asarray(
            accumulator["curvatures_physical"],
            dtype=np.float32,
        )
        row = {
            "component_id": component_id,
            "segment_count": int(accumulator["segment_count"]),
            "point_count": int(accumulator["point_count"]),
            "length_px": float(component_lengths.get(component_id, 0.0)),
            "mean_normal_diameter_px": safe_stat(normal_values, np.mean),
            "median_normal_diameter_px": safe_stat(normal_values, np.median),
            "p95_normal_diameter_px": safe_stat(
                normal_values,
                lambda values: np.percentile(values, 95),
            ),
            "mean_diameter_px": safe_stat(dt_values, np.mean),
            "median_diameter_px": safe_stat(dt_values, np.median),
            "p95_diameter_px": safe_stat(
                dt_values,
                lambda values: np.percentile(values, 95),
            ),
            "mean_curvature_per_px": safe_stat(component_curvatures, np.mean),
            "median_curvature_per_px": safe_stat(component_curvatures, np.median),
            "p95_curvature_per_px": safe_stat(
                component_curvatures,
                lambda values: np.percentile(values, 95),
            ),
        }
        if args.pixel_size_um is not None:
            pixel_size_um = args.pixel_size_um
            row.update(
                {
                    "length_um": float(row["length_px"] * pixel_size_um),
                    "mean_normal_diameter_um": float(
                        row["mean_normal_diameter_px"] * pixel_size_um
                    ),
                    "median_normal_diameter_um": float(
                        row["median_normal_diameter_px"] * pixel_size_um
                    ),
                    "p95_normal_diameter_um": float(
                        row["p95_normal_diameter_px"] * pixel_size_um
                    ),
                    "mean_diameter_um": float(row["mean_diameter_px"] * pixel_size_um),
                    "median_diameter_um": float(
                        row["median_diameter_px"] * pixel_size_um
                    ),
                    "p95_diameter_um": float(row["p95_diameter_px"] * pixel_size_um),
                    "mean_curvature_per_um": float(
                        row["mean_curvature_per_px"] / pixel_size_um
                    ),
                    "median_curvature_per_um": float(
                        row["median_curvature_per_px"] / pixel_size_um
                    ),
                    "p95_curvature_per_um": float(
                        row["p95_curvature_per_px"] / pixel_size_um
                    ),
                }
            )
        if scale["enabled"]:
            unit = scale["unit"]
            mean_scale = float(scale["mean_per_px"])
            row.update(
                {
                    f"length_{unit}": float(accumulator["length_physical"]),
                    f"mean_normal_diameter_{unit}": float(
                        row["mean_normal_diameter_px"] * mean_scale
                    ),
                    f"median_normal_diameter_{unit}": float(
                        row["median_normal_diameter_px"] * mean_scale
                    ),
                    f"p95_normal_diameter_{unit}": float(
                        row["p95_normal_diameter_px"] * mean_scale
                    ),
                    f"mean_diameter_{unit}": float(row["mean_diameter_px"] * mean_scale),
                    f"median_diameter_{unit}": float(
                        row["median_diameter_px"] * mean_scale
                    ),
                    f"p95_diameter_{unit}": float(row["p95_diameter_px"] * mean_scale),
                    f"mean_curvature_per_{unit}": safe_stat(
                        component_curvatures_physical,
                        np.mean,
                    ),
                    f"median_curvature_per_{unit}": safe_stat(
                        component_curvatures_physical,
                        np.median,
                    ),
                    f"p95_curvature_per_{unit}": safe_stat(
                        component_curvatures_physical,
                        lambda values: np.percentile(values, 95),
                    ),
                }
            )
        component_rows.append(row)

    thick_vessel_rows, thick_vessel_summary = thick_vessel_analysis(
        segment_rows,
        args,
    )
    progress("analyzed thick vessels")

    split_merge_rows, split_merge_point_rows, split_merge_geometries = (
        split_merge_analysis(
            segment_rows,
            segment_geometries,
            segment_normal_diameter_profiles,
            args,
        )
    )
    progress(f"analyzed {len(split_merge_rows)} split-merge groups")

    (
        junction_branch_rows,
        junction_branch_point_rows,
        junction_cluster_rows,
    ) = junction_branch_curvature_analysis(
        skeleton_clean,
        probability,
        diameter_map,
        measurement_binary,
        args,
    )
    progress(
        f"analyzed {len(junction_branch_rows)} junction branches "
        f"from {len(junction_cluster_rows)} clusters"
    )

    network_row = {
        "input": str(input_path),
        "skeleton_source": "centerline_output" if centerline_probability is not None else "mask_probability",
        "diameter_source": "distance_output" if predicted_distance_map is not None else "mask_distance_transform",
        "distance_to_diameter_scale": float(args.distance_to_diameter_scale),
        "skeleton_threshold": float(skeleton_threshold),
        "measurement_threshold": float(measurement_threshold),
        "skeleton_pixels_initial": int(skeleton_initial.sum()),
        "skeleton_loop_hole_size": int(args.skeleton_loop_hole_size),
        "skeleton_loop_hole_pixels_filled": int(skeleton_loop_hole_pixels_filled),
        "skeleton_pixels_final": int(skeleton_clean.sum()),
        "terminal_branches_removed": len(terminal_records),
        "junction_links_removed": len(junction_records),
        "endpoints": endpoint_count,
        "junction_pixels": junction_pixel_count,
        "connected_components": component_count,
        "measured_segments": len(segment_rows),
        "total_length_px": float(total_length_px),
        "total_component_edge_length_px": float(sum(component_lengths.values())),
        "mean_diameter_px": safe_stat(diameter_values, np.mean),
        "median_diameter_px": safe_stat(diameter_values, np.median),
        "p95_diameter_px": safe_stat(
            diameter_values,
            lambda values: np.percentile(values, 95),
        ),
        "mean_normal_diameter_px": safe_stat(normal_diameter_values, np.mean),
        "median_normal_diameter_px": safe_stat(normal_diameter_values, np.median),
        "p95_normal_diameter_px": safe_stat(
            normal_diameter_values,
            lambda values: np.percentile(values, 95),
        ),
        "mean_curvature_per_px": safe_stat(
            curvature_values,
            np.mean,
        ),
        "median_curvature_per_px": safe_stat(
            curvature_values,
            np.median,
        ),
        "p95_curvature_per_px": safe_stat(
            curvature_values,
            lambda values: np.percentile(values, 95),
        ),
    }
    network_row.update(density_roi_summary)
    network_row.update(thick_vessel_summary)
    network_row["junction_cluster_count"] = len(junction_cluster_rows)
    network_row["junction_branch_count"] = len(junction_branch_rows)
    network_row.update(graph_summary)
    cycle_unified_summary, cycle_summary_rows = unified_cycle_summary_rows(
        graph_summary,
        split_merge_count=len(split_merge_rows),
    )
    network_row.update(cycle_unified_summary)
    graph_summary.update(cycle_unified_summary)
    sknw_cycle_rows, sknw_cycle_geometries = sknw_cycle_rows_and_geometries(
        skeleton_clean,
        min_cycle_length_px=cycle_filter["min_physical"]
        if cycle_filter["unit_mode"] == "physical"
        else cycle_filter["min_px"],
        max_cycle_length_px=(
            cycle_filter["max_physical"]
            if cycle_filter["unit_mode"] == "physical"
            else cycle_filter["max_px"]
        )
        if np.isfinite(cycle_filter["max_px"])
        else None,
        cycle_length_unit=cycle_filter["unit_mode"],
        scale=physical_scale,
    )
    if density_roi_requested:
        cycle_skeleton = skeletonize(skeleton_binary_before_density_roi)
        cycle_skeleton, _ = fill_small_skeleton_loops(
            cycle_skeleton,
            args.skeleton_loop_hole_size,
        )
    else:
        cycle_skeleton = skeleton_clean

    cycle_rows, cycle_geometries, enclosed_loop_summary = (
        enclosed_loop_rows_and_geometries(
            cycle_skeleton,
            roi_mask,
            scale=physical_scale,
            min_perimeter=cycle_filter["min_physical"]
            if cycle_filter["unit_mode"] == "physical"
            else cycle_filter["min_px"],
            max_perimeter=(
                cycle_filter["max_physical"]
                if cycle_filter["unit_mode"] == "physical"
                else cycle_filter["max_px"]
            )
            if np.isfinite(cycle_filter["max_px"])
            else None,
            perimeter_unit=cycle_filter["unit_mode"],
            min_equivalent_diameter=float(args.min_cycle_equivalent_diameter),
            max_aspect_ratio=args.max_cycle_shape_index,
        )
    )
    cycle_rows, cycle_geometries, orientation_cycle_summary = (
        filter_cycles_by_orientation_score(
            cycle_rows,
            cycle_geometries,
            skeleton_source,
            args,
        )
    )
    enclosed_loop_summary = enclosed_loop_summary_from_rows(
        cycle_rows,
        physical_scale,
    )
    if args.enable_cycle_orientation_filter:
        progress(
            "filtered cycles by orientation score "
            f"({orientation_cycle_summary['cycle_orientation_filter_removed_count']} removed)"
        )
    network_row.update(enclosed_loop_summary)
    network_row.update(orientation_cycle_summary)
    network_row["vascular_cycle_count"] = len(cycle_rows)
    network_row["vascular_cycle_count_method"] = (
        "enclosed_background_faces_orientation_filtered"
        if args.enable_cycle_orientation_filter
        else "enclosed_background_faces"
    )
    network_row["sknw_cycle_basis_filtered_count"] = len(sknw_cycle_rows)
    network_row["cycle_length_filter_unit"] = cycle_filter["filter_unit"]
    network_row["cycle_length_filter_min_input"] = float(args.min_cycle_length)
    network_row["cycle_length_filter_max_input"] = (
        float(args.max_cycle_length) if args.max_cycle_length is not None else np.nan
    )
    network_row["cycle_length_filter_min_px"] = float(cycle_filter["min_px"])
    network_row["cycle_length_filter_max_px"] = float(cycle_filter["max_px"])
    if cycle_rows:
        cycle_lengths = np.asarray([row["length_px"] for row in cycle_rows], dtype=np.float32)
        network_row["cycle_length_median_px"] = safe_stat(cycle_lengths, np.median)
        network_row["cycle_length_p90_px"] = safe_stat(
            cycle_lengths,
            lambda values: np.percentile(values, 90),
        )
    else:
        network_row["cycle_length_median_px"] = np.nan
        network_row["cycle_length_p90_px"] = np.nan
    graph_summary["vascular_cycle_count"] = network_row["vascular_cycle_count"]
    graph_summary["vascular_cycle_count_method"] = network_row["vascular_cycle_count_method"]
    graph_summary.update(enclosed_loop_summary)
    graph_summary.update(orientation_cycle_summary)
    cycle_summary_rows.insert(
        0,
        {
            "category": "vascular_cycles",
            "method": network_row["vascular_cycle_count_method"],
            "count": network_row["vascular_cycle_count"],
            "total_length_px": float(
                np.sum([row["length_px"] for row in cycle_rows])
                if cycle_rows
                else 0.0
            ),
            "note": (
                "Final reported cycle count as enclosed background faces; "
                "optionally orientation-score filtered; "
                "matches vascular_cycles.csv and vascular_cycle_overlay.png."
            ),
        },
    )
    if args.enable_cycle_orientation_filter:
        cycle_summary_rows.append(
            {
                "category": "orientation_score_removed_cycles",
                "method": "bicros_inspired_cycle_candidate_filter",
                "count": int(
                    orientation_cycle_summary[
                        "cycle_orientation_filter_removed_count"
                    ]
                ),
                "total_length_px": np.nan,
                "note": (
                    "Removed only among detected cycle candidates when boundary "
                    "support lacked multi-orientation junction-like points."
                ),
            }
        )
    cycle_summary_rows.append(
        {
            "category": "sknw_cycle_basis_filtered",
            "method": "sknw_cycle_basis_length_filtered_auxiliary",
            "count": len(sknw_cycle_rows),
            "total_length_px": float(
                np.sum([row["length_px"] for row in sknw_cycle_rows])
                if sknw_cycle_rows
                else 0.0
            ),
            "note": "Auxiliary graph cycle-basis count; not used as final loop density.",
        }
    )
    cycle_summary_rows.extend(
        loop_perimeter_distribution_rows(cycle_rows, physical_scale)
    )
    metrics_summary = vascular_metrics_summary(
        measurement_binary,
        roi_mask,
        endpoint_count,
        internal_endpoint_count,
        component_count,
        component_lengths,
        prepared_segments,
        segment_rows,
        curvature_values,
        curvature_values_physical if scale["enabled"] else None,
        graph_summary,
        junction_cluster_rows,
        args,
    )
    metrics_summary["cycle_length_median_px"] = network_row["cycle_length_median_px"]
    metrics_summary["cycle_length_p90_px"] = network_row["cycle_length_p90_px"]
    if scale["enabled"]:
        unit = scale["unit"]
        cycle_lengths_physical = np.asarray(
            [row.get(f"length_{unit}", np.nan) for row in cycle_rows],
            dtype=np.float64,
        )
        cycle_lengths_physical = cycle_lengths_physical[
            np.isfinite(cycle_lengths_physical)
        ]
        metrics_summary[f"cycle_length_median_{unit}"] = (
            float(np.median(cycle_lengths_physical))
            if cycle_lengths_physical.size
            else np.nan
        )
        metrics_summary[f"cycle_length_p90_{unit}"] = (
            float(np.percentile(cycle_lengths_physical, 90))
            if cycle_lengths_physical.size
            else np.nan
        )
    if args.pixel_size_um is not None:
        network_row.update(
            {
                "pixel_size_um": float(args.pixel_size_um),
                "total_length_um": float(
                    total_length_px * args.pixel_size_um
                ),
                "total_component_edge_length_um": float(
                    network_row["total_component_edge_length_px"]
                    * args.pixel_size_um
                ),
                "mean_diameter_um": float(
                    network_row["mean_diameter_px"]
                    * args.pixel_size_um
                ),
                "median_diameter_um": float(
                    network_row["median_diameter_px"]
                    * args.pixel_size_um
                ),
                "p95_diameter_um": float(
                    network_row["p95_diameter_px"]
                    * args.pixel_size_um
                ),
                "mean_normal_diameter_um": float(
                    network_row["mean_normal_diameter_px"]
                    * args.pixel_size_um
                ),
                "median_normal_diameter_um": float(
                    network_row["median_normal_diameter_px"]
                    * args.pixel_size_um
                ),
                "p95_normal_diameter_um": float(
                    network_row["p95_normal_diameter_px"]
                    * args.pixel_size_um
                ),
                "mean_curvature_per_um": float(
                    network_row["mean_curvature_per_px"]
                    / args.pixel_size_um
                ),
                "median_curvature_per_um": float(
                    network_row["median_curvature_per_px"]
                    / args.pixel_size_um
                ),
                "p95_curvature_per_um": float(
                    network_row["p95_curvature_per_px"]
                    / args.pixel_size_um
                ),
                "thick_vessel_max_path_um": float(
                    network_row["thick_vessel_max_path_px"]
                    * args.pixel_size_um
                ),
                "networkx_dfs_total_length_um": float(
                    network_row["networkx_dfs_total_length_px"]
                    * args.pixel_size_um
                ),
                "vascular_graph_total_length_um": float(
                    network_row["vascular_graph_total_length_px"]
                    * args.pixel_size_um
                ),
            }
        )
    scale = physical_scale_from_args(args)
    if scale["enabled"]:
        unit = scale["unit"]
        mean_scale = float(scale["mean_per_px"])
        area_scale = float(scale["x_per_px"]) * float(scale["y_per_px"])
        network_row.update(
            {
                "physical_unit": unit,
                f"physical_pixel_width_{unit}_per_px": float(scale["x_per_px"]),
                f"physical_pixel_height_{unit}_per_px": float(scale["y_per_px"]),
                f"density_roi_area_{unit}2": float(
                    network_row["density_roi_area_px"] * area_scale
                ),
                f"density_roi_width_{unit}": float(
                    network_row["density_roi_width_px"] * float(scale["x_per_px"])
                ),
                f"density_roi_height_{unit}": float(
                    network_row["density_roi_height_px"] * float(scale["y_per_px"])
                ),
                f"cycle_length_filter_min_{unit}": float(
                    cycle_filter["min_physical"]
                ),
                f"cycle_length_filter_max_{unit}": (
                    float(cycle_filter["max_physical"])
                    if np.isfinite(cycle_filter["max_physical"])
                    else np.nan
                ),
                f"total_length_{unit}": float(total_length_physical),
                f"total_component_edge_length_{unit}": float(
                    sum(
                        row.get(f"length_{unit}", row["length_px"] * mean_scale)
                        for row in component_rows
                    )
                ),
                f"mean_diameter_{unit}": safe_stat(
                    diameter_values_physical,
                    np.mean,
                ),
                f"median_diameter_{unit}": safe_stat(
                    diameter_values_physical,
                    np.median,
                ),
                f"p95_diameter_{unit}": safe_stat(
                    diameter_values_physical,
                    lambda values: np.percentile(values, 95),
                ),
                f"mean_normal_diameter_{unit}": float(
                    network_row["mean_normal_diameter_px"] * mean_scale
                ),
                f"median_normal_diameter_{unit}": float(
                    network_row["median_normal_diameter_px"] * mean_scale
                ),
                f"p95_normal_diameter_{unit}": float(
                    network_row["p95_normal_diameter_px"] * mean_scale
                ),
                f"mean_curvature_per_{unit}": safe_stat(
                    curvature_values_physical,
                    np.mean,
                ),
                f"median_curvature_per_{unit}": safe_stat(
                    curvature_values_physical,
                    np.median,
                ),
                f"p95_curvature_per_{unit}": safe_stat(
                    curvature_values_physical,
                    lambda values: np.percentile(values, 95),
                ),
                f"networkx_dfs_total_length_{unit}": float(
                    graph_summary.get(f"networkx_dfs_total_length_{unit}", np.nan)
                ),
                f"vascular_graph_total_length_{unit}": float(
                    graph_summary.get(f"sknw_total_edge_length_{unit}")
                    or graph_summary.get(f"networkx_dfs_total_length_{unit}")
                    or (network_row["vascular_graph_total_length_px"] * mean_scale)
                ),
            }
        )
        segment_row_by_id = {
            int(segment_row["segment_id"]): segment_row for segment_row in segment_rows
        }
        for row in thick_vessel_rows:
            thick_segment_ids = parse_id_list(row.get("segment_ids", ""))
            if f"total_length_{unit}" not in row:
                row[f"total_length_{unit}"] = float(
                    sum(
                        segment_row_by_id[segment_id].get(
                            f"length_{unit}",
                            segment_row_by_id[segment_id]["length_px"] * mean_scale,
                        )
                        for segment_id in thick_segment_ids
                        if segment_id in segment_row_by_id
                    )
                )
            if f"longest_path_{unit}" not in row:
                row[f"longest_path_{unit}"] = float(row["longest_path_px"] * mean_scale)
            row[f"median_normal_diameter_{unit}"] = float(
                row["median_normal_diameter_px"] * mean_scale
            )
            row[f"mean_normal_diameter_{unit}"] = float(
                row["mean_normal_diameter_px"] * mean_scale
            )

    cv2.imwrite(
        str(images_dir / "binary_mask_for_skeleton.png"),
        skeleton_binary.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(images_dir / "binary_mask_for_diameter.png"),
        measurement_binary.astype(np.uint8) * 255,
    )
    if centerline_probability is not None:
        cv2.imwrite(
            str(images_dir / "centerline_probability.png"),
            np.clip(centerline_probability * 255.0, 0, 255).astype(np.uint8),
        )
    if predicted_distance_map is not None:
        np.save(images_dir / "predicted_distance_map.npy", predicted_distance_map)
        np.save(images_dir / "diameter_map_from_distance_output.npy", diameter_map)
    if diameter_map_physical is not None:
        np.save(
            images_dir / f"diameter_map_physical_{scale['unit']}.npy",
            diameter_map_physical,
        )
    cv2.imwrite(
        str(images_dir / "skeleton_initial.png"),
        skeleton_initial.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(images_dir / "skeleton_pruned.png"),
        skeleton_clean.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(images_dir / "skeleton_smoothed_display.png"),
        display_skeleton.astype(np.uint8) * 255,
    )
    if args.sigma_y > 0 or args.sigma_x > 0:
        cv2.imwrite(
            str(images_dir / "probability_smoothed_for_skeleton.png"),
            np.clip(
                skeleton_probability * 255.0,
                0,
                255,
            ).astype(np.uint8),
        )
    progress("saved masks and skeletons")

    display_unit = unit if scale["enabled"] else "px"
    diameter_display_key = (
        f"normal_diameter_{unit}" if scale["enabled"] else "normal_diameter_px"
    )
    curvature_display_key = (
        f"curvature_per_{unit}" if scale["enabled"] else "curvature_per_px"
    )
    diameter_colorbar_label = (
        f"Normal diameter ({display_unit})"
        if scale["enabled"]
        else "Normal diameter (pixel)"
    )
    curvature_colorbar_label = (
        f"Curvature (1/{display_unit})"
        if scale["enabled"]
        else "Curvature (1/pixel)"
    )

    save_overlay(
        probability,
        display_skeleton,
        images_dir / "skeleton_overlay.png",
    )
    save_pruning_overlay(
        probability,
        skeleton_clean,
        removed_mask,
        images_dir / "pruning_overlay.png",
    )
    if density_roi_requested:
        save_density_roi_overlay(
            probability,
            measurement_binary_before_density_roi,
            density_roi_visual_mask,
            images_dir / "density_roi_overlay.png",
        )
    save_scalar_map(
        probability,
        point_rows,
        value_key=diameter_display_key,
        colorbar_label=diameter_colorbar_label,
        path=images_dir / "diameter_overlay.png",
    )
    save_combined_curvature_overlay(
        probability,
        point_rows,
        junction_branch_point_rows,
        images_dir / "curvature_overlay.png",
        value_key=curvature_display_key,
        colorbar_label=curvature_colorbar_label,
    )
    progress("saved scalar overlays")
    save_thick_vessel_overlay(
        probability,
        segment_geometries,
        thick_vessel_rows,
        images_dir / "thick_vessels_overlay.png",
        diameter_key=f"median_normal_diameter_{unit}" if scale["enabled"] else "median_normal_diameter_px",
        length_key=f"total_length_{unit}" if scale["enabled"] else "total_length_px",
        unit_label=display_unit,
    )
    progress("saved thick vessel overlay")
    if args.analyze_split_merge:
        save_split_merge_overlay(
            probability,
            segment_geometries,
            split_merge_rows,
            split_merge_geometries,
            images_dir / "split_merge_overlay.png",
        )
        progress("saved split-merge overlay")
    save_cycle_overlay(
        probability,
        cycle_rows,
        cycle_geometries,
        images_dir / "vascular_cycle_overlay.png",
    )
    progress("saved vascular cycle overlay")
    scale = physical_scale_from_args(args)
    unit = scale["unit"] if scale["enabled"] else ""

    point_fields = [
        "segment_id",
        "component_id",
        "point_index",
        "y",
        "x",
        "probability",
        "diameter_px",
        "normal_diameter_px",
        "curvature_per_px",
    ]
    if scale["enabled"]:
        point_fields.extend(
            [f"diameter_{unit}", f"normal_diameter_{unit}", f"curvature_per_{unit}"]
        )
    write_csv(
        csv_dir / "vessel_points.csv",
        point_fields,
        point_rows,
    )

    segment_fields = [
        "segment_id",
        "component_id",
        "start_y",
        "start_x",
        "end_y",
        "end_x",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "n_points",
        "length_px",
        "endpoint_distance_px",
        "tortuosity",
        "mean_probability",
        "mean_diameter_px",
        "median_diameter_px",
        "p95_diameter_px",
        "mean_normal_diameter_px",
        "median_normal_diameter_px",
        "p95_normal_diameter_px",
        "mean_curvature_per_px",
        "median_curvature_per_px",
        "p95_curvature_per_px",
        "max_curvature_per_px",
        "curvature_end_margin_px",
    ]
    if scale["enabled"]:
        segment_fields.extend(
            [
                f"length_{unit}",
                f"endpoint_distance_{unit}",
                f"mean_diameter_{unit}",
                f"median_diameter_{unit}",
                f"p95_diameter_{unit}",
                f"mean_normal_diameter_{unit}",
                f"median_normal_diameter_{unit}",
                f"p95_normal_diameter_{unit}",
                f"mean_curvature_per_{unit}",
                f"median_curvature_per_{unit}",
                f"p95_curvature_per_{unit}",
                f"max_curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "vessel_segments.csv",
        segment_fields,
        segment_rows,
    )
    component_fields = [
        "component_id",
        "segment_count",
        "point_count",
        "length_px",
        "mean_normal_diameter_px",
        "median_normal_diameter_px",
        "p95_normal_diameter_px",
        "mean_diameter_px",
        "median_diameter_px",
        "p95_diameter_px",
        "mean_curvature_per_px",
        "median_curvature_per_px",
        "p95_curvature_per_px",
    ]
    if scale["enabled"]:
        component_fields.extend(
            [
                f"length_{unit}",
                f"mean_normal_diameter_{unit}",
                f"median_normal_diameter_{unit}",
                f"p95_normal_diameter_{unit}",
                f"mean_diameter_{unit}",
                f"median_diameter_{unit}",
                f"p95_diameter_{unit}",
                f"mean_curvature_per_{unit}",
                f"median_curvature_per_{unit}",
                f"p95_curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "vessel_components.csv",
        component_fields,
        component_rows,
    )
    thick_vessel_fields = [
        "thick_vessel_id",
        "segment_count",
        "branch_count",
        "total_length_px",
        "longest_path_px",
        "min_x",
        "median_normal_diameter_px",
        "mean_normal_diameter_px",
        "segment_ids",
        "branch_segment_ids",
    ]
    if scale["enabled"]:
        thick_vessel_fields.extend(
            [
                f"total_length_{unit}",
                f"longest_path_{unit}",
                f"median_normal_diameter_{unit}",
                f"mean_normal_diameter_{unit}",
            ]
        )
    write_csv(
        csv_dir / "thick_vessels.csv",
        thick_vessel_fields,
        thick_vessel_rows,
    )
    split_merge_fields = [
        "split_merge_id",
        "node_a_y",
        "node_a_x",
        "node_b_y",
        "node_b_x",
        "branch_count",
        "branch_segment_ids",
        "branch_lengths_px",
        "virtual_length_px",
        "median_summed_diameter_px",
        "mean_summed_diameter_px",
        "median_area_equivalent_diameter_px",
        "mean_area_equivalent_diameter_px",
        "mean_curvature_per_px",
        "median_curvature_per_px",
        "p95_curvature_per_px",
        "curvature_end_margin_px",
    ]
    if scale["enabled"]:
        split_merge_fields.extend(
            [
                f"virtual_length_{unit}",
                f"median_summed_diameter_{unit}",
                f"mean_summed_diameter_{unit}",
                f"median_area_equivalent_diameter_{unit}",
                f"mean_area_equivalent_diameter_{unit}",
                f"mean_curvature_per_{unit}",
                f"median_curvature_per_{unit}",
                f"p95_curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "split_merge_groups.csv",
        split_merge_fields,
        split_merge_rows,
    )
    split_merge_point_fields = [
        "split_merge_id",
        "point_index",
        "y",
        "x",
        "summed_diameter_px",
        "area_equivalent_diameter_px",
        "curvature_per_px",
    ]
    if scale["enabled"]:
        split_merge_point_fields.extend(
            [
                f"summed_diameter_{unit}",
                f"area_equivalent_diameter_{unit}",
                f"curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "split_merge_points.csv",
        split_merge_point_fields,
        split_merge_point_rows,
    )
    junction_cluster_fields = [
        "junction_id",
        "junction_degree",
        "cluster_pixel_count",
        "junction_center_y",
        "junction_center_x",
        "branch_count",
        "branch_ids",
        "total_branch_length_px",
        "mean_branch_length_px",
        "median_branch_length_px",
        "mean_probability",
        "mean_diameter_px",
        "median_diameter_px",
        "mean_normal_diameter_px",
        "median_normal_diameter_px",
        "mean_curvature_per_px",
        "median_curvature_per_px",
        "p95_curvature_per_px",
        "max_curvature_per_px",
    ]
    if scale["enabled"]:
        junction_cluster_fields.extend(
            [
                f"total_branch_length_{unit}",
                f"mean_branch_length_{unit}",
                f"median_branch_length_{unit}",
                f"mean_diameter_{unit}",
                f"median_diameter_{unit}",
                f"mean_normal_diameter_{unit}",
                f"median_normal_diameter_{unit}",
                f"mean_curvature_per_{unit}",
                f"median_curvature_per_{unit}",
                f"p95_curvature_per_{unit}",
                f"max_curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "junction_cluster_curvatures.csv",
        junction_cluster_fields,
        junction_cluster_rows,
    )
    junction_branch_fields = [
        "junction_id",
        "branch_index",
        "branch_id",
        "junction_degree",
        "junction_center_y",
        "junction_center_x",
        "start_y",
        "start_x",
        "end_y",
        "end_x",
        "end_type",
        "end_junction_id",
        "n_points",
        "length_px",
        "mean_probability",
        "mean_diameter_px",
        "median_diameter_px",
        "mean_normal_diameter_px",
        "median_normal_diameter_px",
        "mean_curvature_per_px",
        "median_curvature_per_px",
        "p95_curvature_per_px",
        "max_curvature_per_px",
        "curvature_end_margin_px",
    ]
    if scale["enabled"]:
        junction_branch_fields.extend(
            [
                f"length_{unit}",
                f"mean_diameter_{unit}",
                f"median_diameter_{unit}",
                f"mean_normal_diameter_{unit}",
                f"median_normal_diameter_{unit}",
                f"mean_curvature_per_{unit}",
                f"median_curvature_per_{unit}",
                f"p95_curvature_per_{unit}",
                f"max_curvature_per_{unit}",
            ]
        )
    write_csv(
        csv_dir / "junction_branch_curvatures.csv",
        junction_branch_fields,
        junction_branch_rows,
    )
    junction_branch_point_fields = [
        "junction_id",
        "branch_id",
        "point_index",
        "y",
        "x",
        "probability",
        "diameter_px",
        "normal_diameter_px",
        "curvature_per_px",
    ]
    if scale["enabled"]:
        junction_branch_point_fields.extend(
            [f"diameter_{unit}", f"normal_diameter_{unit}", f"curvature_per_{unit}"]
        )
    write_csv(
        csv_dir / "junction_branch_points.csv",
        junction_branch_point_fields,
        junction_branch_point_rows,
    )
    public_network_row = public_summary_row(network_row)
    write_csv(
        csv_dir / "network_summary.csv",
        list(public_network_row.keys()),
        [public_network_row],
    )
    public_graph_row = public_summary_row(graph_summary)
    write_csv(
        csv_dir / "graph_summary.csv",
        list(public_graph_row.keys()),
        [public_graph_row],
    )
    write_csv(
        csv_dir / "vascular_cycle_summary.csv",
        ["category", "method", "count", "total_length_px", "note"],
        cycle_summary_rows,
    )
    write_csv(
        csv_dir / "vascular_metrics_summary.csv",
        list(public_metrics_summary_row(metrics_summary).keys()),
        [public_metrics_summary_row(metrics_summary)],
    )
    cycle_fields = [
        "cycle_id",
        "loop_id",
        "cycle_type",
        "area_px2",
        "perimeter_px",
        "length_px",
        "equivalent_diameter_px",
        "perimeter_area_shape_index",
        "centroid_y",
        "centroid_x",
        "min_y",
        "min_x",
        "max_y",
        "max_x",
        "touches_image_boundary",
        "touches_roi_boundary",
    ]
    if args.enable_cycle_orientation_filter:
        cycle_fields.extend(
            [
                "orientation_filter_keep",
                "orientation_sample_points",
                "orientation_junction_like_points",
                "orientation_junction_like_fraction",
                "orientation_max_dominant_count",
            ]
        )
    if scale["enabled"]:
        cycle_fields.extend(
            [
                f"area_{unit}2",
                f"perimeter_{unit}",
                f"length_{unit}",
                f"equivalent_diameter_{unit}",
            ]
        )
    write_csv(
        csv_dir / "vascular_cycles.csv",
        cycle_fields,
        cycle_rows,
    )
    sknw_cycle_fields = [
        "cycle_id",
        "node_count",
        "edge_count",
        "length_px",
        "centroid_y",
        "centroid_x",
        "min_y",
        "min_x",
        "max_y",
        "max_x",
    ]
    if scale["enabled"]:
        sknw_cycle_fields.append(f"length_{unit}")
    write_csv(
        csv_dir / "sknw_cycle_basis_cycles.csv",
        sknw_cycle_fields,
        sknw_cycle_rows,
    )
    terminal_fields = [
        "iteration",
        "endpoint_y",
        "endpoint_x",
        "junction_y",
        "junction_x",
        "length_px",
        "local_diameter_px",
        "branch_probability",
        "parent_probability",
        "probability_ratio",
        "verticality",
        "reason",
    ]
    write_csv(
        csv_dir / "pruned_terminal_branches.csv",
        terminal_fields,
        terminal_records,
    )

    junction_fields = [
        "iteration",
        "start_y",
        "start_x",
        "end_y",
        "end_x",
        "length_px",
        "median_diameter_px",
    ]
    write_csv(
        csv_dir / "pruned_junction_links.csv",
        junction_fields,
        junction_records,
    )
    progress("wrote csv files")

    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"skeleton_source: {network_row['skeleton_source']}")
    print(f"diameter_source: {network_row['diameter_source']}")
    print(f"distance_to_diameter_scale: {args.distance_to_diameter_scale:.6f}")
    print(f"skeleton_threshold: {skeleton_threshold:.6f}")
    print(f"measurement_threshold: {measurement_threshold:.6f}")
    print(f"gaussian_sigma_y: {args.sigma_y:.3f}")
    print(f"gaussian_sigma_x: {args.sigma_x:.3f}")
    print(f"prune_mode: {args.prune_mode}")
    print(f"terminal_branches_removed: {len(terminal_records)}")
    print(f"junction_links_removed: {len(junction_records)}")
    print(f"skeleton_pixels_initial: {int(skeleton_initial.sum())}")
    print(f"skeleton_loop_hole_size: {args.skeleton_loop_hole_size}")
    print(f"skeleton_loop_hole_pixels_filled: {skeleton_loop_hole_pixels_filled}")
    print(f"skeleton_pixels_final: {int(skeleton_clean.sum())}")
    print(f"endpoints: {endpoint_count}")
    print(f"junction_pixels: {junction_pixel_count}")
    print(f"measured_segments: {len(segment_rows)}")
    print(f"total_length_px: {total_length_px:.3f}")
    print(
        "median_diameter_px: "
        f"{network_row['median_diameter_px']:.3f}"
    )
    print(
        "median_normal_diameter_px: "
        f"{network_row['median_normal_diameter_px']:.3f}"
    )
    print(
        "median_curvature_per_px: "
        f"{network_row['median_curvature_per_px']:.6f}"
    )
    print(
        "thick_vessel_threshold_px: "
        f"{network_row['thick_vessel_threshold_px']:.3f}"
    )
    print(
        "thick_vessel_left_width_px: "
        f"{network_row['thick_vessel_left_width_px']:.3f}"
    )
    print(f"thick_vessel_count: {network_row['thick_vessel_count']}")
    print(
        "thick_vessel_branch_count: "
        f"{network_row['thick_vessel_branch_count']}"
    )
    print(
        "thick_vessel_max_path_px: "
        f"{network_row['thick_vessel_max_path_px']:.3f}"
    )
    print(
        "vascular_cycle_count: "
        f"{network_row['vascular_cycle_count']} "
        f"({network_row['vascular_cycle_count_method']})"
    )
    print(
        "vascular_graph_total_length_px: "
        f"{network_row['vascular_graph_total_length_px']:.3f}"
    )
    cycle_filter_unit = network_row["cycle_length_filter_unit"]
    cycle_filter_min_key = (
        f"cycle_length_filter_min_{cycle_filter_unit}"
        if cycle_filter_unit != "px"
        else "cycle_length_filter_min_px"
    )
    cycle_filter_max_key = (
        f"cycle_length_filter_max_{cycle_filter_unit}"
        if cycle_filter_unit != "px"
        else "cycle_length_filter_max_px"
    )
    cycle_max_text = (
        f"{network_row[cycle_filter_max_key]:.3f}"
        if np.isfinite(network_row[cycle_filter_max_key])
        else "unlimited"
    )
    print(
        f"cycle_length_filter_{cycle_filter_unit}: "
        f"{network_row[cycle_filter_min_key]:.3f} to {cycle_max_text}"
    )
    print(
        "direct_parallel_split_merge_count: "
        f"{network_row['direct_parallel_split_merge_count']}"
    )
    print(
        "vessel_area_density: "
        f"{metrics_summary['vessel_area_density']:.6f}"
    )
    print(
        "skeleton_length_density_per_px: "
        f"{metrics_summary['skeleton_length_density_per_px']:.8f}"
    )
    print(f"junction_cluster_count: {len(junction_cluster_rows)}")
    print(f"junction_branch_count: {len(junction_branch_rows)}")
    print(f"sknw_available: {network_row['sknw_available']}")
    print(f"skeleton_min_hole_size: {skeleton_hole_size}")
    print(f"measurement_min_hole_size: {measurement_hole_size}")
    if args.pixel_size_um is not None:
        print(
            "total_length_um: "
            f"{network_row['total_length_um']:.3f}"
        )
        print(
            "median_diameter_um: "
            f"{network_row['median_diameter_um']:.3f}"
        )
        print(
            "median_normal_diameter_um: "
            f"{network_row['median_normal_diameter_um']:.3f}"
        )
        print(
            "median_curvature_per_um: "
            f"{network_row['median_curvature_per_um']:.6f}"
        )
        print(
            "thick_vessel_max_path_um: "
            f"{network_row['thick_vessel_max_path_um']:.3f}"
        )


if __name__ == "__main__":
    main()
