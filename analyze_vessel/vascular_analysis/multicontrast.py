from .common import (
    Image,
    Path,
    argparse,
    cv2,
    distance_transform_edt,
    np,
    remove_small_objects,
    skeletonize,
)

from .io_utils import multicontrast_save_probability_png

from .visualization import write_csv

def multicontrast_axial_angle_difference(theta_a: float, theta_b: float) -> float:
    diff = abs(float(theta_a) - float(theta_b)) % np.pi
    return float(min(diff, np.pi - diff))

def multicontrast_max_axial_angle_spread(angles: np.ndarray) -> float:
    finite = angles[np.isfinite(angles)]
    if finite.size < 2:
        return 0.0
    max_diff = 0.0
    for index, angle in enumerate(finite[:-1]):
        for other in finite[index + 1 :]:
            max_diff = max(max_diff, multicontrast_axial_angle_difference(angle, other))
    return float(max_diff)

def multicontrast_estimate_skeleton_orientation(
    skeleton: np.ndarray,
    window_radius: int,
) -> np.ndarray:
    orientation = np.full(skeleton.shape, np.nan, dtype=np.float32)
    height, width = skeleton.shape
    for y, x in np.argwhere(skeleton):
        y0 = max(0, int(y) - window_radius)
        y1 = min(height, int(y) + window_radius + 1)
        x0 = max(0, int(x) - window_radius)
        x1 = min(width, int(x) + window_radius + 1)
        neighbors = np.argwhere(skeleton[y0:y1, x0:x1])
        if len(neighbors) < 2:
            continue
        neighbors = neighbors.astype(np.float32)
        neighbors[:, 0] += y0
        neighbors[:, 1] += x0
        centered = neighbors - np.mean(neighbors, axis=0, keepdims=True)
        covariance = centered.T @ centered
        values, vectors = np.linalg.eigh(covariance)
        tangent = vectors[:, int(np.argmax(values))]
        orientation[int(y), int(x)] = np.arctan2(float(tangent[0]), float(tangent[1]))
    return orientation

def multicontrast_nearest_skeleton_orientation(
    skeleton: np.ndarray,
    orientation: np.ndarray,
    match_radius: float,
) -> np.ndarray:
    if not np.any(skeleton) or match_radius <= 0:
        return np.full(skeleton.shape, np.nan, dtype=np.float32)
    distances, indices = distance_transform_edt(
        ~skeleton,
        return_indices=True,
    )
    nearest = orientation[indices[0], indices[1]]
    return np.where(distances <= match_radius, nearest, np.nan).astype(np.float32)

def multicontrast_consensus_probability(
    variant_rows: list[dict],
    method: str,
    support_threshold: float,
    min_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack([row["probability"] for row in variant_rows], axis=0)
    weights = np.asarray([row["weight"] for row in variant_rows], dtype=np.float32)
    if method == "median":
        consensus = np.median(stack, axis=0).astype(np.float32)
    elif method == "mean":
        consensus = np.mean(stack, axis=0).astype(np.float32)
    elif method == "weighted-mean":
        consensus = np.average(stack, axis=0, weights=weights).astype(np.float32)
    else:
        raise ValueError(f"Unsupported multicontrast consensus method: {method}")
    support_count = np.sum(stack >= support_threshold, axis=0).astype(np.uint8)
    support_mask = support_count >= min_support
    consensus = np.where(support_mask, consensus, 0.0).astype(np.float32)
    return consensus, support_count, support_mask

def multicontrast_orientation_filter(
    variant_rows: list[dict],
    support_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if args.disable_multicontrast_orientation_filter:
        empty = np.zeros_like(support_mask, dtype=bool)
        return support_mask.astype(bool), empty, empty, {
            "orientation_filter_enabled": False,
            "orientation_evaluated_px": 0,
            "orientation_ambiguous_px": 0,
            "orientation_ambiguous_dilated_px": 0,
            "orientation_excluded_support_px": 0,
            "max_angle_diff_deg": float(args.multicontrast_max_angle_diff_deg),
            "orientation_match_radius_px": float(args.multicontrast_orientation_match_radius),
        }

    orientation_maps = []
    skeleton_rows = []
    for row in variant_rows:
        mask = row["probability"] >= args.multicontrast_support_threshold
        mask = remove_small_objects(
            mask,
            max_size=max(0, int(args.min_object_size) - 1),
            connectivity=2,
        )
        skeleton = skeletonize(mask)
        orientation = multicontrast_estimate_skeleton_orientation(
            skeleton,
            window_radius=args.multicontrast_orientation_window_radius,
        )
        nearest = multicontrast_nearest_skeleton_orientation(
            skeleton,
            orientation,
            match_radius=args.multicontrast_orientation_match_radius,
        )
        orientation_maps.append(nearest)
        skeleton_rows.append(
            {
                "name": row["name"],
                "orientation_skeleton_pixels": int(skeleton.sum()),
            }
        )

    orientation_stack = np.stack(orientation_maps, axis=0)
    valid_count = np.sum(np.isfinite(orientation_stack), axis=0)
    candidate_mask = (
        support_mask
        & evaluation_mask
        & (valid_count >= args.multicontrast_min_support)
    )
    ambiguous = np.zeros_like(support_mask, dtype=bool)
    max_allowed = np.deg2rad(args.multicontrast_max_angle_diff_deg)
    for y, x in np.argwhere(candidate_mask):
        if multicontrast_max_axial_angle_spread(orientation_stack[:, y, x]) > max_allowed:
            ambiguous[y, x] = True

    if args.multicontrast_orientation_dilate_radius > 0 and np.any(ambiguous):
        radius = args.multicontrast_orientation_dilate_radius
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        ambiguous_dilated = cv2.dilate(
            ambiguous.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
    else:
        ambiguous_dilated = ambiguous

    orientation_mask = support_mask & ~(ambiguous_dilated & evaluation_mask)
    summary = {
        "orientation_filter_enabled": True,
        "orientation_evaluated_px": int(candidate_mask.sum()),
        "orientation_ambiguous_px": int(ambiguous.sum()),
        "orientation_ambiguous_dilated_px": int(ambiguous_dilated.sum()),
        "orientation_excluded_support_px": int(
            (support_mask & ambiguous_dilated & evaluation_mask).sum()
        ),
        "max_angle_diff_deg": float(args.multicontrast_max_angle_diff_deg),
        "orientation_match_radius_px": float(args.multicontrast_orientation_match_radius),
        "orientation_window_radius_px": int(args.multicontrast_orientation_window_radius),
        "orientation_dilate_radius_px": int(args.multicontrast_orientation_dilate_radius),
        "orientation_variant_skeletons": ";".join(
            f"{row['name']}:{row['orientation_skeleton_pixels']}"
            for row in skeleton_rows
        ),
    }
    return orientation_mask, ambiguous, ambiguous_dilated, summary

def make_multicontrast_input(
    args: argparse.Namespace,
    output_dir: Path,
    progress,
) -> tuple[Path, dict]:
    from predict_vessel_mask.predict import (
        DEFAULT_MODEL,
        build_model_input,
        calculate_frangi_response,
        create_overlay,
        load_model,
        normalize_image,
        postprocess_probability,
        predict_outputs,
        select_device,
    )
    from predict_vessel_mask.predict_multicontrast import build_variants, read_image_as_grayscale

    image_path = args.multicontrast_image
    model_path = args.multicontrast_model or DEFAULT_MODEL
    prediction_dir = output_dir / "multicontrast_prediction"
    variant_dir = prediction_dir / f"{image_path.stem}_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    raw_image = read_image_as_grayscale(image_path, frame=args.multicontrast_frame)
    image = normalize_image(raw_image)
    device = select_device(args.multicontrast_device)
    model, metadata = load_model(model_path, device)
    progress("loaded multicontrast model and image")

    variant_rows = []
    centerline_variant_rows = []
    distance_variant_maps = []
    for name, variant, weight in build_variants(image):
        frangi_response = calculate_frangi_response(
            variant,
            sigma_min=args.multicontrast_sigma_min,
            sigma_max=args.multicontrast_sigma_max,
        )
        model_input = build_model_input(
            image=variant,
            frangi_response=frangi_response,
            in_channels=metadata["in_channels"],
            frangi_input_weight=args.multicontrast_frangi_input_weight,
        )
        raw_outputs = predict_outputs(model, model_input, device)
        raw_probability = raw_outputs[0]
        probability = postprocess_probability(
            raw_probability,
            sigma_min=args.multicontrast_sigma_min,
            sigma_max=args.multicontrast_sigma_max,
            frangi_weight=args.multicontrast_post_frangi_weight,
        )
        probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
        multicontrast_save_probability_png(variant_dir / f"{name}_probability.png", probability)
        np.save(variant_dir / f"{name}_probability.npy", probability)
        multicontrast_save_probability_png(variant_dir / f"{name}_input.png", variant)
        variant_rows.append(
            {
                "name": name,
                "weight": float(weight),
                "probability": probability,
            }
        )
        if raw_outputs.shape[0] >= 3:
            centerline_probability = np.clip(raw_outputs[1].astype(np.float32), 0.0, 1.0)
            distance_probability = np.clip(raw_outputs[2].astype(np.float32), 0.0, 1.0)
            multicontrast_save_probability_png(
                variant_dir / f"{name}_centerline_probability.png",
                centerline_probability,
            )
            np.save(variant_dir / f"{name}_centerline_probability.npy", centerline_probability)
            np.save(variant_dir / f"{name}_distance_map.npy", distance_probability)
            centerline_variant_rows.append(
                {
                    "name": name,
                    "weight": float(weight),
                    "probability": centerline_probability,
                }
            )
            distance_variant_maps.append((distance_probability, float(weight)))
    progress(f"predicted {len(variant_rows)} contrast variants")

    consensus, support_count, support_mask = multicontrast_consensus_probability(
        variant_rows,
        method=args.multicontrast_consensus,
        support_threshold=args.multicontrast_support_threshold,
        min_support=args.multicontrast_min_support,
    )
    orientation_mask, ambiguous, ambiguous_dilated, orientation_summary = (
        multicontrast_orientation_filter(
            variant_rows,
            support_mask,
            consensus >= args.multicontrast_final_threshold,
            args,
        )
    )
    consensus = np.where(orientation_mask, consensus, 0.0).astype(np.float32)
    binary_mask = consensus >= args.multicontrast_final_threshold

    stem = f"{image_path.stem}_multicontrast_consensus"
    probability_path = prediction_dir / f"{stem}_probability.npy"
    centerline_path = None
    distance_path = None
    np.save(probability_path, consensus)
    multicontrast_save_probability_png(prediction_dir / f"{stem}_probability.png", consensus)
    if centerline_variant_rows and distance_variant_maps:
        centerline_consensus, _, _ = multicontrast_consensus_probability(
            centerline_variant_rows,
            method=args.multicontrast_consensus,
            support_threshold=args.multicontrast_support_threshold,
            min_support=max(1, min(args.multicontrast_min_support, len(centerline_variant_rows))),
        )
        centerline_consensus = np.where(orientation_mask, centerline_consensus, 0.0).astype(np.float32)
        distance_stack = np.stack([item[0] for item in distance_variant_maps], axis=0)
        distance_weights = np.asarray([item[1] for item in distance_variant_maps], dtype=np.float32)
        if np.sum(distance_weights) <= 0:
            distance_consensus = np.mean(distance_stack, axis=0).astype(np.float32)
        else:
            distance_consensus = np.average(
                distance_stack,
                axis=0,
                weights=distance_weights,
            ).astype(np.float32)
        distance_consensus = np.where(orientation_mask, distance_consensus, 0.0).astype(np.float32)
        centerline_path = prediction_dir / f"{stem}_centerline_probability.npy"
        distance_path = prediction_dir / f"{stem}_distance_map.npy"
        np.save(centerline_path, centerline_consensus)
        np.save(distance_path, distance_consensus)
        multicontrast_save_probability_png(
            prediction_dir / f"{stem}_centerline_probability.png",
            centerline_consensus,
        )
        args.centerline_input = centerline_path
        args.distance_input = distance_path
    Image.fromarray((binary_mask.astype(np.uint8) * 255)).save(
        prediction_dir / f"{stem}_mask.png"
    )
    Image.fromarray(support_count.astype(np.uint8)).save(
        prediction_dir / f"{stem}_support_count.png"
    )
    Image.fromarray((support_mask.astype(np.uint8) * 255)).save(
        prediction_dir / f"{stem}_support_mask.png"
    )
    Image.fromarray((orientation_mask.astype(np.uint8) * 255)).save(
        prediction_dir / f"{stem}_orientation_consistent_mask.png"
    )
    Image.fromarray((ambiguous.astype(np.uint8) * 255)).save(
        prediction_dir / f"{stem}_ambiguous_orientation.png"
    )
    Image.fromarray((ambiguous_dilated.astype(np.uint8) * 255)).save(
        prediction_dir / f"{stem}_ambiguous_orientation_dilated.png"
    )
    overlay = create_overlay(image, consensus, binary_mask)
    Image.fromarray(np.clip(np.rint(overlay * 255.0), 0, 255).astype(np.uint8)).save(
        prediction_dir / f"{stem}_overlay.png"
    )

    summary = {
        "multicontrast_image": str(image_path),
        "multicontrast_model": str(model_path),
        "multicontrast_device": str(device),
        "multicontrast_base_channels": metadata["base_channels"],
        "multicontrast_in_channels": metadata["in_channels"],
        "multicontrast_out_channels": metadata.get("out_channels", 1),
        "multicontrast_epoch": metadata.get("epoch"),
        "multicontrast_consensus": args.multicontrast_consensus,
        "multicontrast_support_threshold": float(args.multicontrast_support_threshold),
        "multicontrast_min_support": int(args.multicontrast_min_support),
        "multicontrast_final_threshold": float(args.multicontrast_final_threshold),
        "multicontrast_probability_npy": str(probability_path),
        "multicontrast_centerline_npy": str(centerline_path) if centerline_path else "",
        "multicontrast_distance_npy": str(distance_path) if distance_path else "",
        "multicontrast_final_mask_area_px": int(binary_mask.sum()),
        "multicontrast_support_mask_area_px": int(support_mask.sum()),
        "multicontrast_orientation_consistent_mask_area_px": int(orientation_mask.sum()),
        "multicontrast_variant_names": ";".join(row["name"] for row in variant_rows),
        "multicontrast_counting_policy": (
            "Final branch/endpoint/cycle counts use this one consensus graph; "
            "per-contrast predictions are diagnostic and must not be summed."
        ),
        **orientation_summary,
    }
    write_csv(
        output_dir / "csv" / "multicontrast_prediction_summary.csv",
        list(summary.keys()),
        [summary],
    )
    progress("built multicontrast consensus input")
    return probability_path, summary
