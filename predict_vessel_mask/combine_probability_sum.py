import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from predict import (
    ROOT,
    build_model_input,
    calculate_frangi_response,
    clean_structure_mask,
    create_overlay,
    load_model,
    normalize_image,
    predict as predict_unet,
    resolve_path,
    select_device,
)
from predict_multicontrast import (
    build_brightness_mask,
    build_intensity_window_variants,
    build_variants,
    combine_probabilities,
    postprocess_probability,
    read_image_as_grayscale,
)
from predict_vessmap_batch import (
    IMAGE_EXTENSIONS,
    load_vessmap_model,
    predict_multicontrast_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine two sets of probability maps by summing probabilities "
            "and keeping pixels whose sum is above a threshold."
        )
    )
    parser.add_argument(
        "--prob-dir-a",
        help="First probability-map directory, searched recursively.",
    )
    parser.add_argument(
        "--prob-dir-b",
        help="Second probability-map directory, searched recursively.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Input image to predict directly. Can be passed more than once.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Input image directory to predict directly. Can be passed more than once.",
    )
    parser.add_argument(
        "--best-model",
        default=str(ROOT.parent / "models" / "unet_2ch_input.pt"),
        help="Path to the regular U-Net checkpoint.",
    )
    parser.add_argument(
        "--vessmap-model",
        default=str(ROOT.parent / "models" / "vessmap.pt"),
        help="Path to the VessMAP checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "predictions" / "probability_sum_consensus"),
        help="Directory for consensus outputs.",
    )
    parser.add_argument(
        "--sum-threshold",
        type=float,
        default=0.8,
        help="Keep pixels where prob_a + prob_b is at least this value.",
    )
    parser.add_argument(
        "--min-prob-a",
        type=float,
        default=0.0,
        help="Also require the first probability map/model to be at least this value.",
    )
    parser.add_argument(
        "--min-prob-b",
        type=float,
        default=0.0,
        help="Also require the second probability map/model to be at least this value.",
    )
    parser.add_argument(
        "--suffix-a",
        default="_multicontrast_probability",
        help="Suffix removed from probability file stems in prob-dir-a.",
    )
    parser.add_argument(
        "--suffix-b",
        default="_multicontrast_probability",
        help="Suffix removed from probability file stems in prob-dir-b.",
    )
    parser.add_argument(
        "--min-structure-area",
        type=int,
        default=48,
        help="Remove connected mask components smaller than this pixel area.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Odd kernel size used for opening/closing postprocessing.",
    )
    parser.add_argument(
        "--resize-b-to-a",
        action="store_true",
        help="Resize B probability maps to A shape if shapes differ.",
    )
    parser.add_argument("--sigma-min", type=int, default=2)
    parser.add_argument("--sigma-max", type=int, default=8)
    parser.add_argument(
        "--frangi-input-weight",
        type=float,
        default=0.35,
        help="Blend weight for Frangi response in 1-channel best_model inputs.",
    )
    parser.add_argument(
        "--post-frangi-weight",
        type=float,
        default=0.15,
        help="Blend this amount of Frangi response into each probability map.",
    )
    parser.add_argument(
        "--combine",
        choices=("max", "mean", "weighted-mean"),
        default="max",
        help="How to combine multicontrast variants inside each model.",
    )
    parser.add_argument(
        "--variant-min-probability",
        type=float,
        default=0.0,
        help="Set variant probabilities below this value to zero before combining.",
    )
    parser.add_argument(
        "--min-vessel-brightness",
        type=float,
        default=0.0,
        help="Only keep predictions where the normalized original image is this bright.",
    )
    parser.add_argument("--brightness-mask-dilate", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--single-contrast",
        action="store_true",
        help="Use one original+Frangi pass per model instead of multicontrast variants.",
    )
    parser.add_argument(
        "--save-model-probabilities",
        action="store_true",
        help="Save the two individual model probability maps as NPY/PNG.",
    )
    parser.add_argument(
        "--normalize-vessmap-probability",
        action="store_true",
        help=(
            "Robustly normalize the VessMAP probability map to 0-1 after "
            "prediction by clipping outlier percentiles."
        ),
    )
    parser.add_argument(
        "--vessmap-normalize-low-percentile",
        type=float,
        default=1.0,
        help="Lower percentile used by --normalize-vessmap-probability.",
    )
    parser.add_argument(
        "--vessmap-normalize-high-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used by --normalize-vessmap-probability.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip direct image inputs whose consensus outputs already exist.",
    )
    parser.add_argument("--intensity-window-ensemble", action="store_true")
    parser.add_argument("--intensity-window-high", type=int, default=255)
    parser.add_argument("--intensity-window-low", type=int, default=95)
    parser.add_argument("--intensity-window-step", type=int, default=20)
    parser.add_argument("--intensity-window-weight", type=float, default=0.75)
    parser.add_argument("--intensity-window-mask-probability", action="store_true")
    return parser.parse_args()


def strip_suffix(value: str, suffix: str) -> str:
    return value[: -len(suffix)] if suffix and value.endswith(suffix) else value


def probability_key(path: Path, root: Path, suffix: str) -> Path:
    relative = path.relative_to(root)
    return relative.with_name(strip_suffix(relative.stem, suffix))


def collect_probability_maps(root: Path, suffix: str) -> dict[Path, Path]:
    maps: dict[Path, Path] = {}
    for path in sorted(root.rglob("*_probability.npy")):
        key = probability_key(path, root, suffix)
        if key in maps:
            raise ValueError(f"Duplicate probability key {key}: {maps[key]} and {path}")
        maps[key] = path
    return maps


def save_uint16(path: Path, array: np.ndarray, scale: float) -> None:
    uint16 = np.clip(np.rint(array * scale), 0, 65535).astype(np.uint16)
    Image.fromarray(uint16).save(path)


def image_paths(input_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Skipping missing directory: {input_dir}")
            continue
        paths.extend(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return sorted(paths)


def collect_image_paths(image_args: list[str], input_dirs: list[Path]) -> list[Path]:
    paths = [resolve_path(path) for path in image_args]
    paths.extend(image_paths(input_dirs))
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def robust_normalize_01(
    array: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    low, high = np.percentile(array, [low_percentile, high_percentile])
    if high <= low:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if maximum - minimum < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        return ((array - minimum) / (maximum - minimum)).astype(np.float32)
    clipped = np.clip(array, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)


def normalize_vessmap_probability_if_requested(
    probability: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if not args.normalize_vessmap_probability:
        return probability.astype(np.float32)
    return robust_normalize_01(
        probability.astype(np.float32),
        low_percentile=args.vessmap_normalize_low_percentile,
        high_percentile=args.vessmap_normalize_high_percentile,
    )


def output_location(output_root: Path, image_path: Path, frame: int) -> tuple[Path, str]:
    try:
        relative = image_path.relative_to(ROOT / "raw")
        output_dir = output_root / relative.parent
    except ValueError:
        output_dir = output_root / image_path.parent.name
    frame_suffix = f"_frame{frame:04d}" if frame > 0 else ""
    return output_dir, f"{image_path.stem}{frame_suffix}"


def predict_standard_multicontrast(
    model,
    metadata: dict,
    raw_image: np.ndarray,
    image: np.ndarray,
    args: argparse.Namespace,
    device,
) -> np.ndarray:
    variants = build_variants(image)
    window_variants = []
    if args.intensity_window_ensemble:
        window_variants = build_intensity_window_variants(
            raw_image,
            high=args.intensity_window_high,
            low=args.intensity_window_low,
            step=args.intensity_window_step,
            weight=args.intensity_window_weight,
        )
        variants.extend((name, variant, weight) for name, variant, weight, _ in window_variants)

    window_masks = {name: mask for name, _, _, mask in window_variants}
    probability_items = []
    for name, variant, weight in variants:
        frangi_response = calculate_frangi_response(
            variant,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        model_input = build_model_input(
            image=variant,
            frangi_response=frangi_response,
            in_channels=metadata["in_channels"],
            frangi_input_weight=args.frangi_input_weight,
        )
        raw_probability = predict_unet(model, model_input, device)
        probability = postprocess_probability(
            raw_probability,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            frangi_weight=args.post_frangi_weight,
        )
        if args.intensity_window_mask_probability and name in window_masks:
            probability = np.where(window_masks[name], probability, 0.0).astype(np.float32)
        probability_items.append((name, probability, weight))

    probability = combine_probabilities(
        probability_items,
        method=args.combine,
        variant_min_probability=args.variant_min_probability,
    )
    brightness_mask = build_brightness_mask(
        image,
        min_brightness=args.min_vessel_brightness,
        dilation_pixels=args.brightness_mask_dilate,
    )
    if brightness_mask is not None:
        probability = np.where(brightness_mask, probability, 0.0).astype(np.float32)
    return probability.astype(np.float32)


def predict_standard_single(
    model,
    metadata: dict,
    image: np.ndarray,
    args: argparse.Namespace,
    device,
) -> np.ndarray:
    frangi_response = calculate_frangi_response(
        image,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    model_input = build_model_input(
        image=image,
        frangi_response=frangi_response,
        in_channels=metadata["in_channels"],
        frangi_input_weight=args.frangi_input_weight,
    )
    raw_probability = predict_unet(model, model_input, device)
    return postprocess_probability(
        raw_probability,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        frangi_weight=args.post_frangi_weight,
    )


def predict_vessmap_single(model, image: np.ndarray, args: argparse.Namespace, device) -> np.ndarray:
    from predict_vessmap_batch import predict_probability

    frangi_response = calculate_frangi_response(
        image,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    model_input = np.stack([image, frangi_response], axis=0).astype(np.float32)
    raw_probability = predict_probability(model, model_input, device)
    return postprocess_probability(
        raw_probability,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        frangi_weight=args.post_frangi_weight,
    )


def save_consensus_outputs(
    output_dir: Path,
    stem: str,
    probability_best: np.ndarray,
    probability_vessmap: np.ndarray,
    args: argparse.Namespace,
    image: np.ndarray | None = None,
) -> Path:
    probability_best = np.clip(probability_best, 0.0, 1.0)
    probability_vessmap = np.clip(probability_vessmap, 0.0, 1.0)
    probability_sum = probability_best + probability_vessmap
    raw_mask = (
        (probability_best >= args.min_prob_a)
        & (probability_vessmap >= args.min_prob_b)
        & (probability_sum >= args.sum_threshold)
    )
    mask = clean_structure_mask(
        raw_mask,
        min_area=args.min_structure_area,
        kernel_size=args.morph_kernel,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}_probability_sum.npy", probability_sum.astype(np.float32))
    save_uint16(output_dir / f"{stem}_probability_sum.png", probability_sum, 65535.0 / 2.0)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{stem}_sum_mask.png")
    if args.save_model_probabilities:
        np.save(output_dir / f"{stem}_best_model_probability.npy", probability_best.astype(np.float32))
        np.save(output_dir / f"{stem}_vessmap_probability.npy", probability_vessmap.astype(np.float32))
        save_uint16(output_dir / f"{stem}_best_model_probability.png", probability_best, 65535.0)
        save_uint16(output_dir / f"{stem}_vessmap_probability.png", probability_vessmap, 65535.0)
    if image is not None:
        overlay = np.clip(create_overlay(image, np.clip(probability_sum / 2.0, 0.0, 1.0), mask), 0.0, 1.0)
        Image.fromarray(np.clip(np.rint(overlay * 255.0), 0, 255).astype(np.uint8)).save(
            output_dir / f"{stem}_sum_overlay.png"
        )
    return output_dir / f"{stem}_sum_mask.png", {
        "raw_pixels": int(np.count_nonzero(raw_mask)),
        "cleaned_pixels": int(np.count_nonzero(mask)),
        "total_pixels": int(mask.size),
        "best_ge_min": int(np.count_nonzero(probability_best >= args.min_prob_a)),
        "vessmap_ge_min": int(np.count_nonzero(probability_vessmap >= args.min_prob_b)),
        "sum_ge_min": int(np.count_nonzero(probability_sum >= args.sum_threshold)),
    }


def run_probability_map_mode(args: argparse.Namespace) -> None:
    if not args.prob_dir_a or not args.prob_dir_b:
        raise ValueError("Pass --prob-dir-a and --prob-dir-b, or pass --image/--input-dir.")

    prob_dir_a = resolve_path(args.prob_dir_a)
    prob_dir_b = resolve_path(args.prob_dir_b)
    output_root = resolve_path(args.output_dir)

    maps_a = collect_probability_maps(prob_dir_a, args.suffix_a)
    maps_b = collect_probability_maps(prob_dir_b, args.suffix_b)
    common_keys = sorted(set(maps_a) & set(maps_b))
    if not common_keys:
        raise ValueError(
            "No matching probability maps found. Check --suffix-a/--suffix-b "
            "and whether both directories use the same relative filenames."
        )

    missing_from_b = sorted(set(maps_a) - set(maps_b))
    missing_from_a = sorted(set(maps_b) - set(maps_a))
    print(
        f"Found {len(common_keys)} matching pairs "
        f"({len(missing_from_b)} only in A, {len(missing_from_a)} only in B)."
    )
    print(
        "Rule: "
        f"prob_a >= {args.min_prob_a:g}, "
        f"prob_b >= {args.min_prob_b:g}, "
        f"prob_a + prob_b >= {args.sum_threshold:g}"
    )

    for index, key in enumerate(common_keys, 1):
        probability_a = np.load(maps_a[key]).astype(np.float32)
        probability_b = np.load(maps_b[key]).astype(np.float32)
        if probability_a.shape != probability_b.shape:
            if not args.resize_b_to_a:
                raise ValueError(
                    f"Shape mismatch for {key}: {probability_a.shape} vs "
                    f"{probability_b.shape}. Pass --resize-b-to-a to resize B."
                )
            probability_b = cv2.resize(
                probability_b,
                (probability_a.shape[1], probability_a.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        mask_path, stats = save_consensus_outputs(
            output_root / key.parent,
            key.name,
            probability_a,
            probability_b,
            args,
        )
        print(
            f"[{index}/{len(common_keys)}] {key} -> {mask_path} "
            f"(raw={stats['raw_pixels']}, cleaned={stats['cleaned_pixels']})"
        )


def run_image_mode(args: argparse.Namespace) -> None:
    input_dirs = [resolve_path(path) for path in args.input_dir]
    paths = collect_image_paths(args.image, input_dirs)
    if not paths:
        raise ValueError("No input images found.")

    device = select_device(args.device)
    best_model, best_metadata = load_model(resolve_path(args.best_model), device)
    vessmap_model, vessmap_metadata = load_vessmap_model(
        resolve_path(args.vessmap_model),
        device,
    )
    output_root = resolve_path(args.output_dir)
    print(
        f"Device: {device}, inputs={len(paths)}, "
        f"best_model_channels={best_metadata['in_channels']}, "
        f"vessmap_channels={vessmap_metadata['in_channels']}"
    )
    print(
        "Rule: "
        f"prob_best_model >= {args.min_prob_a:g}, "
        f"prob_vessmap >= {args.min_prob_b:g}, "
        f"sum >= {args.sum_threshold:g}"
    )
    print(
        "Mode: "
        + ("single contrast" if args.single_contrast else "multicontrast")
        + f", combine={args.combine}, sigmas={args.sigma_min}-{args.sigma_max}"
    )

    for index, image_path in enumerate(paths, 1):
        output_dir, stem = output_location(output_root, image_path, args.frame)
        expected = output_dir / f"{stem}_sum_mask.png"
        if args.skip_existing and expected.is_file():
            print(f"[{index}/{len(paths)}] Skipping existing output: {image_path}")
            continue

        raw_image = read_image_as_grayscale(image_path, frame=args.frame)
        image = normalize_image(raw_image)
        if args.single_contrast:
            probability_best = predict_standard_single(
                best_model,
                best_metadata,
                image,
                args,
                device,
            )
            probability_vessmap = predict_vessmap_single(
                vessmap_model,
                image,
                args,
                device,
            )
            probability_vessmap = normalize_vessmap_probability_if_requested(
                probability_vessmap,
                args,
            )
        else:
            probability_best = predict_standard_multicontrast(
                best_model,
                best_metadata,
                raw_image,
                image,
                args,
                device,
            )
            probability_vessmap, _, _ = predict_multicontrast_probability(
                vessmap_model,
                raw_image,
                image,
                args,
                device,
            )
            probability_vessmap = normalize_vessmap_probability_if_requested(
                probability_vessmap,
                args,
            )

        if probability_best.shape != probability_vessmap.shape:
            probability_vessmap = cv2.resize(
                probability_vessmap,
                (probability_best.shape[1], probability_best.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        mask_path, stats = save_consensus_outputs(
            output_dir,
            stem,
            probability_best,
            probability_vessmap,
            args,
            image=image,
        )
        print(
            f"[{index}/{len(paths)}] {image_path} -> {mask_path} "
            f"(best>=min: {stats['best_ge_min']}/{stats['total_pixels']}, "
            f"vessmap>=min: {stats['vessmap_ge_min']}/{stats['total_pixels']}, "
            f"sum>=min: {stats['sum_ge_min']}/{stats['total_pixels']}, "
            f"raw: {stats['raw_pixels']}, cleaned: {stats['cleaned_pixels']})"
        )


def main() -> None:
    args = parse_args()
    if args.sum_threshold <= 0:
        raise ValueError("--sum-threshold must be positive.")
    if not 0.0 <= args.min_prob_a <= 1.0:
        raise ValueError("--min-prob-a must be in [0, 1].")
    if not 0.0 <= args.min_prob_b <= 1.0:
        raise ValueError("--min-prob-b must be in [0, 1].")
    if not 0.0 <= args.variant_min_probability < 1.0:
        raise ValueError("--variant-min-probability must be in [0, 1).")
    if not 0.0 <= args.min_vessel_brightness < 1.0:
        raise ValueError("--min-vessel-brightness must be in [0, 1).")
    if args.brightness_mask_dilate < 0:
        raise ValueError("--brightness-mask-dilate must be zero or positive.")
    if not 0.0 <= args.vessmap_normalize_low_percentile < args.vessmap_normalize_high_percentile <= 100.0:
        raise ValueError(
            "--vessmap-normalize-low-percentile/high-percentile must satisfy "
            "0 <= low < high <= 100."
        )

    if args.image or args.input_dir:
        run_image_mode(args)
    else:
        run_probability_map_mode(args)


if __name__ == "__main__":
    main()
