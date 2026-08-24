import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from predict import (
    DEFAULT_MODEL,
    ROOT,
    build_model_input,
    calculate_frangi_response,
    calculate_metrics,
    clean_structure_mask,
    create_overlay,
    load_model,
    normalize_image,
    postprocess_probability,
    predict as predict_unet,
    resolve_path,
    save_outputs,
    select_device,
    visualize,
)


DEFAULT_OUTPUT_DIR = ROOT / "predictions" / "multicontrast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict vessels by running one U-Net on several contrast-enhanced "
            "versions of the same image, then combining probability maps."
        )
    )
    parser.add_argument(
        "--image",
        default="rabbit_bladder.png",
        help="Input 8-bit or 16-bit grayscale PNG/TIFF.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to models/unet_2ch_input.pt.",
    )
    parser.add_argument(
        "--ground-truth",
        help="Optional ground-truth mask used to calculate metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for combined and per-variant output files.",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--combine",
        choices=("max", "mean", "weighted-mean"),
        default="max",
        help="How to combine per-variant probability maps.",
    )
    parser.add_argument(
        "--variant-min-probability",
        type=float,
        default=0.0,
        help=(
            "Set probabilities below this value to zero before combining. "
            "Useful for suppressing weak noise when --combine=max."
        ),
    )
    parser.add_argument(
        "--min-vessel-brightness",
        type=float,
        default=0.0,
        help=(
            "Only keep predictions where the normalized original image "
            "brightness is at least this value. 0 disables brightness filtering."
        ),
    )
    parser.add_argument(
        "--brightness-mask-dilate",
        type=int,
        default=0,
        help=(
            "Dilate the brightness mask by this many pixels before applying it. "
            "Useful when bright vessel cores have dim edges."
        ),
    )
    parser.add_argument(
        "--sigma-min",
        type=int,
        default=2,
        help="Minimum Frangi sigma used for preprocessing/postprocessing.",
    )
    parser.add_argument(
        "--sigma-max",
        type=int,
        default=8,
        help="Maximum Frangi sigma used for preprocessing/postprocessing.",
    )
    parser.add_argument(
        "--frangi-input-weight",
        type=float,
        default=0.35,
        help="Blend weight for Frangi response in 1-channel model inputs.",
    )
    parser.add_argument(
        "--post-frangi-weight",
        type=float,
        default=0.15,
        help="Blend this amount of Frangi response into each probability map.",
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
        "--frame",
        type=int,
        default=0,
        help="Frame index when reading a multi-frame TIFF.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device such as "auto", "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--save-variants",
        action="store_true",
        help="Save each variant image and probability map for inspection.",
    )
    parser.add_argument(
        "--intensity-window-ensemble",
        action="store_true",
        help=(
            "Add U-Net runs for raw-intensity windows. Pixels inside each "
            "window are set to maximum brightness and other pixels are set to zero."
        ),
    )
    parser.add_argument(
        "--intensity-window-high",
        type=int,
        default=255,
        help="Highest 8-bit intensity used by --intensity-window-ensemble.",
    )
    parser.add_argument(
        "--intensity-window-low",
        type=int,
        default=95,
        help="Lowest 8-bit intensity included by --intensity-window-ensemble.",
    )
    parser.add_argument(
        "--intensity-window-step",
        type=int,
        default=20,
        help=(
            "Intensity interval size for --intensity-window-ensemble. The "
            "default produces windows like 255-235, 234-215, and so on."
        ),
    )
    parser.add_argument(
        "--intensity-window-weight",
        type=float,
        default=0.75,
        help="Combination weight assigned to each intensity-window prediction.",
    )
    parser.add_argument(
        "--intensity-window-mask-probability",
        action="store_true",
        help=(
            "After predicting each intensity-window variant, keep probability "
            "only inside that intensity window."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save results without opening a GUI window.",
    )
    return parser.parse_args()


def normalize_01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - low) / (high - low)


def read_image_as_grayscale(path: Path, frame: int = 0) -> np.ndarray:
    with Image.open(path) as image:
        frame_count = getattr(image, "n_frames", 1)
        if frame < 0 or frame >= frame_count:
            raise ValueError(
                f"Frame {frame} is outside the valid range 0-{frame_count - 1}."
            )
        image.seek(frame)
        array = np.asarray(image).copy()
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] in (3, 4):
            return np.asarray(image.convert("L")).copy()
    raise ValueError(f"Expected a grayscale or RGB/RGBA image, got {array.shape}: {path}")


def percentile_stretch(
    image: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high <= low:
        return image.astype(np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    uint8_image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(uint8_image).astype(np.float32) / 255.0


def local_contrast_normalize(image: np.ndarray, sigma: float = 18.0) -> np.ndarray:
    background = cv2.GaussianBlur(
        image.astype(np.float32),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    corrected = image.astype(np.float32) - background
    return normalize_01(corrected)


def build_variants(image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    clahe = apply_clahe(image)
    local = local_contrast_normalize(image)
    frangi_friendly = np.maximum(clahe, local)

    return [
        ("original", image.astype(np.float32), 1.00),
        ("clahe", clahe, 0.95),
        ("gamma_brighten", np.power(image, 0.65).astype(np.float32), 0.90),
        ("low_window", percentile_stretch(image, 0.5, 88.0), 0.90),
        ("mid_window", percentile_stretch(image, 5.0, 96.0), 0.85),
        ("local_contrast", local, 0.85),
        ("clahe_local_max", frangi_friendly.astype(np.float32), 0.80),
    ]


def intensity_window_ranges(
    high: int,
    low: int,
    step: int,
) -> list[tuple[int, int]]:
    ranges = []
    current_high = int(high)
    while current_high >= low:
        current_low = max(int(low), current_high - int(step))
        ranges.append((current_low, current_high))
        current_high = current_low - 1
    return ranges


def build_intensity_window_variants(
    raw_image: np.ndarray,
    high: int,
    low: int,
    step: int,
    weight: float,
) -> list[tuple[str, np.ndarray, float, np.ndarray]]:
    if high > 255 or low < 0 or low > high:
        raise ValueError("--intensity-window-low/high must satisfy 0 <= low <= high <= 255.")
    if step < 1:
        raise ValueError("--intensity-window-step must be >= 1.")
    if weight <= 0:
        raise ValueError("--intensity-window-weight must be > 0.")

    if np.issubdtype(raw_image.dtype, np.integer):
        max_value = float(np.iinfo(raw_image.dtype).max)
        raw_8bit = np.clip(np.rint(raw_image.astype(np.float32) / max_value * 255.0), 0, 255)
    else:
        raw_8bit = np.clip(np.rint(raw_image.astype(np.float32) * 255.0), 0, 255)
    raw_8bit = raw_8bit.astype(np.uint8)

    variants = []
    for window_low, window_high in intensity_window_ranges(high, low, step):
        mask = (raw_8bit >= window_low) & (raw_8bit <= window_high)
        window_image = np.where(mask, 1.0, 0.0).astype(np.float32)
        name = f"intensity_{window_low:03d}_{window_high:03d}"
        variants.append((name, window_image, float(weight), mask))
    return variants


def combine_probabilities(
    probability_items: list[tuple[str, np.ndarray, float]],
    method: str,
    variant_min_probability: float,
) -> np.ndarray:
    if not probability_items:
        raise ValueError("At least one probability map is required.")

    maps = []
    weights = []
    for _, probability, weight in probability_items:
        clipped = np.clip(probability.astype(np.float32), 0.0, 1.0)
        if variant_min_probability > 0:
            clipped = np.where(clipped >= variant_min_probability, clipped, 0.0)
        maps.append(clipped)
        weights.append(float(weight))

    stack = np.stack(maps, axis=0)
    if method == "max":
        return np.max(stack * np.asarray(weights, dtype=np.float32)[:, None, None], axis=0)
    if method == "mean":
        return np.mean(stack, axis=0)
    if method == "weighted-mean":
        weights_array = np.asarray(weights, dtype=np.float32)
        if np.sum(weights_array) <= 0:
            raise ValueError("Variant weights must have a positive sum.")
        return np.average(stack, axis=0, weights=weights_array).astype(np.float32)
    raise ValueError(f"Unsupported combine method: {method}")


def build_brightness_mask(
    image: np.ndarray,
    min_brightness: float,
    dilation_pixels: int,
) -> np.ndarray | None:
    if min_brightness <= 0:
        return None

    mask = image >= min_brightness
    if dilation_pixels > 0:
        kernel_size = 2 * dilation_pixels + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    return mask


def save_variant_outputs(
    output_dir: Path,
    stem: str,
    variants: list[tuple[str, np.ndarray, float]],
    probability_items: list[tuple[str, np.ndarray, float]],
) -> None:
    variant_dir = output_dir / f"{stem}_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    for name, variant, _ in variants:
        path = variant_dir / f"{name}_input.png"
        Image.fromarray(
            np.clip(np.rint(variant * 65535.0), 0, 65535).astype(np.uint16)
        ).save(path)

    for name, probability, _ in probability_items:
        path = variant_dir / f"{name}_probability.png"
        npy_path = variant_dir / f"{name}_probability.npy"
        Image.fromarray(
            np.clip(np.rint(probability * 65535.0), 0, 65535).astype(np.uint16)
        ).save(path)
        np.save(npy_path, probability.astype(np.float32))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if not 0.0 <= args.variant_min_probability < 1.0:
        raise ValueError("--variant-min-probability must be in [0, 1).")
    if not 0.0 <= args.min_vessel_brightness < 1.0:
        raise ValueError("--min-vessel-brightness must be in [0, 1).")
    if args.brightness_mask_dilate < 0:
        raise ValueError("--brightness-mask-dilate must be zero or positive.")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    image_path = resolve_path(args.image)
    model_path = resolve_path(args.model)
    output_dir = resolve_path(args.output_dir)
    ground_truth_path = (
        resolve_path(args.ground_truth) if args.ground_truth else None
    )
    device = select_device(args.device)

    raw_image = read_image_as_grayscale(image_path, frame=args.frame)
    image = normalize_image(raw_image)
    model, metadata = load_model(model_path, device)

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
    binary_mask = clean_structure_mask(
        probability >= args.threshold,
        min_area=args.min_structure_area,
        kernel_size=args.morph_kernel,
    )

    ground_truth = None
    metrics = None
    if ground_truth_path is not None:
        ground_truth = read_image_as_grayscale(ground_truth_path) > 0
        if ground_truth.shape != binary_mask.shape:
            raise ValueError(
                f"Ground-truth shape {ground_truth.shape} does not match "
                f"prediction shape {binary_mask.shape}."
            )
        metrics = calculate_metrics(binary_mask, ground_truth)

    frame_suffix = f"_frame{args.frame:04d}" if args.frame > 0 else ""
    stem = f"{image_path.stem}{frame_suffix}_multicontrast"
    mask_path, probability_path, probability_npy_path = save_outputs(
        output_dir,
        stem,
        probability,
        binary_mask,
    )
    if args.save_variants:
        save_variant_outputs(output_dir, stem, variants, probability_items)

    overlay = create_overlay(image, probability, binary_mask)
    figure_path = output_dir / f"{stem}_visualization.png"

    print(f"Device: {device}")
    print(
        f"Model: {model_path} "
        f"(base_channels={metadata['base_channels']}, "
        f"in_channels={metadata['in_channels']}, "
        f"epoch={metadata.get('epoch')})"
    )
    print(f"Input: {image_path}, shape={raw_image.shape}, dtype={raw_image.dtype}")
    print(f"Variants: {', '.join(name for name, _, _ in variants)}")
    if args.intensity_window_ensemble:
        print(
            "Intensity-window ensemble: "
            f"enabled, high={args.intensity_window_high}, "
            f"low={args.intensity_window_low}, "
            f"step={args.intensity_window_step}, "
            f"weight={args.intensity_window_weight}, "
            f"mask_probability={args.intensity_window_mask_probability}"
        )
    else:
        print("Intensity-window ensemble: disabled")
    print(
        f"Combine: {args.combine}, "
        f"variant_min_probability={args.variant_min_probability}"
    )
    if brightness_mask is not None:
        kept_ratio = float(np.count_nonzero(brightness_mask)) / float(brightness_mask.size)
        print(
            "Brightness filter: "
            f"min_vessel_brightness={args.min_vessel_brightness}, "
            f"dilate={args.brightness_mask_dilate}px, "
            f"kept_pixels={kept_ratio:.2%}"
        )
    else:
        print("Brightness filter: disabled")
    print(f"Threshold: {args.threshold}")
    print(
        "Frangi: "
        f"sigmas={args.sigma_min}-{args.sigma_max}, "
        f"input_weight={args.frangi_input_weight}, "
        f"post_weight={args.post_frangi_weight}"
    )
    print(
        "Structure cleanup: "
        f"min_area={args.min_structure_area}, "
        f"morph_kernel={args.morph_kernel}"
    )
    print(f"Mask: {mask_path}")
    print(f"Probability PNG: {probability_path}")
    print(f"Probability NPY: {probability_npy_path}")
    if args.save_variants:
        print(f"Variant outputs: {output_dir / f'{stem}_variants'}")
    if metrics is not None:
        print(
            "Metrics: "
            + " ".join(
                f"{name}={value:.4f}" for name, value in metrics.items()
            )
        )

    visualize(
        image=image,
        probability=probability,
        binary_mask=binary_mask,
        overlay=overlay,
        ground_truth=ground_truth,
        metrics=metrics,
        figure_path=figure_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
