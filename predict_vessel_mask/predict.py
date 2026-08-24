import argparse
import math
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from skimage.filters import frangi

from trainunet import UNet


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "models" / "unet_2ch_input.pt"
DEFAULT_OUTPUT_DIR = ROOT / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a vessel mask with a trained 2D U-Net."
    )
    parser.add_argument(
        "--image",
        default="gray_Figure_2.png",
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
        help="Directory for mask, probability, and visualization PNG files.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--sigma-min",
        type=int,
        default=3,
        help="Minimum Frangi sigma used for preprocessing/postprocessing.",
    )
    parser.add_argument(
        "--sigma-max",
        type=int,
        default=7,
        help="Maximum Frangi sigma used for preprocessing/postprocessing.",
    )
    parser.add_argument(
        "--frangi-input-weight",
        type=float,
        default=0.35,
        help=(
            "For 1-channel models, blend this amount of Frangi response into "
            "the normalized image before prediction. 2-channel models receive "
            "the raw image and Frangi response as separate channels."
        ),
    )
    parser.add_argument(
        "--post-frangi-weight",
        type=float,
        default=0.2,
        help="Blend this amount of Frangi response into the probability map after prediction.",
    )
    parser.add_argument(
        "--min-structure-area",
        type=int,
        default=64,
        help="Remove connected mask components smaller than this pixel area.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Odd kernel size used for simple opening/closing postprocessing.",
    )
    parser.add_argument(
        "--no-frangi-preprocess",
        action="store_true",
        help="Disable Frangi preprocessing before U-Net prediction.",
    )
    parser.add_argument(
        "--no-frangi-postprocess",
        action="store_true",
        help="Disable Frangi probability-map postprocessing after U-Net prediction.",
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
        "--no-show",
        action="store_true",
        help="Save results without opening a GUI window.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def read_grayscale(path: Path, frame: int = 0) -> np.ndarray:
    with Image.open(path) as image:
        frame_count = getattr(image, "n_frames", 1)
        if frame < 0 or frame >= frame_count:
            raise ValueError(
                f"Frame {frame} is outside the valid range 0-{frame_count - 1}."
            )
        image.seek(frame)
        array = np.asarray(image).copy()

    if array.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image, got shape {array.shape}: {path}"
        )
    return array


def normalize_image(array: np.ndarray) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            "The model expects an 8-bit or 16-bit integer image. "
            "Convert float TIFF data with convert_rsom_slices_to_png.py first."
        )
    maximum = float(np.iinfo(array.dtype).max)
    return array.astype(np.float32) / maximum


def normalize_01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return (array - minimum) / (maximum - minimum)


def calculate_frangi_response(
    image: np.ndarray,
    sigma_min: int,
    sigma_max: int,
) -> np.ndarray:
    if sigma_min < 1 or sigma_max < sigma_min:
        raise ValueError("Sigma values must satisfy 1 <= sigma-min <= sigma-max.")

    response = frangi(
        image.astype(np.float32),
        sigmas=range(sigma_min, sigma_max + 1),
        black_ridges=False,
    )
    return normalize_01(response)


def build_model_input(
    image: np.ndarray,
    frangi_response: np.ndarray | None,
    in_channels: int,
    frangi_input_weight: float,
) -> np.ndarray:
    if frangi_response is None:
        if in_channels == 1:
            return image
        raise ValueError(
            f"Model expects {in_channels} input channels, but Frangi preprocessing is disabled."
        )

    if in_channels == 1:
        return np.clip(
            image + frangi_input_weight * frangi_response,
            0.0,
            1.0,
        ).astype(np.float32)
    if in_channels == 2:
        return np.stack([image, frangi_response], axis=0).astype(np.float32)

    raise ValueError(
        f"Unsupported model input channel count: {in_channels}. "
        "This script supports 1-channel or 2-channel U-Net checkpoints."
    )


def postprocess_probability(
    probability: np.ndarray,
    sigma_min: int,
    sigma_max: int,
    frangi_weight: float,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)
    if frangi_weight <= 0:
        return np.clip(probability, 0.0, 1.0)

    frangi_response = calculate_frangi_response(
        probability,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    return np.clip(
        probability + frangi_weight * frangi_response,
        0.0,
        1.0,
    ).astype(np.float32)


def clean_structure_mask(
    binary_mask: np.ndarray,
    min_area: int,
    kernel_size: int,
) -> np.ndarray:
    mask = binary_mask.astype(np.uint8)
    if kernel_size > 1:
        if kernel_size % 2 == 0:
            raise ValueError("--morph-kernel must be odd.")
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if min_area <= 1:
        return mask.astype(bool)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(bool)


def pad_for_unet(
    image: np.ndarray, minimum_size: int = 256, divisor: int = 16
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[-2:]
    padded_height = max(minimum_size, math.ceil(height / divisor) * divisor)
    padded_width = max(minimum_size, math.ceil(width / divisor) * divisor)

    top = (padded_height - height) // 2
    bottom = padded_height - height - top
    left = (padded_width - width) // 2
    right = padded_width - width - left
    if image.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    elif image.ndim == 3:
        pad_width = ((0, 0), (top, bottom), (left, right))
    else:
        raise ValueError(f"Expected 2-D or 3-D image input, got {image.shape}.")
    padded = np.pad(image, pad_width, constant_values=0)
    return padded, (top, bottom, left, right)


def remove_padding(
    array: np.ndarray,
    padding: tuple[int, int, int, int],
    original_shape: tuple[int, int],
) -> np.ndarray:
    top, _, left, _ = padding
    height, width = original_shape
    if array.ndim == 2:
        return array[top : top + height, left : left + width]
    return array[..., top : top + height, left : left + width]


def clean_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[UNet, dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {checkpoint_path}\n"
            "Train the model first or specify --model."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
        }
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        metadata = {}
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = clean_state_dict(state_dict)
    first_weight = state_dict.get("encoder1.block.0.weight")
    if first_weight is None or first_weight.ndim != 4:
        raise ValueError("Unable to infer U-Net width from the checkpoint.")
    base_channels = int(first_weight.shape[0])
    in_channels = int(first_weight.shape[1])
    output_weight = state_dict.get("output.weight")
    if output_weight is None or output_weight.ndim != 4:
        raise ValueError("Unable to infer U-Net output channels from the checkpoint.")
    out_channels = int(output_weight.shape[0])

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    metadata["base_channels"] = base_channels
    metadata["in_channels"] = in_channels
    metadata["out_channels"] = out_channels
    return model, metadata


@torch.inference_mode()
def predict_outputs(
    model: UNet,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    padded, padding = pad_for_unet(image)
    tensor = torch.from_numpy(padded)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0).to(device)

    use_amp = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(tensor)
        probability = torch.sigmoid(logits)

    probability_array = probability[0].float().cpu().numpy()
    return remove_padding(probability_array, padding, image.shape[-2:])


@torch.inference_mode()
def predict(
    model: UNet,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    return predict_outputs(model, image, device)[0]


def calculate_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> dict[str, float]:
    predicted = prediction.astype(bool)
    truth = ground_truth.astype(bool)
    true_positive = float(np.count_nonzero(predicted & truth))
    false_positive = float(np.count_nonzero(predicted & ~truth))
    false_negative = float(np.count_nonzero(~predicted & truth))
    epsilon = 1e-8

    return {
        "dice": (2.0 * true_positive + epsilon)
        / (
            2.0 * true_positive
            + false_positive
            + false_negative
            + epsilon
        ),
        "iou": (true_positive + epsilon)
        / (true_positive + false_positive + false_negative + epsilon),
        "precision": (true_positive + epsilon)
        / (true_positive + false_positive + epsilon),
        "recall": (true_positive + epsilon)
        / (true_positive + false_negative + epsilon),
    }


def display_normalization(image: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(image, [1.0, 99.5])
    if upper <= lower:
        lower = float(image.min())
        upper = float(image.max())
    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lower) / (upper - lower), 0.0, 1.0)


def create_overlay(
    image: np.ndarray, probability: np.ndarray, binary_mask: np.ndarray
) -> np.ndarray:
    display_image = display_normalization(image)
    overlay = np.repeat(display_image[..., None], 3, axis=2)
    alpha = np.clip(probability[..., None], 0.0, 0.7)
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    selected_alpha = alpha * binary_mask[..., None]
    return overlay * (1.0 - selected_alpha) + red * selected_alpha


def save_outputs(
    output_dir: Path,
    stem: str,
    probability: np.ndarray,
    binary_mask: np.ndarray,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{stem}_mask.png"
    probability_path = output_dir / f"{stem}_probability.png"
    probability_npy_path = output_dir / f"{stem}_probability.npy"

    mask_uint8 = binary_mask.astype(np.uint8) * 255
    probability_uint16 = np.clip(
        np.rint(probability * 65535.0), 0, 65535
    ).astype(np.uint16)
    Image.fromarray(mask_uint8).save(mask_path)
    Image.fromarray(probability_uint16).save(probability_path)
    np.save(probability_npy_path, probability.astype(np.float32))
    return mask_path, probability_path, probability_npy_path


def visualize(
    image: np.ndarray,
    probability: np.ndarray,
    binary_mask: np.ndarray,
    overlay: np.ndarray,
    ground_truth: np.ndarray | None,
    metrics: dict[str, float] | None,
    figure_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    figure.canvas.manager.set_window_title("U-Net Vessel Segmentation")

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=np.percentile(image, 1.0),
        vmax=np.percentile(image, 99.5),
    )
    axes[0].set_title("Input")
    probability_plot = axes[1].imshow(
        probability,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Vessel probability")
    figure.colorbar(probability_plot, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted mask")

    axes[3].imshow(overlay)
    overlay_title = "Prediction overlay"
    if ground_truth is not None and np.any(ground_truth):
        axes[3].contour(
            ground_truth,
            levels=[0.5],
            colors="lime",
            linewidths=0.8,
        )
        overlay_title += " (GT: green)"
    axes[3].set_title(overlay_title)

    for axis in axes:
        axis.axis("off")

    if metrics is not None:
        figure.suptitle(
            " ".join(
                f"{name.capitalize()}={value:.4f}"
                for name, value in metrics.items()
            )
        )

    figure.tight_layout()
    figure.savefig(figure_path, dpi=160, bbox_inches="tight")
    print(f"Visualization: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

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

    raw_image = read_grayscale(image_path, frame=args.frame)
    image = normalize_image(raw_image)
    model, metadata = load_model(model_path, device)
    frangi_response = None
    if not args.no_frangi_preprocess:
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
    raw_probability = predict(model, model_input, device)
    probability = raw_probability
    if not args.no_frangi_postprocess:
        probability = postprocess_probability(
            probability,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            frangi_weight=args.post_frangi_weight,
        )
    binary_mask = clean_structure_mask(
        probability >= args.threshold,
        min_area=args.min_structure_area,
        kernel_size=args.morph_kernel,
    )

    ground_truth = None
    metrics = None
    if ground_truth_path is not None:
        ground_truth = read_grayscale(ground_truth_path) > 0
        if ground_truth.shape != binary_mask.shape:
            raise ValueError(
                f"Ground-truth shape {ground_truth.shape} does not match "
                f"prediction shape {binary_mask.shape}."
            )
        metrics = calculate_metrics(binary_mask, ground_truth)

    frame_suffix = f"_frame{args.frame:04d}" if args.frame > 0 else ""
    stem = f"{image_path.stem}{frame_suffix}"
    mask_path, probability_path, probability_npy_path = save_outputs(
        output_dir,
        stem,
        probability,
        binary_mask,
    )
    if frangi_response is not None:
        frangi_path = output_dir / f"{stem}_frangi_preprocess.png"
        Image.fromarray(
            np.clip(np.rint(frangi_response * 65535.0), 0, 65535).astype(np.uint16)
        ).save(frangi_path)
    raw_probability_npy_path = output_dir / f"{stem}_probability_raw.npy"
    np.save(raw_probability_npy_path, raw_probability.astype(np.float32))
    overlay = create_overlay(image, probability, binary_mask)
    figure_path = output_dir / f"{stem}_visualization.png"

    print(f"Device: {device}")
    print(
        f"Model: {model_path} "
        f"(base_channels={metadata['base_channels']}, "
        f"epoch={metadata.get('epoch')})"
    )
    print(f"Input: {image_path}, shape={raw_image.shape}, dtype={raw_image.dtype}")
    print(f"Threshold: {args.threshold}")
    print(
        "Frangi: "
        f"preprocess={not args.no_frangi_preprocess}, "
        f"postprocess={not args.no_frangi_postprocess}, "
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
    print(f"Raw probability NPY: {raw_probability_npy_path}")
    if frangi_response is not None:
        print(f"Frangi preprocessing map: {frangi_path}")
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
