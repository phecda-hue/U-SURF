import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from predict import (
    calculate_frangi_response,
    clean_structure_mask,
    create_overlay,
    display_normalization,
    normalize_image,
    pad_for_unet,
    remove_padding,
    resolve_path,
    save_outputs,
    select_device,
)
from predict_multicontrast import (
    build_brightness_mask,
    build_intensity_window_variants,
    build_variants,
    combine_probabilities,
    postprocess_probability,
    save_variant_outputs,
)


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VessMapUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, cb = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        self.e1 = ConvBlock(in_channels, c1)
        self.e2 = ConvBlock(c1, c2)
        self.e3 = ConvBlock(c2, c3)
        self.e4 = ConvBlock(c3, c4)
        self.b = ConvBlock(c4, cb)
        self.u4 = nn.ConvTranspose2d(cb, c4, kernel_size=2, stride=2)
        self.d4 = ConvBlock(c4 + c4, c4)
        self.u3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.d3 = ConvBlock(c3 + c3, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.d2 = ConvBlock(c2 + c2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.d1 = ConvBlock(c1 + c1, c1)
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        b = self.b(F.max_pool2d(e4, 2))

        d4 = self.d4(torch.cat((self.u4(b), e4), dim=1))
        d3 = self.d3(torch.cat((self.u3(d4), e3), dim=1))
        d2 = self.d2(torch.cat((self.u2(d3), e2), dim=1))
        d1 = self.d1(torch.cat((self.u1(d2), e1), dim=1))
        return self.out(d1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-predict vessel masks with a VessMAP checkpoint by running "
            "multicontrast variants and combining probability maps."
        )
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Single input image. Can be passed more than once.",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help="Input directory. Can be passed more than once.",
    )
    parser.add_argument(
        "--model",
        default=str(ROOT.parent / "models" / "vessmap.pt"),
        help="Path to models/vessmap.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "predictions" / "vessmap"),
        help="Directory for predictions.",
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
        help="Set probabilities below this value to zero before combining.",
    )
    parser.add_argument(
        "--min-vessel-brightness",
        type=float,
        default=0.0,
        help="Only keep predictions where the normalized original image is this bright.",
    )
    parser.add_argument(
        "--brightness-mask-dilate",
        type=int,
        default=0,
        help="Dilate the brightness mask by this many pixels before applying it.",
    )
    parser.add_argument("--sigma-min", type=int, default=2)
    parser.add_argument("--sigma-max", type=int, default=8)
    parser.add_argument(
        "--post-frangi-weight",
        type=float,
        default=0.15,
        help="Blend this amount of Frangi response into each probability map.",
    )
    parser.add_argument("--min-structure-area", type=int, default=48)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-variants",
        action="store_true",
        help="Save each variant image and probability map for inspection.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose main multicontrast outputs already exist.",
    )
    parser.add_argument(
        "--intensity-window-ensemble",
        action="store_true",
        help="Add U-Net runs for raw-intensity windows.",
    )
    parser.add_argument("--intensity-window-high", type=int, default=255)
    parser.add_argument("--intensity-window-low", type=int, default=95)
    parser.add_argument("--intensity-window-step", type=int, default=20)
    parser.add_argument("--intensity-window-weight", type=float, default=0.75)
    parser.add_argument(
        "--intensity-window-mask-probability",
        action="store_true",
        help="Keep each intensity-window probability only inside that window.",
    )
    return parser.parse_args()


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_vessmap_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a VessMAP checkpoint with a 'model' key: {checkpoint_path}")

    state_dict = clean_state_dict(checkpoint["model"])
    first_weight = state_dict["e1.layers.0.weight"]
    base_channels = int(first_weight.shape[0])
    in_channels = int(first_weight.shape[1])
    model = VessMapUNet(in_channels=in_channels, base_channels=base_channels)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("val_metrics", {}),
        "config": checkpoint.get("config", {}),
        "in_channels": in_channels,
        "base_channels": base_channels,
    }


@torch.inference_mode()
def predict_probability(model: nn.Module, model_input: np.ndarray, device: torch.device) -> np.ndarray:
    padded, padding = pad_for_unet(model_input)
    tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        probability = torch.sigmoid(model(tensor))
    probability_array = probability[0, 0].float().cpu().numpy()
    return remove_padding(probability_array, padding, model_input.shape[-2:])


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


def read_image_grayscale(path: Path, frame: int = 0) -> np.ndarray:
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
        if array.ndim == 3:
            return np.asarray(image.convert("L")).copy()
    raise ValueError(f"Expected a 2-D or RGB/RGBA image, got shape {array.shape}: {path}")


def save_visualization(path: Path, raw_image: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overlay = np.clip(create_overlay(normalize_image(raw_image), probability, mask), 0.0, 1.0)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(display_normalization(normalize_image(raw_image)), cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(probability, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Probability")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def predict_multicontrast_probability(
    model: nn.Module,
    raw_image: np.ndarray,
    image: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray, float]], list[tuple[str, np.ndarray, float]]]:
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
        model_input = np.stack([variant, frangi_response], axis=0).astype(np.float32)
        raw_probability = predict_probability(model, model_input, device)
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
    return probability, variants, probability_items


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

    input_dirs = [resolve_path(path) for path in args.input_dir]
    if not input_dirs and not args.image:
        input_dirs = [ROOT / "raw" / "external", ROOT / "raw" / "others"]

    model_path = resolve_path(args.model)
    output_root = resolve_path(args.output_dir)
    device = select_device(args.device)
    model, metadata = load_vessmap_model(model_path, device)
    if metadata["in_channels"] != 2:
        raise ValueError(f"Expected a 2-channel model, got {metadata['in_channels']} channels.")

    paths = collect_image_paths(args.image, input_dirs)
    print(
        f"Device: {device}, model: {model_path}, "
        f"epoch={metadata.get('epoch')}, inputs={len(paths)}"
    )
    print(f"Validation metrics: {metadata.get('metrics')}")
    print(
        f"Multicontrast: combine={args.combine}, "
        f"threshold={args.threshold}, sigmas={args.sigma_min}-{args.sigma_max}, "
        f"post_frangi_weight={args.post_frangi_weight}"
    )

    for index, image_path in enumerate(paths, 1):
        try:
            relative = image_path.relative_to(ROOT / "raw")
            output_dir = output_root / relative.parent
        except ValueError:
            output_dir = output_root / image_path.parent.name
        frame_suffix = f"_frame{args.frame:04d}" if args.frame > 0 else ""
        stem = f"{image_path.stem}{frame_suffix}_multicontrast"
        expected_outputs = [
            output_dir / f"{stem}_mask.png",
            output_dir / f"{stem}_probability.png",
            output_dir / f"{stem}_probability.npy",
            output_dir / f"{stem}_visualization.png",
        ]
        if args.skip_existing and all(path.is_file() for path in expected_outputs):
            print(f"[{index}/{len(paths)}] Skipping existing outputs: {image_path}")
            continue

        raw_image = read_image_grayscale(image_path, frame=args.frame)
        image = normalize_image(raw_image)
        probability, variants, probability_items = predict_multicontrast_probability(
            model,
            raw_image,
            image,
            args,
            device,
        )
        mask = clean_structure_mask(
            probability >= args.threshold,
            min_area=args.min_structure_area,
            kernel_size=args.morph_kernel,
        )

        mask_path, probability_path, probability_npy_path = save_outputs(
            output_dir,
            stem,
            probability,
            mask,
        )
        if args.save_variants:
            save_variant_outputs(output_dir, stem, variants, probability_items)
        save_visualization(
            output_dir / f"{stem}_visualization.png",
            raw_image,
            probability,
            mask,
        )
        print(
            f"[{index}/{len(paths)}] {image_path} -> "
            f"{mask_path}, {probability_path}, {probability_npy_path}"
        )


if __name__ == "__main__":
    main()
