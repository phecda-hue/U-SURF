import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path("././raw/pig_bile_duct.png")


def normalize01(a: np.ndarray, lohi: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, lohi)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def read_rgba_luma(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        alpha = None
    elif image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3]
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = None

    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    return rgb, luma, alpha


def imwrite_checked(path: Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"Could not encode image as {suffix}: {path}")
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Could not write image: {path}")


def save_with_luma(path: Path, rgb: np.ndarray, luma: np.ndarray, alpha: np.ndarray | None) -> None:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = np.clip(luma * 255, 0, 255)
    out_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    if alpha is None:
        out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    else:
        out = cv2.cvtColor(np.dstack([out_rgb, alpha]), cv2.COLOR_RGBA2BGRA)
    imwrite_checked(path, out)


def save_gray(path: Path, image: np.ndarray, lohi: tuple[float, float] = (0.5, 99.5)) -> None:
    imwrite_checked(path, (normalize01(image, lohi) * 255).astype(np.uint8))


def dilated_kernel_1d(level: int) -> np.ndarray:
    base = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    if level == 0:
        return base
    step = 2**level
    kernel = np.zeros((len(base) - 1) * step + 1, dtype=np.float32)
    kernel[::step] = base
    return kernel


def separable_reflect_blur(image: np.ndarray, level: int) -> np.ndarray:
    kernel = dilated_kernel_1d(level)
    tmp = cv2.sepFilter2D(image, cv2.CV_32F, kernel, np.array([1], dtype=np.float32), borderType=cv2.BORDER_REFLECT)
    return cv2.sepFilter2D(tmp, cv2.CV_32F, np.array([1], dtype=np.float32), kernel, borderType=cv2.BORDER_REFLECT)


def atrous_decompose(image: np.ndarray, levels: int) -> tuple[list[np.ndarray], np.ndarray]:
    current = image.astype(np.float32)
    bands: list[np.ndarray] = []
    for level in range(levels):
        smooth = separable_reflect_blur(current, level)
        bands.append(current - smooth)
        current = smooth
    return bands, current


def robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(max(1.4826 * mad, 1e-6))


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def white_annotation_mask(rgb: np.ndarray) -> np.ndarray:
    white = (rgb[..., 0] > 235) & (rgb[..., 1] > 235) & (rgb[..., 2] > 235)
    return cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def structure_weight(luma: np.ndarray, annotation: np.ndarray) -> np.ndarray:
    local = cv2.GaussianBlur(luma, (0, 0), 7)
    detail = np.maximum(luma - local, 0)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    mask = (
        (luma > np.percentile(luma[~annotation], 86))
        | (detail > np.percentile(detail[~annotation], 91))
        | (grad > np.percentile(grad[~annotation], 92))
    )
    mask |= annotation
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), 2.0)


def wavelet_denoise_luma(
    luma: np.ndarray,
    structure: np.ndarray,
    levels: int,
    strength: float,
    mid_strength: float,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    bands, coarse = atrous_decompose(luma, levels)
    denoised_bands: list[np.ndarray] = []
    removed_bands: list[np.ndarray] = []

    for idx, band in enumerate(bands):
        sigma = robust_sigma(band)
        # Levels 2-4 roughly correspond to the visible 8-64 px ripple family.
        level_scale = 1.0
        if 1 <= idx <= 3:
            level_scale = mid_strength
        if idx >= 4:
            level_scale = 0.75 * mid_strength

        threshold = strength * level_scale * sigma
        shrunk = soft_threshold(band, threshold)

        # Preserve vessel-like structures, suppress background wavelet coefficients more strongly.
        preserve = np.clip(structure * (0.85 if idx <= 2 else 0.65), 0, 1)
        denoised = preserve * band + (1 - preserve) * shrunk
        denoised_bands.append(denoised.astype(np.float32))
        removed_bands.append((band - denoised).astype(np.float32))

    denoised_luma = coarse.copy()
    removed = np.zeros_like(luma, dtype=np.float32)
    for denoised, removed_band in zip(denoised_bands, removed_bands):
        denoised_luma += denoised
        removed += removed_band

    return np.clip(denoised_luma, 0, 1), removed, bands, denoised_bands


def make_band_contact_sheet(bands: list[np.ndarray]) -> np.ndarray:
    previews = [(normalize01(b, (0.2, 99.8)) * 255).astype(np.uint8) for b in bands]
    return np.hstack(previews)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wavelet denoising for pig bile duct image.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--strength", type=float, default=1.15)
    parser.add_argument("--mid-strength", type=float, default=1.8)
    parser.add_argument("--strong-strength", type=float, default=1.55)
    parser.add_argument("--strong-mid-strength", type=float, default=2.7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb, luma, alpha = read_rgba_luma(args.input)
    annotation = white_annotation_mask(rgb)
    structure = structure_weight(luma, annotation)

    denoised, removed, original_bands, denoised_bands = wavelet_denoise_luma(
        luma, structure, args.levels, args.strength, args.mid_strength
    )
    denoised_strong, removed_strong, _, _ = wavelet_denoise_luma(
        luma, structure, args.levels, args.strong_strength, args.strong_mid_strength
    )

    save_with_luma(args.output_dir / "pig_bile_duct_wavelet_denoised.png", rgb, denoised, alpha)
    save_with_luma(args.output_dir / "pig_bile_duct_wavelet_denoised_strong.png", rgb, denoised_strong, alpha)
    save_gray(args.output_dir / "pig_bile_duct_wavelet_removed_noise.png", removed, (0.2, 99.8))
    save_gray(args.output_dir / "pig_bile_duct_wavelet_removed_noise_strong.png", removed_strong, (0.2, 99.8))
    cv2.imwrite(str(args.output_dir / "pig_bile_duct_wavelet_structure_weight.png"), (structure * 255).astype(np.uint8))
    cv2.imwrite(str(args.output_dir / "pig_bile_duct_wavelet_original_bands.png"), make_band_contact_sheet(original_bands))
    cv2.imwrite(str(args.output_dir / "pig_bile_duct_wavelet_denoised_bands.png"), make_band_contact_sheet(denoised_bands))

    with open(args.output_dir / "pig_bile_duct_wavelet_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"levels: {args.levels}\n")
        f.write(f"strength: {args.strength}\n")
        f.write(f"mid_strength: {args.mid_strength}\n")
        f.write(f"strong_strength: {args.strong_strength}\n")
        f.write(f"strong_mid_strength: {args.strong_mid_strength}\n")
        f.write(f"removed std: {float(np.std(removed)):.6g}\n")
        f.write(f"removed strong std: {float(np.std(removed_strong)):.6g}\n")


if __name__ == "__main__":
    main()
