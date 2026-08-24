from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tifffile
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    gaussian_filter1d,
    label,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PhantomConfig:
    # ---------- dataset ----------
    n_aug_per_slice: int = 1
    min_vessel_pixels: int = 50
    random_seed: int = 42
    patch_size: int = 256
    patches_per_slice: int = 4

    # ---------- mask preprocessing ----------
    mask_threshold: float = 0.5
    random_morphology: bool = False

    # ---------- pseudo-PA clean image generation ----------
    use_input_as_background: bool = False
    input_background_weight_range: Tuple[float, float] = (0.03, 0.12)

    depth_attenuation_alpha_range: Tuple[float, float] = (0.4, 1.4)
    vessel_component_amplitude_range: Tuple[float, float] = (0.75, 1.35)

    lowfreq_field_sigma_range: Tuple[float, float] = (18.0, 40.0)

    blur_sigma_x_range: Tuple[float, float] = (0.5, 1.6)
    blur_sigma_y_range: Tuple[float, float] = (1.2, 3.2)

    additive_noise_std_range: Tuple[float, float] = (0.01, 0.05)
    speckle_noise_std_range: Tuple[float, float] = (0.03, 0.12)
    stripe_noise_std_range: Tuple[float, float] = (0.00, 0.03)

    gamma_range: Tuple[float, float] = (0.85, 1.25)
    log_compression_prob: float = 0.35

    # ---------- distortion generation ----------
    generate_distortion: bool = False
    max_abs_shift: int = 8

    n_sine_range: Tuple[int, int] = (1, 3)
    local_bump_range: Tuple[int, int] = (0, 2)
    impulse_run_range: Tuple[int, int] = (0, 3)

    impulse_width_range: Tuple[int, int] = (1, 6)
    impulse_shift_range: Tuple[int, int] = (1, 4)

    smooth_shift_sigma_range: Tuple[float, float] = (2.0, 7.0)

    # ---------- saving ----------
    save_uint16_image: bool = True
    save_float32_image: bool = False


# ============================================================
# Utility functions
# ============================================================

def normalize_01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    vmin = float(image.min())
    vmax = float(image.max())
    if vmax - vmin < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - vmin) / (vmax - vmin)


def ensure_stack(arr: np.ndarray) -> np.ndarray:
    """
    TIFF를 읽었을 때 결과가
    - (H, W) 이면 단일 slice로 간주
    - (N, H, W) 이면 multi-slice stack으로 간주
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        return arr

    raise ValueError(f"지원하지 않는 TIFF shape입니다: {arr.shape}")


def load_tiff_stack(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    return ensure_stack(arr)


def save_tiff_image(path: Path, image: np.ndarray, as_uint16: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if as_uint16:
        img = normalize_01(image)
        img = np.clip(np.round(img * 65535.0), 0, 65535).astype(np.uint16)
        tifffile.imwrite(str(path), img)
    else:
        tifffile.imwrite(str(path), image.astype(np.float32))


def save_tiff_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    tifffile.imwrite(str(path), mask_u8)


def rand_uniform(rng: np.random.Generator, value_range: Tuple[float, float]) -> float:
    return float(rng.uniform(value_range[0], value_range[1]))


def rand_int(rng: np.random.Generator, value_range: Tuple[int, int]) -> int:
    """
    value_range = (low, high), both inclusive
    """
    low, high = value_range
    return int(rng.integers(low, high + 1))


# ============================================================
# Mask / morphology
# ============================================================

def preprocess_mask(mask_slice: np.ndarray, cfg: PhantomConfig, rng: np.random.Generator) -> np.ndarray:
    """
    GT 마스크를 binary vessel mask로 변환합니다.
    """
    mask = mask_slice.astype(np.float32)

    if mask.max() > 1.0:
        mask = mask / mask.max()

    vessel = mask >= cfg.mask_threshold

    if cfg.random_morphology:
        # 약간의 형태 변화로 다양성 확보
        p = rng.random()

        if p < 0.25:
            vessel = binary_dilation(vessel, iterations=1)
        elif p < 0.40:
            vessel = binary_dilation(vessel, iterations=2)
        elif p < 0.55:
            vessel = binary_erosion(vessel, iterations=1)

    return vessel.astype(bool)


# ============================================================
# Pseudo-PA generation
# ============================================================

def make_low_frequency_field(shape: Tuple[int, int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    field = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    field = gaussian_filter(field, sigma=sigma)
    field = normalize_01(field)
    return field


def make_component_amplitude_map(vessel_mask: np.ndarray, cfg: PhantomConfig, rng: np.random.Generator) -> np.ndarray:
    """
    연결된 혈관 성분마다 다른 흡수 강도를 부여합니다.
    """
    lbl, num = label(vessel_mask)
    amp_map = np.zeros_like(vessel_mask, dtype=np.float32)

    if num == 0:
        return amp_map

    for comp_id in range(1, num + 1):
        amp = rand_uniform(rng, cfg.vessel_component_amplitude_range)
        amp_map[lbl == comp_id] = amp

    return amp_map


def generate_clean_pseudo_pa(
    vessel_mask: np.ndarray,
    original_slice: Optional[np.ndarray],
    cfg: PhantomConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    혈관 마스크를 이용해 clean pseudo-PA 영상을 생성합니다.
    """
    vessel_mask = vessel_mask.astype(bool)
    h, w = vessel_mask.shape

    # 1) 혈관 컴포넌트별 강도
    amp_map = make_component_amplitude_map(vessel_mask, cfg, rng)

    # 2) 혈관 내부 profile 강화 (거리변환 기반)
    dist = distance_transform_edt(vessel_mask).astype(np.float32)
    if dist.max() > 0:
        center_profile = 0.6 + 0.4 * np.sqrt(dist / (dist.max() + 1e-8))
    else:
        center_profile = np.zeros_like(dist, dtype=np.float32)

    # 3) 저주파 랜덤 field (공간적으로 intensity variation)
    lowfreq_sigma = rand_uniform(rng, cfg.lowfreq_field_sigma_range)
    lowfreq_field = make_low_frequency_field((h, w), sigma=lowfreq_sigma, rng=rng)
    lowfreq_gain = 0.75 + 0.50 * lowfreq_field

    # 4) 깊이 감쇠
    alpha = rand_uniform(rng, cfg.depth_attenuation_alpha_range)
    depth = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    attenuation = np.exp(-alpha * depth)

    # 5) 혈관 흡수 분포
    absorption = vessel_mask.astype(np.float32) * amp_map * center_profile * lowfreq_gain
    absorption *= attenuation

    # 6) PSF / 시스템 blur
    sigma_x = rand_uniform(rng, cfg.blur_sigma_x_range)
    sigma_y = rand_uniform(rng, cfg.blur_sigma_y_range)
    pa = gaussian_filter(absorption, sigma=(sigma_y, sigma_x))

    # 7) 약한 배경 생성
    background = np.zeros((h, w), dtype=np.float32)

    # 7-1) 저주파 배경
    bg_low = make_low_frequency_field((h, w), sigma=rng.uniform(25, 60), rng=rng)
    background += 0.04 * bg_low

    # 7-2) 원본 모달리티를 약하게 섞고 싶다면
    if cfg.use_input_as_background and original_slice is not None:
        orig = normalize_01(original_slice.astype(np.float32))
        orig_weight = rand_uniform(rng, cfg.input_background_weight_range)
        background += orig_weight * orig

    pa = pa + background

    # 8) multiplicative speckle noise
    speckle_std = rand_uniform(rng, cfg.speckle_noise_std_range)
    speckle = rng.normal(1.0, speckle_std, size=(h, w)).astype(np.float32)
    pa = pa * speckle

    # 9) additive correlated noise
    additive_std = rand_uniform(rng, cfg.additive_noise_std_range)
    additive_noise = rng.normal(0.0, additive_std, size=(h, w)).astype(np.float32)
    additive_noise = gaussian_filter(additive_noise, sigma=0.8)
    pa = pa + additive_noise

    # 10) stripe / scan noise
    stripe_std = rand_uniform(rng, cfg.stripe_noise_std_range)
    if stripe_std > 0:
        stripe_1d = rng.normal(0.0, stripe_std, size=w).astype(np.float32)
        stripe_1d = gaussian_filter1d(stripe_1d, sigma=rng.uniform(2.0, 8.0))
        pa = pa + stripe_1d[None, :]

    # 11) normalize
    pa = normalize_01(pa)

    # 12) 감마 또는 log compression
    gamma = rand_uniform(rng, cfg.gamma_range)
    pa = np.power(np.clip(pa, 0.0, 1.0), gamma)

    if rng.random() < cfg.log_compression_prob:
        k = rng.uniform(4.0, 10.0)
        pa = np.log1p(k * pa) / np.log1p(k)

    pa = normalize_01(pa)
    return pa.astype(np.float32)


# ============================================================
# Distortion generation
# ============================================================

def sample_displacement(width: int, cfg: PhantomConfig, rng: np.random.Generator) -> np.ndarray:
    """
    A-line별 수직 이동량 d(x)를 생성합니다.
    """
    x = np.arange(width, dtype=np.float32)
    d = np.zeros(width, dtype=np.float32)

    # 1) 부드러운 sine 조합
    n_sines = rand_int(rng, cfg.n_sine_range)
    for _ in range(n_sines):
        amp = rng.uniform(0.8, cfg.max_abs_shift * 0.7)
        wavelength = rng.uniform(width * 0.12, width * 0.65)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        d += amp * np.sin(2.0 * math.pi * x / wavelength + phase)

    # 2) 국소 bump 왜곡
    n_bumps = rand_int(rng, cfg.local_bump_range)
    for _ in range(n_bumps):
        center = rng.uniform(0, width - 1)
        sigma = rng.uniform(width * 0.02, width * 0.08)
        amp = rng.uniform(-cfg.max_abs_shift, cfg.max_abs_shift)
        d += amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    # 3) 부드러운 부분은 smoothing
    smooth_sigma = rand_uniform(rng, cfg.smooth_shift_sigma_range)
    d = gaussian_filter1d(d, sigma=smooth_sigma)

    # 4) 짧은 impulse-like line jump 추가
    n_runs = rand_int(rng, cfg.impulse_run_range)
    for _ in range(n_runs):
        start = int(rng.integers(0, max(1, width - 1)))
        run_width = rand_int(rng, cfg.impulse_width_range)
        shift_mag = rand_int(rng, cfg.impulse_shift_range)
        sign = -1 if rng.random() < 0.5 else 1
        d[start:start + run_width] += sign * shift_mag

    d = np.clip(d, -cfg.max_abs_shift, cfg.max_abs_shift)
    return d.astype(np.float32)


def warp_columns_linear(image: np.ndarray, displacement: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    image[y, x]를 column-wise vertical warp 합니다.
    displacement[x] = +2 이면 구조가 아래로 2픽셀 이동하도록 생성됩니다.

    output[y, x] = input[y - displacement[x], x]
    """
    image = image.astype(np.float32)
    h, w = image.shape
    y = np.arange(h, dtype=np.float32)

    out = np.zeros_like(image, dtype=np.float32)
    for x in range(w):
        source_y = y - displacement[x]
        out[:, x] = np.interp(
            source_y,
            y,
            image[:, x],
            left=fill_value,
            right=fill_value,
        )
    return out


def warp_columns_mask(mask: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    mask는 nearest-like 방식으로 warp 후 threshold합니다.
    """
    mask_f = mask.astype(np.float32)
    warped = warp_columns_linear(mask_f, displacement, fill_value=0.0)
    return (warped >= 0.5)


# ============================================================
# File matching
# ============================================================

def find_matching_mask(image_path: Path, mask_dir: Path) -> Path:
    """
    기본 가정:
    - image와 mask의 파일명이 같음
    필요하면 여기 규칙을 바꾸시면 됩니다.

    예:
    image: case001.tif
    mask : case001.tif
    """
    candidate = mask_dir / image_path.name
    if candidate.exists():
        return candidate

    # 흔한 예시를 몇 개 더 시도
    alt_names = [
        f"{image_path.stem}_gt{image_path.suffix}",
        f"{image_path.stem}_mask{image_path.suffix}",
        f"{image_path.stem.replace('image', 'mask')}{image_path.suffix}",
    ]

    for name in alt_names:
        p = mask_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(f"대응하는 mask 파일을 찾지 못했습니다: {image_path.name}")


# ============================================================
# Main dataset builder
# ============================================================

def patch_starts(length: int, patch_size: int) -> list[int]:
    """Return non-overlapping starts plus a final edge-aligned patch."""
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, patch_size))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def crop_or_pad(array: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    patch = np.asarray(array[y:y + size, x:x + size])
    pad_y = size - patch.shape[0]
    pad_x = size - patch.shape[1]
    if pad_y > 0 or pad_x > 0:
        patch = np.pad(patch, ((0, pad_y), (0, pad_x)), constant_values=0)
    return patch


def list_non_rsom_label_files(mask_dir: Path) -> list[Path]:
    label_files = sorted(
        list(mask_dir.glob("*.tif")) + list(mask_dir.glob("*.tiff"))
    )
    label_files = [path for path in label_files if "rsom" not in path.name.lower()]
    if not label_files:
        raise FileNotFoundError(f"No non-RSOM TIFF labels found in {mask_dir}")
    return label_files


def inspect_mask_files(mask_dir: Path) -> list[dict]:
    records = []
    for path in list_non_rsom_label_files(mask_dir):
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            records.append(
                {
                    "file": path.name,
                    "shape": tuple(int(value) for value in series.shape),
                    "axes": series.axes,
                    "dtype": str(series.dtype),
                }
            )
    return records


def build_dataset_from_non_rsom_masks(
    mask_dir: Path,
    output_dir: Path,
    cfg: PhantomConfig,
    max_label_files: int = 0,
    max_slices_per_file: int = 0,
) -> None:
    """Build clean pseudo-PA patches using only non-RSOM TubeNet labels."""
    rng = np.random.default_rng(cfg.random_seed)
    clean_pa_dir = output_dir / "clean_pa"
    clean_mask_dir = output_dir / "clean_mask"
    for directory in (clean_pa_dir, clean_mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    label_files = list_non_rsom_label_files(mask_dir)
    if max_label_files > 0:
        label_files = label_files[:max_label_files]

    rows = []
    skipped_empty_slices = 0
    for file_index, label_path in enumerate(label_files, start=1):
        # All current TubeNet label TIFFs are uncompressed and memmappable.
        # This prevents the 5+ GB optical-HREM label volume from being loaded
        # into RAM as one array.
        mask_stack = tifffile.memmap(label_path)
        if mask_stack.ndim == 2:
            mask_stack = mask_stack[None, ...]
        if mask_stack.ndim != 3:
            raise ValueError(
                f"Expected a ZYX label stack, got {mask_stack.shape}: {label_path}"
            )

        slice_count = int(mask_stack.shape[0])
        if max_slices_per_file > 0:
            slice_count = min(slice_count, max_slices_per_file)
        modality = label_path.stem.removesuffix("_labels")
        print(
            f"[{file_index}/{len(label_files)}] {label_path.name}: "
            f"using {slice_count}/{mask_stack.shape[0]} slices, "
            f"shape={tuple(mask_stack.shape)}",
            flush=True,
        )

        for slice_index in range(slice_count):
            raw_mask = np.asarray(mask_stack[slice_index])
            y_starts = patch_starts(raw_mask.shape[0], cfg.patch_size)
            x_starts = patch_starts(raw_mask.shape[1], cfg.patch_size)
            candidates = []
            for y in y_starts:
                for x in x_starts:
                    raw_patch = crop_or_pad(raw_mask, y, x, cfg.patch_size)
                    foreground = raw_patch > 0
                    if int(foreground.sum()) >= cfg.min_vessel_pixels:
                        candidates.append((y, x, raw_patch))

            if not candidates:
                skipped_empty_slices += 1
                continue
            if cfg.patches_per_slice > 0 and len(candidates) > cfg.patches_per_slice:
                chosen = rng.choice(
                    len(candidates), size=cfg.patches_per_slice, replace=False
                )
                candidates = [candidates[int(index)] for index in sorted(chosen)]

            for patch_index, (y, x, raw_patch) in enumerate(candidates):
                for aug_index in range(cfg.n_aug_per_slice):
                    vessel_mask = preprocess_mask(raw_patch, cfg, rng)
                    if int(vessel_mask.sum()) < cfg.min_vessel_pixels:
                        continue
                    clean_pa = generate_clean_pseudo_pa(
                        vessel_mask=vessel_mask,
                        original_slice=None,
                        cfg=cfg,
                        rng=rng,
                    )
                    sample_id = (
                        f"{modality}_z{slice_index:04d}_"
                        f"p{patch_index:02d}_a{aug_index:02d}"
                    )
                    clean_pa_path = clean_pa_dir / f"{sample_id}.tif"
                    clean_mask_path = clean_mask_dir / f"{sample_id}.tif"
                    save_tiff_image(
                        clean_pa_path, clean_pa, as_uint16=cfg.save_uint16_image
                    )
                    save_tiff_mask(clean_mask_path, vessel_mask)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source_modality": modality,
                            "source_mask_path": str(label_path.resolve()),
                            "slice_index": slice_index,
                            "patch_index": patch_index,
                            "patch_y": y,
                            "patch_x": x,
                            "augmentation_index": aug_index,
                            "clean_pa_path": str(clean_pa_path.resolve()),
                            "clean_mask_path": str(clean_mask_path.resolve()),
                            "vessel_pixels": int(vessel_mask.sum()),
                        }
                    )

        del mask_stack
        print(f"  cumulative samples: {len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No phantom samples were generated.")
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    info = {
        "purpose": "clean pseudo-photoacoustic phantom generation",
        "source": "non-RSOM TubeNet vessel masks only",
        "distortion_applied": False,
        "sample_count": len(rows),
        "source_label_count": len(label_files),
        "skipped_empty_slices": skipped_empty_slices,
        "config": {
            "patch_size": cfg.patch_size,
            "patches_per_slice": cfg.patches_per_slice,
            "augmentations_per_patch": cfg.n_aug_per_slice,
            "min_vessel_pixels": cfg.min_vessel_pixels,
            "random_seed": cfg.random_seed,
            "random_morphology": cfg.random_morphology,
            "use_input_as_background": False,
            "generate_distortion": False,
        },
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, ensure_ascii=False, indent=2)
    print(f"\nDone. Total clean phantom samples: {len(rows)}")
    print(f"Manifest: {manifest_path}")


def build_dataset(
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    cfg: PhantomConfig,
) -> None:
    rng = np.random.default_rng(cfg.random_seed)

    clean_pa_dir = output_dir / "clean_pa"
    distorted_pa_dir = output_dir / "distorted_pa"
    clean_mask_dir = output_dir / "clean_mask"
    distorted_mask_dir = output_dir / "distorted_mask"
    shift_dir = output_dir / "shift"
    meta_dir = output_dir / "meta"

    for d in [clean_pa_dir, distorted_pa_dir, clean_mask_dir, distorted_mask_dir, shift_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"

    image_files = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    if not image_files:
        raise FileNotFoundError(f"{image_dir} 안에 tif/tiff 파일이 없습니다.")

    rows = []

    for image_path in image_files:
        mask_path = find_matching_mask(image_path, mask_dir)

        image_stack = load_tiff_stack(image_path)
        mask_stack = load_tiff_stack(mask_path)

        if image_stack.shape[0] != mask_stack.shape[0]:
            raise ValueError(
                f"slice 수가 다릅니다.\n"
                f"image: {image_path} -> {image_stack.shape}\n"
                f"mask : {mask_path} -> {mask_stack.shape}"
            )

        if image_stack.shape[1:] != mask_stack.shape[1:]:
            raise ValueError(
                f"slice 크기가 다릅니다.\n"
                f"image: {image_path} -> {image_stack.shape}\n"
                f"mask : {mask_path} -> {mask_stack.shape}"
            )

        base_name = image_path.stem

        for slice_idx in range(image_stack.shape[0]):
            image_slice = image_stack[slice_idx].astype(np.float32)
            mask_slice = mask_stack[slice_idx]

            vessel_mask = preprocess_mask(mask_slice, cfg, rng)

            if int(vessel_mask.sum()) < cfg.min_vessel_pixels:
                continue

            for aug_idx in range(cfg.n_aug_per_slice):
                sample_id = f"{base_name}_s{slice_idx:04d}_a{aug_idx:02d}"

                # clean pseudo-PA 생성
                clean_pa = generate_clean_pseudo_pa(
                    vessel_mask=vessel_mask,
                    original_slice=image_slice,
                    cfg=cfg,
                    rng=rng,
                )

                clean_mask = vessel_mask.astype(bool)

                # 왜곡 생성
                if cfg.generate_distortion:
                    displacement = sample_displacement(clean_pa.shape[1], cfg, rng)
                    distorted_pa = warp_columns_linear(clean_pa, displacement, fill_value=0.0)
                    distorted_mask = warp_columns_mask(clean_mask, displacement)
                else:
                    displacement = np.zeros(clean_pa.shape[1], dtype=np.float32)
                    distorted_pa = clean_pa.copy()
                    distorted_mask = clean_mask.copy()

                # 저장 경로
                clean_pa_path = clean_pa_dir / f"{sample_id}.tif"
                distorted_pa_path = distorted_pa_dir / f"{sample_id}.tif"
                clean_mask_path = clean_mask_dir / f"{sample_id}.tif"
                distorted_mask_path = distorted_mask_dir / f"{sample_id}.tif"
                shift_path = shift_dir / f"{sample_id}.npy"
                meta_path = meta_dir / f"{sample_id}.npz"

                # 저장
                if cfg.save_uint16_image:
                    save_tiff_image(clean_pa_path, clean_pa, as_uint16=True)
                    save_tiff_image(distorted_pa_path, distorted_pa, as_uint16=True)
                else:
                    save_tiff_image(clean_pa_path, clean_pa, as_uint16=False)
                    save_tiff_image(distorted_pa_path, distorted_pa, as_uint16=False)

                save_tiff_mask(clean_mask_path, clean_mask)
                save_tiff_mask(distorted_mask_path, distorted_mask)

                np.save(shift_path, displacement.astype(np.float32))

                # 추가 메타데이터 저장
                np.savez_compressed(
                    meta_path,
                    displacement=displacement.astype(np.float32),
                    clean_mask=clean_mask.astype(np.uint8),
                    distorted_mask=distorted_mask.astype(np.uint8),
                )

                rows.append([
                    sample_id,
                    str(image_path),
                    str(mask_path),
                    slice_idx,
                    aug_idx,
                    str(clean_pa_path),
                    str(distorted_pa_path),
                    str(clean_mask_path),
                    str(distorted_mask_path),
                    str(shift_path),
                    int(clean_mask.sum()),
                    float(np.max(np.abs(displacement))),
                ])

                print(f"[Saved] {sample_id}")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "source_image_path",
            "source_mask_path",
            "slice_idx",
            "aug_idx",
            "clean_pa_path",
            "distorted_pa_path",
            "clean_mask_path",
            "distorted_mask_path",
            "shift_path",
            "vessel_pixels",
            "max_abs_shift",
        ])
        writer.writerows(rows)

    print(f"\nDone. Total samples: {len(rows)}")
    print(f"Manifest saved to: {manifest_path}")


# ============================================================
# Example usage
# ============================================================

if False:  # Legacy image/mask-pair example retained for reference.
    # 예시 경로
    IMAGE_DIR = Path("data/other_modality/images")
    MASK_DIR = Path("data/other_modality/masks")
    OUTPUT_DIR = Path("data/pseudo_pa_dataset")

    cfg = PhantomConfig(
        n_aug_per_slice=3,
        min_vessel_pixels=50,
        random_seed=42,

        # 처음에는 원본 CT/MRI intensity를 거의 안 섞는 것이 안전합니다.
        use_input_as_background=False,

        # 왜곡 생성
        generate_distortion=True,
        max_abs_shift=8,
    )

    build_dataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_DIR,
        cfg=cfg,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Generate clean pseudo-photoacoustic phantom patches from non-RSOM "
            "TubeNet label TIFFs. No geometric distortion is applied."
        )
    )
    parser.add_argument(
        "--mask-dir",
        default=str(root / "data" / "tubenet data" / "Labels"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "data" / "tubenet_non_rsom_pa_phantoms"),
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patches-per-slice", type=int, default=4)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--min-vessel-pixels", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-label-files",
        type=int,
        default=0,
        help="0 uses every non-RSOM label file; positive values are useful for tests.",
    )
    parser.add_argument(
        "--max-slices-per-file",
        type=int,
        default=0,
        help="0 uses every slice; positive values are useful for tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list eligible non-RSOM label volumes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size < 16:
        raise ValueError("--patch-size must be >= 16.")
    if args.patches_per_slice < 0 or args.augmentations < 1:
        raise ValueError("patches-per-slice must be >= 0 and augmentations >= 1.")
    if args.min_vessel_pixels < 1:
        raise ValueError("--min-vessel-pixels must be >= 1.")

    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    inventory = inspect_mask_files(mask_dir)
    print("Eligible non-RSOM TubeNet label volumes:")
    for item in inventory:
        print(
            f"  {item['file']}: shape={item['shape']}, "
            f"axes={item['axes']}, dtype={item['dtype']}"
        )
    if args.dry_run:
        print(f"Dry run complete: {len(inventory)} label volumes.")
        return

    cfg = PhantomConfig(
        n_aug_per_slice=args.augmentations,
        min_vessel_pixels=args.min_vessel_pixels,
        random_seed=args.seed,
        patch_size=args.patch_size,
        patches_per_slice=args.patches_per_slice,
        random_morphology=False,
        use_input_as_background=False,
        generate_distortion=False,
    )
    build_dataset_from_non_rsom_masks(
        mask_dir=mask_dir,
        output_dir=output_dir,
        cfg=cfg,
        max_label_files=args.max_label_files,
        max_slices_per_file=args.max_slices_per_file,
    )


if __name__ == "__main__":
    main()
