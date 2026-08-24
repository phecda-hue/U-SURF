from __future__ import annotations

"""
MAC-Net-inspired A-line motion correction for multi-page TIFF phantom data.

Paper reference:
Zheng Sun et al., "A Deep Learning Method for Motion Artifact Correction in
Intravascular Photoacoustic Image Sequence," IEEE TMI, 2023.

This is an independent PyTorch reimplementation adapted to column-wise
(A-line) vertical distortion. It is not the authors' official source code.

Main idea
---------
1. Read clean phantom, RSOM, and photoacoustic images.
2. Create strong multi-component high-frequency A-line jitter online.
3. A VGG16-like encoder predicts one vertical correction value per x-column.
4. A differentiable spatial transformer warps only along y.
5. A conditional PatchGAN encourages clean-looking local structure.
6. Paired losses preserve the original vessel geometry.
"""

import argparse
import csv
import math
import random
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Native-resolution 16-bit MAP images collected for the target domain.
    data_root: str = "data/augmented_raw_train_3x"
    phantom_root: str = ""
    rsom_root: str = ""
    pa_root: str = ""
    output_dir: str = "runs/macnet_measured_jitter_256x512"
    resume_checkpoint: str = ""

    image_height: int = 256
    image_width: int = 512

    epochs: int = 150
    pretrain_epochs: int = 20
    batch_size: int = 4
    num_workers: int = 4

    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999

    val_fraction: float = 0.15
    split_seed: int = 42
    train_seed: int = 1234

    # Observed oscillation is normally 3--4 px. Sparse 2-column spikes can
    # jump as far as 10 px and immediately return to the oscillatory baseline.
    max_distortion_px: float = 4.0
    tail_max_distortion_px: float = 10.0
    tail_probability: float = 0.08
    identity_probability: float = 0.05
    jitter_profile: str = "smooth_v2"

    # Abrupt pulse trains: each displacement lasts 2--3 columns and consecutive
    # pulses are separated by 2--3 columns. Up to four trains overlap per x.
    min_jitter_width_px: int = 2
    max_jitter_width_px: int = 3
    min_jitter_gap_px: int = 2
    max_jitter_gap_px: int = 3
    min_jitter_std_px: float = 2.0
    max_jitter_std_px: float = 3.0
    min_jitter_components: int = 2
    max_jitter_components: int = 4

    # Optional use of the measured high_frequency_jitter.npy profile. Keep the
    # probability below 0.5 because one estimated profile can contain anatomy.
    empirical_jitter_path: str = ""
    empirical_jitter_probability: float = 0.25

    # Training-only clean-image augmentation. Distortion is applied afterward.
    augment_probability: float = 0.8

    # 0=no balancing, 1=equal source probabilities. 0.75 prevents the much
    # larger phantom set from overwhelming RSOM/PA without fully equalizing.
    source_balance_power: float = 0.75

    # Generator losses
    # Shift supervision is primary. Image/identity losses remain auxiliary so
    # predicting zero cannot win merely by preserving most unchanged pixels.
    lambda_l1: float = 30.0
    lambda_ssim: float = 5.0
    lambda_gradient: float = 5.0
    lambda_shift: float = 30.0
    lambda_shift_edge: float = 15.0
    lambda_identity: float = 1.0
    lambda_adversarial: float = 0.5

    save_every: int = 50
    preview_every: int = 5
    jitter_preview_count: int = 8
    patience: int = 30

    device: str = "auto"


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    page_index: int
    source: str
    group: str


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def robust_normalize_01(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)

    values = image[finite]
    low = float(np.percentile(values, 0.5))
    high = float(np.percentile(values, 99.5))

    if high - low < 1e-8:
        low = float(values.min())
        high = float(values.max())

    if high - low < 1e-8:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def list_tiff_files(root: Path) -> List[Path]:
    files = sorted(root.rglob("*.tif")) + sorted(root.rglob("*.tiff"))
    # Remove duplicates if the filesystem is case-insensitive.
    return sorted(set(files))


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def list_image_files(root: Path) -> List[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def count_tiff_pages(path: Path) -> int:
    with tifffile.TiffFile(str(path)) as tif:
        return len(tif.pages)


def read_tiff_page(path: Path, page_index: int) -> np.ndarray:
    image = tifffile.imread(str(path), key=page_index)
    image = np.asarray(image)

    if image.ndim == 3:
        # Handle RGB/RGBA or singleton channel.
        if image.shape[-1] in (3, 4):
            image = cv2.cvtColor(
                image[..., :3].astype(np.float32),
                cv2.COLOR_RGB2GRAY,
            )
        elif image.shape[0] == 1:
            image = image[0]
        else:
            raise ValueError(
                f"Expected a 2-D TIFF page, but got shape {image.shape} "
                f"from {path}, page {page_index}."
            )

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2-D TIFF page, but got shape {image.shape} "
            f"from {path}, page {page_index}."
        )

    return image


def read_image_page(path: Path, page_index: int = 0) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        return read_tiff_page(path, page_index)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got {image.shape}: {path}")
    return image


def resize_image(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA if (
            image.shape[0] > height or image.shape[1] > width
        ) else cv2.INTER_LINEAR,
    ).astype(np.float32)


def resize_height_and_crop_width(
    image: np.ndarray,
    height: int,
    crop_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resize only y, then take consecutive A-lines without x downsampling."""
    interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (image.shape[1], height),
        interpolation=interpolation,
    ).astype(np.float32)

    width = resized.shape[1]
    if width > crop_width:
        start = int(rng.integers(0, width - crop_width + 1))
        return np.ascontiguousarray(resized[:, start:start + crop_width])
    if width < crop_width:
        # A narrow source cannot provide 512 distinct A-lines. Upscale only
        # in this fallback case so all batches retain a fixed tensor shape.
        return cv2.resize(
            resized,
            (crop_width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    return np.ascontiguousarray(resized)


def _records_from_image_root(root: Path, source: str) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for path in list_image_files(root):
        relative = path.relative_to(root)
        if source == "extra":
            # raw_train is mostly flat and contains multiple ImageN exports
            # from the same acquisition. Keep a complete acquisition in one
            # split to prevent near-identical frames leaking into validation.
            stem = re.sub(r"_index\d+$", "", path.stem, flags=re.IGNORECASE)
            stem = re.sub(r"_Image\d+$", "", stem, flags=re.IGNORECASE)
            stem = re.sub(r"\s*\(\d+\)$", "", stem)
            group_name = str(relative.parent / stem.strip())
        elif source == "pa":
            # The official train set is organized as patch_1, patch_2, ...;
            # keep a whole patch directory in one split.
            group_name = relative.parts[0] if len(relative.parts) > 1 else path.stem
        elif source == "rsom" and path.stem.isdigit():
            # Consecutive RSOM slices are highly correlated. Keep blocks of 20
            # adjacent slices together to reduce train/validation leakage.
            group_name = f"slice_block_{int(path.stem) // 20:04d}"
        else:
            group_name = str(relative.parent / path.stem)

        page_count = (
            count_tiff_pages(path)
            if path.suffix.lower() in {".tif", ".tiff"}
            else 1
        )
        records.extend(
            SampleRecord(path, page, source, f"{source}:{group_name}")
            for page in range(page_count)
        )
    return records


def _records_from_phantom_root(root: Path) -> List[SampleRecord]:
    manifest = root / "manifest.csv"
    clean_root = root / "clean_pa"
    if not manifest.exists() and root.name.lower() == "clean_pa":
        manifest = root.parent / "manifest.csv"
        clean_root = root

    records: List[SampleRecord] = []
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                path = Path(row["clean_pa_path"])
                if not path.exists():
                    path = clean_root / path.name
                if not path.exists():
                    continue

                modality = row.get("source_modality", "unknown")
                try:
                    slice_block = int(row.get("slice_index", 0)) // 10
                except ValueError:
                    slice_block = 0
                # All patches/augmentations from nearby source slices stay in
                # the same split, preventing near-duplicate leakage.
                group = f"phantom:{modality}:zblock_{slice_block:04d}"
                records.append(SampleRecord(path, 0, "phantom", group))
        return records

    if clean_root.exists():
        return _records_from_image_root(clean_root, "phantom")
    return _records_from_image_root(root, "phantom")


def collect_training_records(config: TrainConfig) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    configured_roots = [
        (config.phantom_root, "phantom"),
        (config.rsom_root, "rsom"),
        (config.pa_root, "pa"),
        (config.data_root, "extra"),
    ]

    for root_text, source in configured_roots:
        if not root_text:
            continue
        root = Path(root_text)
        if not root.exists():
            print(f"Warning: skipping missing {source} root: {root}")
            continue
        current = (
            _records_from_phantom_root(root)
            if source == "phantom"
            else _records_from_image_root(root, source)
        )
        print(f"Found {len(current):,} {source} image/page samples: {root}")
        records.extend(current)

    # Several raw exports are byte-for-byte duplicates under different names.
    # Removing them avoids overweighting those acquisitions and prevents an
    # exact copy from appearing in both train and validation.
    unique_records: List[SampleRecord] = []
    seen_hashes: set[str] = set()
    duplicate_count = 0
    for record in records:
        if record.source == "extra" and record.page_index == 0:
            digest = hashlib.md5(record.path.read_bytes()).hexdigest()
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(digest)
        unique_records.append(record)
    if duplicate_count:
        print(f"Skipped {duplicate_count:,} duplicate raw_train images.")
    records = unique_records

    if not records:
        raise FileNotFoundError("No supported clean images were found in any data root.")
    return records


# ---------------------------------------------------------------------
# Synthetic A-line distortion
# ---------------------------------------------------------------------

def load_empirical_jitter_profiles(path_text: str) -> np.ndarray | None:
    """Load one or more measured 1-D jitter profiles from a .npy file.

    Accepted shapes are [W] and [N, W]. Each profile is centered independently.
    The measured profile is optional because it is an estimate, not ground truth.
    """
    if not path_text:
        return None

    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Empirical jitter profile was not found: {path}")

    profiles = np.load(path, allow_pickle=False)
    profiles = np.asarray(profiles, dtype=np.float32)
    if profiles.ndim == 1:
        profiles = profiles[None, :]
    elif profiles.ndim != 2:
        raise ValueError(
            "Empirical jitter array must have shape [W] or [N, W], "
            f"but got {profiles.shape}."
        )

    cleaned: List[np.ndarray] = []
    for profile in profiles:
        finite = np.isfinite(profile)
        if finite.sum() < 4:
            continue
        current = profile.copy()
        if not finite.all():
            valid_x = np.flatnonzero(finite)
            current[~finite] = np.interp(
                np.flatnonzero(~finite), valid_x, current[finite]
            )
        current -= np.median(current)
        if float(np.std(current)) > 1e-6:
            cleaned.append(current.astype(np.float32))

    if not cleaned:
        raise ValueError(f"No usable empirical jitter profile was found in {path}.")

    lengths = {len(profile) for profile in cleaned}
    if len(lengths) != 1:
        raise ValueError("All empirical jitter profiles must have the same length.")
    return np.stack(cleaned, axis=0)


def sample_empirical_jitter_field(
    profiles: np.ndarray,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly crop/stretch/flip a measured jitter profile to ``width``."""
    profile = profiles[int(rng.integers(0, len(profiles)))].astype(np.float32)

    # A small horizontal stretch changes the period without destroying the
    # measured autocorrelation structure.
    source_width = max(8, int(round(width * rng.uniform(0.85, 1.15))))
    if len(profile) >= source_width:
        start = int(rng.integers(0, len(profile) - source_width + 1))
        segment = profile[start:start + source_width]
    else:
        repeats = int(math.ceil(source_width / len(profile)))
        segment = np.tile(profile, repeats)[:source_width]

    source_x = np.linspace(0.0, 1.0, len(segment), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    field = np.interp(target_x, source_x, segment).astype(np.float32)

    if rng.random() < 0.5:
        field = field[::-1].copy()
    if rng.random() < 0.5:
        field = -field
    field -= np.median(field)
    return field.astype(np.float32)


def sample_block_jitter_basis(
    width: int,
    rng: np.random.Generator,
    min_width_px: int,
    max_width_px: int,
    min_gap_px: int,
    max_gap_px: int,
) -> np.ndarray:
    """Create sparse local episodes of abrupt displacement blocks."""
    field = np.zeros(width, dtype=np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cursor = int(rng.integers(0, max(width - min_width_px + 1, 1)))
        sign = float(rng.choice((-1.0, 1.0)))
        for _ in range(int(rng.integers(2, 9))):
            pulse_width = int(rng.integers(min_width_px, max_width_px + 1))
            if cursor + pulse_width > width:
                break
            stop = cursor + pulse_width
            # Alternation produces the observed up/down vibration, while an
            # occasional repeated sign avoids a perfectly periodic square wave.
            if rng.random() < 0.75:
                sign = -sign
            else:
                sign = float(rng.choice((-1.0, 1.0)))
            field[cursor:stop] = sign * float(rng.uniform(0.65, 1.0))
            gap = int(rng.integers(min_gap_px, max_gap_px + 1))
            cursor = stop + gap
    return field.astype(np.float32)


def sample_smooth_v2_jitter_field(
    width: int,
    rng: np.random.Generator,
    max_abs_shift: float = 3.0,
    tail_max_abs_shift: float = 10.0,
    tail_probability: float = 0.08,
    identity_probability: float = 0.05,
) -> np.ndarray:
    """Synthetic-v2 ablation matched to the measured real-jitter spectrum.

    Most motion is a locally activated, amplitude/frequency-modulated 20--50 px
    oscillation below 3 px. Smooth stochastic displacement is mixed in, while
    rare localized events may reach 3--10 px.
    """
    if rng.random() < identity_probability:
        return np.zeros(width, dtype=np.float32)

    x = np.arange(width, dtype=np.float32)
    field = np.zeros(width, dtype=np.float32)
    component_count = int(rng.integers(1, 4))
    for _ in range(component_count):
        control_count = int(rng.integers(4, 9))
        control_x = np.linspace(0, width - 1, control_count, dtype=np.float32)
        periods = rng.uniform(20.0, 50.0, control_count).astype(np.float32)
        amplitudes = rng.uniform(0.25, 1.0, control_count).astype(np.float32)
        period = np.interp(x, control_x, periods).astype(np.float32)
        amplitude = np.interp(x, control_x, amplitudes).astype(np.float32)
        phase = np.cumsum(2.0 * math.pi / period, dtype=np.float32)
        phase += float(rng.uniform(0.0, 2.0 * math.pi))

        # Activate each component only in one or two broad local intervals.
        envelope = np.zeros(width, dtype=np.float32)
        for _ in range(int(rng.integers(1, 3))):
            span = int(rng.integers(max(24, width // 8), max(25, width // 2 + 1)))
            start = int(rng.integers(0, max(width - span + 1, 1)))
            stop = min(width, start + span)
            ramp = min(int(rng.integers(6, 17)), max((stop - start) // 2, 1))
            envelope[start:stop] = 1.0
            if ramp > 1:
                fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                envelope[start:start + ramp] *= fade
                envelope[stop - ramp:stop] *= fade[::-1]
        field += amplitude * envelope * np.sin(phase)

    # Broad stochastic component without introducing anatomical-scale drift.
    noise = rng.normal(0.0, 1.0, width).astype(np.float32)
    sigma = float(rng.uniform(6.0, 16.0))
    smooth_noise = cv2.GaussianBlur(
        noise[None, :], (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101
    )[0]
    smooth_noise -= np.median(smooth_noise)
    smooth_noise /= max(float(np.std(smooth_noise)), 1e-6)
    field += float(rng.uniform(0.10, 0.45)) * smooth_noise
    field -= np.median(field)

    # Small displacements dominate the distribution.
    draw = float(rng.random())
    if draw < 0.70:
        target_peak = float(rng.uniform(0.3, 1.5))
    elif draw < 0.95:
        target_peak = float(rng.uniform(1.5, max_abs_shift))
    else:
        target_peak = max_abs_shift
    peak = max(float(np.max(np.abs(field))), 1e-6)
    field *= target_peak / peak

    if rng.random() < tail_probability:
        center = float(rng.uniform(0, width - 1))
        sigma = float(rng.uniform(2.0, 8.0))
        magnitude = float(rng.uniform(max_abs_shift, tail_max_abs_shift))
        sign = float(rng.choice((-1.0, 1.0)))
        field += sign * magnitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    field -= np.median(field)
    return np.clip(field, -tail_max_abs_shift, tail_max_abs_shift).astype(np.float32)


def sample_jitter_field(
    width: int,
    max_abs_shift: float,
    tail_max_abs_shift: float,
    tail_probability: float,
    rng: np.random.Generator,
    identity_probability: float = 0.15,
    min_width_px: int = 2,
    max_width_px: int = 3,
    min_gap_px: int = 2,
    max_gap_px: int = 3,
    min_std_px: float = 2.0,
    max_std_px: float = 3.0,
    min_components: int = 2,
    max_components: int = 4,
    empirical_profiles: np.ndarray | None = None,
    empirical_probability: float = 0.25,
) -> np.ndarray:
    """Generate strong, composite, correlated A-line jitter.

    Each field superimposes two to four abrupt block-pulse trains and clips their
    normal amplitude to ``max_abs_shift``. Some samples also receive one to
    three upward, 2-column spikes reaching ``tail_max_abs_shift``.
    """
    if rng.random() < identity_probability:
        return np.zeros(width, dtype=np.float32)

    if max_abs_shift <= 0:
        raise ValueError("max_abs_shift must be positive.")
    if tail_max_abs_shift < max_abs_shift:
        raise ValueError("tail_max_abs_shift must be >= max_abs_shift.")
    if not 0.0 <= tail_probability <= 1.0:
        raise ValueError("tail_probability must be between 0 and 1.")
    if not 0.0 <= identity_probability <= 1.0:
        raise ValueError("identity_probability must be between 0 and 1.")
    if min_width_px < 1 or max_width_px < min_width_px:
        raise ValueError("Require 1 <= min_width_px <= max_width_px.")
    if min_gap_px < 0 or max_gap_px < min_gap_px:
        raise ValueError("Require 0 <= min_gap_px <= max_gap_px.")
    if min_std_px <= 0 or max_std_px < min_std_px:
        raise ValueError("Require 0 < min_std_px <= max_std_px.")
    if min_components < 1 or max_components < min_components:
        raise ValueError("Require 1 <= min_components <= max_components.")
    if not 0.0 <= empirical_probability <= 1.0:
        raise ValueError("empirical_probability must be between 0 and 1.")

    use_empirical = (
        empirical_profiles is not None
        and rng.random() < empirical_probability
    )
    component_count = int(rng.integers(min_components, max_components + 1))
    components: List[np.ndarray] = []
    if use_empirical:
        components.append(sample_empirical_jitter_field(empirical_profiles, width, rng))

    while len(components) < component_count:
        components.append(sample_block_jitter_basis(
            width,
            rng,
            min_width_px,
            max_width_px,
            min_gap_px,
            max_gap_px,
        ))

    # Normalize each basis before weighting so no accidental high-variance basis
    # dominates. All components still overlap and add at every x coordinate.
    field = np.zeros(width, dtype=np.float32)
    for component in components:
        component = component - np.median(component)
        component /= max(float(np.std(component)), 1e-6)
        field += float(rng.uniform(0.65, 1.0)) * component

    field -= np.median(field)
    target_std = float(rng.uniform(min_std_px, max_std_px))
    field *= target_std / max(float(np.std(field)), 1e-6)

    # Normal oscillation is limited to the observed +/-4 px range.
    field = np.clip(field, -max_abs_shift, max_abs_shift)

    # A spike replaces (rather than adds to) the oscillatory value at exactly
    # one or two columns, then immediately returns to the baseline waveform.
    if tail_max_abs_shift > max_abs_shift and rng.random() < tail_probability:
        occupied = np.zeros(width, dtype=bool)
        for _ in range(int(rng.integers(1, 4))):
            spike_width = 2
            candidates = [
                start for start in range(width - spike_width + 1)
                if not occupied[
                    max(0, start - 1):min(width, start + spike_width + 1)
                ].any()
            ]
            if not candidates:
                break
            start = int(candidates[int(rng.integers(0, len(candidates)))])
            magnitude = float(rng.uniform(
                max_abs_shift * 1.05,
                tail_max_abs_shift,
            ))
            # Negative distortion moves the whole x-column upward.
            field[start:start + spike_width] = -magnitude
            occupied[start:start + spike_width] = True

    field = np.clip(field, -tail_max_abs_shift, tail_max_abs_shift)
    return field.astype(np.float32)

def warp_columns_numpy(
    image: np.ndarray,
    shift_px: np.ndarray,
    border_value: float | None = None,
) -> np.ndarray:
    """
    Column-wise vertical spatial transform.

    output[y, x] = input[y - shift_px[x], x]

    Therefore positive shift moves content downward.
    """
    image = image.astype(np.float32)
    height, width = image.shape

    if shift_px.shape != (width,):
        raise ValueError(
            f"shift_px must have shape ({width},), got {shift_px.shape}."
        )

    y = np.arange(height, dtype=np.float32)
    output = np.empty_like(image, dtype=np.float32)

    for x in range(width):
        source_y = y - float(shift_px[x])
        output[:, x] = np.interp(
            source_y,
            y,
            image[:, x],
            left=image[0, x] if border_value is None else border_value,
            right=image[-1, x] if border_value is None else border_value,
        )

    return output


# ---------------------------------------------------------------------
# Mixed clean-image dataset
# ---------------------------------------------------------------------

def augment_clean_image(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> np.ndarray:
    """Mild acquisition-style augmentation applied before distortion."""
    if probability <= 0 or rng.random() >= probability:
        return image.astype(np.float32)

    augmented = image.astype(np.float32).copy()
    if rng.random() < 0.5:
        augmented = np.ascontiguousarray(augmented[:, ::-1])

    gamma = float(rng.uniform(0.85, 1.20))
    gain = float(rng.uniform(0.85, 1.15))
    augmented = gain * np.power(np.clip(augmented, 0.0, 1.0), gamma)

    if rng.random() < 0.35:
        sigma = float(rng.uniform(0.002, 0.015))
        augmented += rng.normal(0.0, sigma, augmented.shape).astype(np.float32)
    return np.clip(augmented, 0.0, 1.0).astype(np.float32)


def native_random_crop(
    image: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Crop MAP data without changing its native A-line/pixel spacing."""
    pad_y = max(0, height - image.shape[0])
    pad_x = max(0, width - image.shape[1])
    if pad_y or pad_x:
        image = np.pad(
            image,
            ((pad_y // 2, pad_y - pad_y // 2),
             (pad_x // 2, pad_x - pad_x // 2)),
            mode="reflect",
        )
    top = int(rng.integers(0, image.shape[0] - height + 1))
    left = int(rng.integers(0, image.shape[1] - width + 1))
    return np.ascontiguousarray(image[top:top + height, left:left + width])


class MixedPhotoacousticDataset(Dataset):
    """
    Clean images are augmented (training only), then distorted online.

    The target is always the clean image immediately before mixed distortion;
    masks are intentionally not used for this image-restoration task.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        image_height: int,
        image_width: int,
        max_distortion_px: float,
        tail_max_distortion_px: float,
        tail_probability: float,
        identity_probability: float,
        jitter_profile: str,
        min_jitter_width_px: int,
        max_jitter_width_px: int,
        min_jitter_gap_px: int,
        max_jitter_gap_px: int,
        min_jitter_std_px: float,
        max_jitter_std_px: float,
        min_jitter_components: int,
        max_jitter_components: int,
        empirical_jitter_path: str,
        empirical_jitter_probability: float,
        augment_probability: float,
        base_seed: int,
    ) -> None:
        self.records = list(records)
        self.image_height = image_height
        self.image_width = image_width
        self.max_distortion_px = max_distortion_px
        self.tail_max_distortion_px = tail_max_distortion_px
        self.tail_probability = tail_probability
        self.identity_probability = identity_probability
        self.jitter_profile = jitter_profile
        self.min_jitter_width_px = min_jitter_width_px
        self.max_jitter_width_px = max_jitter_width_px
        self.min_jitter_gap_px = min_jitter_gap_px
        self.max_jitter_gap_px = max_jitter_gap_px
        self.min_jitter_std_px = min_jitter_std_px
        self.max_jitter_std_px = max_jitter_std_px
        self.min_jitter_components = min_jitter_components
        self.max_jitter_components = max_jitter_components
        self.empirical_jitter_probability = empirical_jitter_probability
        self.empirical_jitter_profiles = load_empirical_jitter_profiles(
            empirical_jitter_path
        )
        self.augment_probability = augment_probability
        self.base_seed = base_seed
        self.epoch = 0

        if not self.records:
            raise RuntimeError("No clean image records were provided.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | int]:
        record = self.records[index]

        # Deterministic for the same epoch/index, different across epochs.
        rng_seed = (
            self.base_seed
            + self.epoch * 1_000_003
            + index * 97
            + record.page_index * 17
        )
        rng = np.random.default_rng(rng_seed)

        clean = read_image_page(record.path, record.page_index)
        clean = robust_normalize_01(clean)
        if record.source == "extra":
            clean = native_random_crop(
                clean, self.image_height, self.image_width, rng
            )
        else:
            clean = resize_height_and_crop_width(
                clean, self.image_height, self.image_width, rng
            )
        clean = augment_clean_image(clean, rng, self.augment_probability)

        if self.jitter_profile == "smooth_v2":
            distortion_shift = sample_smooth_v2_jitter_field(
                width=self.image_width,
                rng=rng,
                max_abs_shift=min(self.max_distortion_px, 3.0),
                tail_max_abs_shift=self.tail_max_distortion_px,
                tail_probability=self.tail_probability,
                identity_probability=self.identity_probability,
            )
        elif self.jitter_profile == "block_v1":
            distortion_shift = sample_jitter_field(
                width=self.image_width,
                max_abs_shift=self.max_distortion_px,
                tail_max_abs_shift=self.tail_max_distortion_px,
                tail_probability=self.tail_probability,
                rng=rng,
                identity_probability=self.identity_probability,
                min_width_px=self.min_jitter_width_px,
                max_width_px=self.max_jitter_width_px,
                min_gap_px=self.min_jitter_gap_px,
                max_gap_px=self.max_jitter_gap_px,
                min_std_px=self.min_jitter_std_px,
                max_std_px=self.max_jitter_std_px,
                min_components=self.min_jitter_components,
                max_components=self.max_jitter_components,
                empirical_profiles=self.empirical_jitter_profiles,
                empirical_probability=self.empirical_jitter_probability,
            )
        else:
            raise ValueError(f"Unknown jitter_profile: {self.jitter_profile}")

        distorted = warp_columns_numpy(clean, distortion_shift)

        # The generator predicts the transform applied to distorted input.
        correction_shift = -distortion_shift

        return {
            "distorted": torch.from_numpy(distorted[None]).float(),
            "clean": torch.from_numpy(clean[None]).float(),
            "correction_shift": torch.from_numpy(correction_shift).float(),
            "path": str(record.path),
            "page_index": int(record.page_index),
            "source": record.source,
        }


def split_records_by_group(
    records: Sequence[SampleRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    rng = random.Random(seed)
    by_source: Dict[str, Dict[str, List[SampleRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_source[record.source][record.group].append(record)

    train_records: List[SampleRecord] = []
    val_records: List[SampleRecord] = []
    for source, grouped in sorted(by_source.items()):
        groups = sorted(grouped)
        rng.shuffle(groups)
        if len(groups) < 2:
            print(
                f"Warning: source '{source}' has one group; using it only "
                "for training."
            )
            train_records.extend(grouped[groups[0]])
            continue

        n_val = max(1, int(round(len(groups) * val_fraction)))
        n_val = min(n_val, len(groups) - 1)
        val_groups = set(groups[:n_val])
        for group, group_records in grouped.items():
            (val_records if group in val_groups else train_records).extend(
                group_records
            )

    if not train_records or not val_records:
        raise RuntimeError("The group-level split produced an empty train or val set.")
    return sorted(train_records, key=lambda x: str(x.path)), sorted(
        val_records, key=lambda x: str(x.path)
    )


def make_source_balanced_sampler(
    records: Sequence[SampleRecord],
    balance_power: float,
    seed: int,
) -> WeightedRandomSampler | None:
    if balance_power <= 0:
        return None
    if balance_power > 1:
        raise ValueError("source_balance_power must be in [0, 1].")
    counts = Counter(record.source for record in records)
    weights = [counts[record.source] ** (-balance_power) for record in records]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


# ---------------------------------------------------------------------
# Differentiable column-wise spatial transformer
# ---------------------------------------------------------------------

def column_spatial_transform(
    image: torch.Tensor,
    shift_px: torch.Tensor,
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    image:    [B, C, H, W]
    shift_px: [B, W]

    output[y, x] = input[y - shift_px[x], x]
    """
    if image.ndim != 4:
        raise ValueError(f"image must be BCHW, got {image.shape}.")

    batch, _, height, width = image.shape

    if shift_px.shape != (batch, width):
        raise ValueError(
            f"shift_px must be {(batch, width)}, got {tuple(shift_px.shape)}."
        )

    dtype = image.dtype
    device = image.device

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)

    if height > 1:
        shift_norm = 2.0 * shift_px / float(height - 1)
    else:
        shift_norm = torch.zeros_like(shift_px)

    source_y = grid_y - shift_norm[:, None, :]
    grid = torch.stack([grid_x, source_y], dim=-1)

    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


# ---------------------------------------------------------------------
# MAC-Net-inspired generator
# ---------------------------------------------------------------------

class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_count: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(conv_count):
            layers.extend([
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.features = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(x)
        pooled = self.pool(feature)
        return feature, pooled


class MACNetColumnGenerator(nn.Module):
    """
    VGG16-like feature encoder + 1-D deformation head + spatial transformer.

    Unlike a conventional image generator, the network cannot freely invent
    vessel pixels. It predicts only one vertical correction per A-line and
    corrects the input through grid_sample().
    """

    def __init__(
        self,
        in_channels: int = 1,
        max_correction_px: float = 12.0,
    ) -> None:
        super().__init__()
        self.max_correction_px = float(max_correction_px)

        channels = [64, 128, 256, 512, 512]
        conv_counts = [2, 2, 3, 3, 3]

        self.blocks = nn.ModuleList()
        current = in_channels
        for out_channels, conv_count in zip(channels, conv_counts):
            self.blocks.append(VGGBlock(current, out_channels, conv_count))
            current = out_channels

        # Each VGG scale is compressed to a 1-D x-axis feature.
        self.scale_projections = nn.ModuleList([
            nn.Conv1d(ch, 64, kernel_size=1) for ch in channels
        ])

        self.shift_head = nn.Sequential(
            nn.Conv1d(64 * len(channels), 256, kernel_size=7, padding=3),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.InstanceNorm1d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.InstanceNorm1d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 1, kernel_size=5, padding=2),
        )

        # Start close to identity transformation.
        last_conv = self.shift_head[-1]
        nn.init.zeros_(last_conv.weight)
        nn.init.zeros_(last_conv.bias)

    def predict_shift(self, image: torch.Tensor) -> torch.Tensor:
        width = image.shape[-1]
        x = image
        scale_features: List[torch.Tensor] = []

        for block, projection in zip(self.blocks, self.scale_projections):
            feature, x = block(x)

            # Average over depth y, retaining the x-axis A-line sequence.
            feature_1d = feature.mean(dim=2)
            feature_1d = projection(feature_1d)
            feature_1d = F.interpolate(
                feature_1d,
                size=width,
                mode="linear",
                align_corners=True,
            )
            scale_features.append(feature_1d)

        fused = torch.cat(scale_features, dim=1)
        raw_shift = self.shift_head(fused).squeeze(1)
        return torch.tanh(raw_shift) * self.max_correction_px

    def forward(
        self,
        distorted: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        correction_shift = self.predict_shift(distorted)
        corrected = column_spatial_transform(distorted, correction_shift)
        return corrected, correction_shift


# ---------------------------------------------------------------------
# Conditional PatchGAN discriminator
# ---------------------------------------------------------------------

def discriminator_block(
    in_channels: int,
    out_channels: int,
    stride: int,
    normalize: bool,
) -> List[nn.Module]:
    layers: List[nn.Module] = [
        nn.utils.spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not normalize,
            )
        )
    ]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class ConditionalPatchDiscriminator(nn.Module):
    def __init__(self, image_channels: int = 1) -> None:
        super().__init__()
        in_channels = image_channels * 2

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, stride=2, normalize=False),
            *discriminator_block(64, 128, stride=2, normalize=True),
            *discriminator_block(128, 256, stride=2, normalize=True),
            *discriminator_block(256, 512, stride=1, normalize=True),
            nn.utils.spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            ),
        )

    def forward(
        self,
        distorted: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(torch.cat([distorted, candidate], dim=1))


# ---------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------

def image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    return grad_x, grad_y


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y = image_gradients(prediction)
    target_x, target_y = image_gradients(target)
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


def ssim_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, 1, padding)
    mu_y = F.avg_pool2d(target, window_size, 1, padding)

    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y ** 2
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / (denominator + 1e-8)).mean()


def weighted_shift_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normal_shift_px: float,
) -> torch.Tensor:
    """Emphasize displaced columns and especially large spike columns."""
    error = F.smooth_l1_loss(prediction, target, beta=0.25, reduction="none")
    relative_magnitude = target.abs() / max(float(normal_shift_px), 1e-6)
    weights = 1.0 + 3.0 * relative_magnitude.clamp(max=2.5)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def shift_edge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Supervise abrupt column-to-column jumps instead of smoothing them."""
    if prediction.shape[-1] < 2:
        return prediction.new_tensor(0.0)
    predicted_edge = prediction[:, 1:] - prediction[:, :-1]
    target_edge = target[:, 1:] - target[:, :-1]
    error = F.smooth_l1_loss(
        predicted_edge,
        target_edge,
        beta=0.25,
        reduction="none",
    )
    weights = 1.0 + 4.0 * (target_edge.abs() / 4.0).clamp(max=2.5)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


# ---------------------------------------------------------------------
# Preview and checkpoint utilities
# ---------------------------------------------------------------------

@torch.no_grad()
def save_preview(
    output_path: Path,
    distorted: torch.Tensor,
    corrected: torch.Tensor,
    clean: torch.Tensor,
    predicted_shift: torch.Tensor,
    target_shift: torch.Tensor,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = [
        distorted[0, 0].detach().cpu().numpy(),
        corrected[0, 0].detach().cpu().numpy(),
        clean[0, 0].detach().cpu().numpy(),
    ]

    panels = []
    labels = ["distorted", "corrected", "clean target"]
    for image, label_text in zip(images, labels):
        panel = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            panel,
            label_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    image_panel = np.concatenate(panels, axis=1)

    height = 180
    width = images[0].shape[1]
    plot = np.full((height, width, 3), 255, dtype=np.uint8)

    pred = predicted_shift[0].detach().cpu().numpy()
    target = target_shift[0].detach().cpu().numpy()
    limit = max(
        1.0,
        float(np.max(np.abs(pred))),
        float(np.max(np.abs(target))),
    )

    def to_points(values: np.ndarray) -> np.ndarray:
        x = np.linspace(0, width - 1, len(values))
        y = (height / 2.0) - values / limit * (height * 0.4)
        return np.stack([x, y], axis=1).round().astype(np.int32)

    cv2.polylines(plot, [to_points(target)], False, (0, 150, 0), 2)
    cv2.polylines(plot, [to_points(pred)], False, (0, 0, 255), 2)
    cv2.putText(
        plot,
        "green=target correction, red=predicted correction",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    if image_panel.shape[1] != plot.shape[1]:
        plot = cv2.resize(plot, (image_panel.shape[1], height))

    preview = np.concatenate([image_panel, plot], axis=0)
    cv2.imwrite(str(output_path), preview)


@torch.no_grad()
def save_jitter_previews(
    dataset: MixedPhotoacousticDataset,
    output_dir: Path,
    count: int,
) -> None:
    """Save clean/synthetic-jitter pairs before optimization starts."""
    if count <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[List[object]] = []
    saved = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        correction = sample["correction_shift"].numpy()
        distortion = -correction
        peak = float(np.max(np.abs(distortion)))
        jitter_std = float(np.std(distortion))
        jitter_abs_p95 = float(np.percentile(np.abs(distortion), 95.0))
        centered = distortion - float(np.mean(distortion))
        spectrum = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(len(centered))
        valid_frequency = (frequencies >= 1.0 / 128.0) & (frequencies <= 1.0 / 4.0)
        if np.any(valid_frequency):
            valid_indices = np.flatnonzero(valid_frequency)
            dominant_index = valid_indices[int(np.argmax(spectrum[valid_frequency]))]
            dominant_period = float(1.0 / frequencies[dominant_index])
        else:
            dominant_period = float("nan")
        # Do not spend the limited preview slots on identity samples.
        if peak < 1e-6:
            continue

        clean = sample["clean"][0].numpy()
        distorted = sample["distorted"][0].numpy()
        difference = np.abs(distorted - clean)
        difference /= max(float(difference.max()), 1e-8)

        panels: List[np.ndarray] = []
        for image, label in (
            (clean, "clean crop"),
            (distorted, f"jittered (peak={peak:.2f}px)"),
            (difference, "absolute difference (scaled)"),
        ):
            panel = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            panels.append(panel)
        image_panel = np.concatenate(panels, axis=1)

        plot_height = 180
        plot = np.full(
            (plot_height, image_panel.shape[1], 3), 255, dtype=np.uint8
        )
        x = np.linspace(0, image_panel.shape[1] - 1, len(distortion))
        y = plot_height / 2.0 - distortion / max(peak, 1.0) * (plot_height * 0.4)
        points = np.stack([x, y], axis=1).round().astype(np.int32)
        cv2.line(
            plot, (0, plot_height // 2),
            (image_panel.shape[1] - 1, plot_height // 2), (200, 200, 200), 1,
        )
        cv2.polylines(plot, [points], False, (255, 0, 0), 2)
        cv2.putText(
            plot, "blue=applied vertical distortion shift (pixels)",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

        stem = f"sample_{saved + 1:02d}"
        cv2.imwrite(str(output_dir / f"{stem}.png"), np.concatenate([
            image_panel, plot
        ], axis=0))
        np.save(output_dir / f"{stem}_distortion_shift.npy", distortion)
        manifest_rows.append([
            stem,
            sample["path"],
            sample["page_index"],
            sample["source"],
            float(distortion.min()),
            float(distortion.max()),
            peak,
            jitter_std,
            jitter_abs_p95,
            dominant_period,
        ])
        saved += 1
        if saved >= count:
            break

    if saved < count:
        print(f"Warning: saved only {saved}/{count} non-identity jitter previews.")
    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "sample", "source_path", "page_index", "source",
            "min_distortion_px", "max_distortion_px", "peak_abs_px",
            "std_px", "abs_p95_px", "dominant_period_px",
        ])
        writer.writerows(manifest_rows)
    print(f"Saved {saved} synthetic jitter previews: {output_dir}")


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: TrainConfig,
    best_val_shift_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": asdict(config),
            "best_val_shift_score": best_val_shift_score,
        },
        path,
    )


# ---------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------

def train_one_epoch(
    generator: MACNetColumnGenerator,
    discriminator: ConditionalPatchDiscriminator,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
    use_adversarial: bool,
) -> Dict[str, float]:
    generator.train()
    discriminator.train()

    bce = nn.BCEWithLogitsLoss()
    totals = {
        "g": 0.0,
        "d": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "grad": 0.0,
        "shift": 0.0,
        "shift_edge": 0.0,
        "identity": 0.0,
        "adv": 0.0,
    }
    count = 0

    for batch in loader:
        distorted = batch["distorted"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        target_shift = batch["correction_shift"].to(device, non_blocking=True)

        # -------------------------------------------------------------
        # Discriminator
        # -------------------------------------------------------------
        loss_d = distorted.new_tensor(0.0)

        if use_adversarial:
            with torch.no_grad():
                fake_clean, _ = generator(distorted)

            logits_real = discriminator(distorted, clean)
            logits_fake = discriminator(distorted, fake_clean)

            real_target = torch.full_like(logits_real, 0.9)
            fake_target = torch.zeros_like(logits_fake)

            loss_d = 0.5 * (
                bce(logits_real, real_target)
                + bce(logits_fake, fake_target)
            )

            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

        # -------------------------------------------------------------
        # Generator
        # -------------------------------------------------------------
        corrected, predicted_shift = generator(distorted)

        loss_l1 = F.l1_loss(corrected, clean)
        loss_ssim = 1.0 - ssim_index(corrected, clean)
        loss_grad = gradient_loss(corrected, clean)
        loss_shift = weighted_shift_loss(
            predicted_shift,
            target_shift,
            config.max_distortion_px,
        )
        loss_shift_edge = shift_edge_loss(predicted_shift, target_shift)

        # Static-frame / identity regularization.
        identity_corrected, identity_shift = generator(clean)
        loss_identity = (
            F.l1_loss(identity_corrected, clean)
            + 0.1 * identity_shift.abs().mean()
        )

        loss_adv = corrected.new_tensor(0.0)
        if use_adversarial:
            fake_logits = discriminator(distorted, corrected)
            loss_adv = bce(fake_logits, torch.ones_like(fake_logits))

        loss_g = (
            config.lambda_l1 * loss_l1
            + config.lambda_ssim * loss_ssim
            + config.lambda_gradient * loss_grad
            + config.lambda_shift * loss_shift
            + config.lambda_shift_edge * loss_shift_edge
            + config.lambda_identity * loss_identity
            + config.lambda_adversarial * loss_adv
        )

        optimizer_g.zero_grad(set_to_none=True)
        loss_g.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        optimizer_g.step()

        batch_size = distorted.shape[0]
        count += batch_size
        totals["g"] += float(loss_g.item()) * batch_size
        totals["d"] += float(loss_d.item()) * batch_size
        totals["l1"] += float(loss_l1.item()) * batch_size
        totals["ssim"] += float(loss_ssim.item()) * batch_size
        totals["grad"] += float(loss_grad.item()) * batch_size
        totals["shift"] += float(loss_shift.item()) * batch_size
        totals["shift_edge"] += float(loss_shift_edge.item()) * batch_size
        totals["identity"] += float(loss_identity.item()) * batch_size
        totals["adv"] += float(loss_adv.item()) * batch_size

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(
    generator: MACNetColumnGenerator,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    generator.eval()

    totals = {
        "l1": 0.0,
        "ssim": 0.0,
        "psnr": 0.0,
        "shift_mae": 0.0,
        "weighted_shift": 0.0,
        "shift_edge": 0.0,
    }
    source_l1_sum: Dict[str, float] = defaultdict(float)
    source_count: Counter[str] = Counter()
    count = 0
    preview_batch: Dict[str, torch.Tensor] | None = None

    for batch in loader:
        distorted = batch["distorted"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        target_shift = batch["correction_shift"].to(device, non_blocking=True)

        corrected, predicted_shift = generator(distorted)

        batch_size = distorted.shape[0]
        count += batch_size

        totals["l1"] += float(F.l1_loss(corrected, clean).item()) * batch_size
        totals["ssim"] += float(ssim_index(corrected, clean).item()) * batch_size
        totals["psnr"] += float(psnr(corrected, clean).item()) * batch_size
        totals["shift_mae"] += float(
            F.l1_loss(predicted_shift, target_shift).item()
        ) * batch_size
        totals["weighted_shift"] += float(weighted_shift_loss(
            predicted_shift, target_shift, config.max_distortion_px
        ).item()) * batch_size
        totals["shift_edge"] += float(shift_edge_loss(
            predicted_shift, target_shift
        ).item()) * batch_size

        per_sample_l1 = (corrected - clean).abs().flatten(1).mean(1).cpu().tolist()
        for source, value in zip(batch["source"], per_sample_l1):
            source_l1_sum[str(source)] += float(value)
            source_count[str(source)] += 1

        if preview_batch is None:
            preview_batch = {
                "distorted": distorted,
                "corrected": corrected,
                "clean": clean,
                "predicted_shift": predicted_shift,
                "target_shift": target_shift,
            }

    if preview_batch is None:
        raise RuntimeError("Validation loader returned no batches.")

    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    source_l1 = {
        source: source_l1_sum[source] / source_count[source]
        for source in sorted(source_count)
    }
    # Equal-source macro score prevents the large phantom validation subset
    # from deciding the best checkpoint by itself.
    metrics["macro_l1"] = sum(source_l1.values()) / max(len(source_l1), 1)
    for source, value in source_l1.items():
        metrics[f"l1_{source}"] = value
    return metrics, preview_batch


def write_split_manifest(
    output_path: Path,
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "source", "group", "path", "page_index"])
        for split, records in (("train", train_records), ("val", val_records)):
            for record in records:
                writer.writerow([
                    split,
                    record.source,
                    record.group,
                    str(record.path),
                    record.page_index,
                ])


def run_training(config: TrainConfig) -> None:
    if config.save_every <= 0:
        raise ValueError("save_every must be greater than zero")
    if config.patience <= 0:
        raise ValueError("patience must be greater than zero")

    seed_everything(config.train_seed)
    device = select_device(config.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_training_records(config)
    train_records, val_records = split_records_by_group(
        records,
        val_fraction=config.val_fraction,
        seed=config.split_seed,
    )
    write_split_manifest(
        output_dir / "split_manifest.csv",
        train_records,
        val_records,
    )

    train_dataset = MixedPhotoacousticDataset(
        train_records,
        image_height=config.image_height,
        image_width=config.image_width,
        max_distortion_px=config.max_distortion_px,
        tail_max_distortion_px=config.tail_max_distortion_px,
        tail_probability=config.tail_probability,
        identity_probability=config.identity_probability,
        jitter_profile=config.jitter_profile,
        min_jitter_width_px=config.min_jitter_width_px,
        max_jitter_width_px=config.max_jitter_width_px,
        min_jitter_gap_px=config.min_jitter_gap_px,
        max_jitter_gap_px=config.max_jitter_gap_px,
        min_jitter_std_px=config.min_jitter_std_px,
        max_jitter_std_px=config.max_jitter_std_px,
        min_jitter_components=config.min_jitter_components,
        max_jitter_components=config.max_jitter_components,
        empirical_jitter_path=config.empirical_jitter_path,
        empirical_jitter_probability=config.empirical_jitter_probability,
        augment_probability=config.augment_probability,
        base_seed=config.train_seed,
    )
    val_dataset = MixedPhotoacousticDataset(
        val_records,
        image_height=config.image_height,
        image_width=config.image_width,
        max_distortion_px=config.max_distortion_px,
        tail_max_distortion_px=config.tail_max_distortion_px,
        tail_probability=config.tail_probability,
        identity_probability=0.0,
        jitter_profile=config.jitter_profile,
        min_jitter_width_px=config.min_jitter_width_px,
        max_jitter_width_px=config.max_jitter_width_px,
        min_jitter_gap_px=config.min_jitter_gap_px,
        max_jitter_gap_px=config.max_jitter_gap_px,
        min_jitter_std_px=config.min_jitter_std_px,
        max_jitter_std_px=config.max_jitter_std_px,
        min_jitter_components=config.min_jitter_components,
        max_jitter_components=config.max_jitter_components,
        empirical_jitter_path=config.empirical_jitter_path,
        empirical_jitter_probability=config.empirical_jitter_probability,
        augment_probability=0.0,
        base_seed=config.train_seed + 99_999,
    )

    sampler = make_source_balanced_sampler(
        train_records,
        balance_power=config.source_balance_power,
        seed=config.train_seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        # Workers are recreated after set_epoch(), so their dataset copy sees
        # the new epoch and generates a different jitter field each epoch.
        persistent_workers=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    save_jitter_previews(
        train_dataset,
        output_dir / "jitter_previews",
        config.jitter_preview_count,
    )

    generator = MACNetColumnGenerator(
        in_channels=1,
        max_correction_px=config.tail_max_distortion_px * 1.10,
    ).to(device)
    discriminator = ConditionalPatchDiscriminator(image_channels=1).to(device)

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.lr_g,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr_d,
        betas=(config.beta1, config.beta2),
    )

    log_path = output_dir / "training_log.csv"
    best_val_shift_score = float("inf")
    initial_epoch = 0
    epochs_without_improvement = 0

    if config.resume_checkpoint:
        resume = torch.load(
            config.resume_checkpoint, map_location=device, weights_only=False
        )
        generator.load_state_dict(resume["generator"])
        discriminator.load_state_dict(resume["discriminator"])
        if "optimizer_g" in resume:
            optimizer_g.load_state_dict(resume["optimizer_g"])
        if "optimizer_d" in resume:
            optimizer_d.load_state_dict(resume["optimizer_d"])
        initial_epoch = int(resume.get("epoch", 0))
        print(f"Resumed weights and optimizers from epoch {initial_epoch}: "
              f"{config.resume_checkpoint}")

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "adversarial_enabled",
            "train_g",
            "train_d",
            "train_l1",
            "train_ssim_loss",
            "train_gradient",
            "train_shift",
            "train_shift_edge",
            "train_identity",
            "train_adv",
            "val_l1",
            "val_macro_l1",
            "val_ssim",
            "val_psnr",
            "val_shift_mae_px",
            "val_weighted_shift",
            "val_shift_edge",
        ])

    print(f"Device: {device}")
    train_counts = Counter(record.source for record in train_records)
    val_counts = Counter(record.source for record in val_records)
    print(f"Training samples by source: {dict(sorted(train_counts.items()))}")
    print(f"Validation samples by source: {dict(sorted(val_counts.items()))}")
    if config.jitter_profile == "smooth_v2":
        print(
            "Online distortion profile=smooth_v2, period=20--50px, "
            "mostly 0.3--3px, local amplitude/frequency modulation + "
            f"smooth random motion, tail cap=+/-{config.tail_max_distortion_px:g}px "
            f"(p={config.tail_probability:g})"
        )
    else:
        print(
            f"Online distortion profile=block_v1, "
            f"std={config.min_jitter_std_px:g}--{config.max_jitter_std_px:g}px, "
            f"components={config.min_jitter_components}--"
            f"{config.max_jitter_components}, "
            f"pulse width={config.min_jitter_width_px}--"
            f"{config.max_jitter_width_px} columns, "
            f"gap={config.min_jitter_gap_px}--"
            f"{config.max_jitter_gap_px} columns, "
            f"typical cap=+/-{config.max_distortion_px:g}px, "
            f"tail cap=+/-{config.tail_max_distortion_px:g}px "
            f"(p={config.tail_probability:g})"
        )
    if config.empirical_jitter_path:
        print(
            "Empirical jitter mix: "
            f"{100.0 * config.empirical_jitter_probability:.0f}% from "
            f"{config.empirical_jitter_path}"
        )
    print(f"Source balance power: {config.source_balance_power:g}")
    if sampler is not None:
        source_mass = {
            source: count ** (1.0 - config.source_balance_power)
            for source, count in train_counts.items()
        }
        total_mass = sum(source_mass.values())
        expected = {
            source: f"{100.0 * mass / total_mass:.1f}%"
            for source, mass in sorted(source_mass.items())
        }
        print(f"Expected sampled source ratio: {expected}")

    final_epoch = initial_epoch + config.epochs
    for epoch in range(initial_epoch + 1, final_epoch + 1):
        train_dataset.set_epoch(epoch)

        use_adversarial = epoch > config.pretrain_epochs

        train_metrics = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            optimizer_g,
            optimizer_d,
            device,
            config,
            use_adversarial=use_adversarial,
        )

        val_metrics, preview = validate(generator, val_loader, device, config)

        print(
            f"Epoch {epoch:03d}/{final_epoch} | "
            f"G {train_metrics['g']:.4f} | "
            f"D {train_metrics['d']:.4f} | "
            f"Val L1 {val_metrics['l1']:.5f} | "
            f"Macro L1 {val_metrics['macro_l1']:.5f} | "
            f"SSIM {val_metrics['ssim']:.4f} | "
            f"PSNR {val_metrics['psnr']:.2f} | "
            f"Shift MAE {val_metrics['shift_mae']:.3f}px | "
            f"Weighted {val_metrics['weighted_shift']:.3f} | "
            f"Edge {val_metrics['shift_edge']:.3f}"
        )
        per_source_text = ", ".join(
            f"{key[3:]}={value:.5f}"
            for key, value in sorted(val_metrics.items())
            if key.startswith("l1_")
        )
        print(f"  Validation L1 by source: {per_source_text}")

        with log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch,
                int(use_adversarial),
                train_metrics["g"],
                train_metrics["d"],
                train_metrics["l1"],
                train_metrics["ssim"],
                train_metrics["grad"],
                train_metrics["shift"],
                train_metrics["shift_edge"],
                train_metrics["identity"],
                train_metrics["adv"],
                val_metrics["l1"],
                val_metrics["macro_l1"],
                val_metrics["ssim"],
                val_metrics["psnr"],
                val_metrics["shift_mae"],
                val_metrics["weighted_shift"],
                val_metrics["shift_edge"],
            ])

        if epoch % config.preview_every == 0 or epoch == 1:
            save_preview(
                output_dir / "previews" / f"epoch_{epoch:04d}.png",
                **preview,
            )

        val_shift_score = val_metrics["weighted_shift"] + val_metrics["shift_edge"]
        if val_shift_score < best_val_shift_score:
            best_val_shift_score = val_shift_score
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                config,
                best_val_shift_score,
            )
        else:
            epochs_without_improvement += 1

        if epoch % config.save_every == 0 or epoch == final_epoch:
            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                config,
                best_val_shift_score,
            )

        if epochs_without_improvement >= config.patience:
            print(
                f"Early stopping at epoch {epoch}: validation shift score did not "
                f"improve for {config.patience} consecutive epochs. "
                f"Best checkpoint: {output_dir / 'best.pt'}"
            )
            break


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    checkpoint_path: Path,
    input_tiff: Path,
    output_tiff: Path,
    page_index: int,
    device_name: str,
    tile_height: int = 256,
    tile_width: int = 512,
    tile_stride_y: int = 128,
    tile_stride_x: int = 256,
    tile_batch_size: int = 4,
) -> None:
    """
    page_index >= 0: correct one page.
    page_index == -1: correct every page and save a multi-page TIFF.
    """
    device = select_device(device_name)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_config = checkpoint.get("config", {})

    max_distortion = float(saved_config.get("max_distortion_px", 1.0))
    if "tail_max_distortion_px" in saved_config:
        max_correction = 1.10 * float(saved_config["tail_max_distortion_px"])
    else:
        # Backward compatibility with old large-shift checkpoints.
        max_correction = 1.25 * max_distortion

    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile_height and tile_width must be greater than zero.")
    if tile_stride_y <= 0 or tile_stride_y > tile_height:
        raise ValueError("tile_stride_y must be in [1, tile_height].")
    if tile_stride_x <= 0 or tile_stride_x > tile_width:
        raise ValueError("tile_stride_x must be in [1, tile_width].")
    if tile_batch_size <= 0:
        raise ValueError("tile_batch_size must be greater than zero.")

    model = MACNetColumnGenerator(
        in_channels=1,
        max_correction_px=max_correction,
    ).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    print(
        f"Tiled inference: tile={tile_height}x{tile_width}, "
        f"stride={tile_stride_y}x{tile_stride_x}, "
        f"overlap={tile_height - tile_stride_y}x"
        f"{tile_width - tile_stride_x}, "
        f"batch={tile_batch_size}"
    )

    def tile_starts(length: int, size: int, stride: int) -> List[int]:
        if length <= size:
            return [0]
        starts = list(range(0, length - size + 1, stride))
        final_start = length - size
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def correct_page(current_page: int) -> Tuple[np.ndarray, np.ndarray]:
        original = read_tiff_page(input_tiff, current_page)
        original_shape = original.shape
        normalized = robust_normalize_01(original)
        height, width = original_shape
        pad_bottom = max(0, tile_height - height)
        pad_right = max(0, tile_width - width)
        padded = np.pad(
            normalized,
            ((0, pad_bottom), (0, pad_right)),
            mode="reflect" if min(height, width) > 1 else "edge",
        ).astype(np.float32)
        padded_height, padded_width = padded.shape

        y_starts = tile_starts(padded_height, tile_height, tile_stride_y)
        x_starts = tile_starts(padded_width, tile_width, tile_stride_x)
        coordinates = [(y, x) for y in y_starts for x in x_starts]

        y_weight = np.maximum(np.hanning(tile_height).astype(np.float32), 0.05)
        x_weight = np.maximum(np.hanning(tile_width).astype(np.float32), 0.05)
        blend_weight = np.outer(y_weight, x_weight).astype(np.float32)
        corrected_sum = np.zeros_like(padded, dtype=np.float32)
        shift_sum = np.zeros_like(padded, dtype=np.float32)
        weight_sum = np.zeros_like(padded, dtype=np.float32)

        for batch_start in range(0, len(coordinates), tile_batch_size):
            batch_coordinates = coordinates[
                batch_start:batch_start + tile_batch_size
            ]
            patches = np.stack([
                padded[y:y + tile_height, x:x + tile_width]
                for y, x in batch_coordinates
            ])
            tensor = torch.from_numpy(patches[:, None]).float().to(device)
            corrected, predicted_shift = model(tensor)
            corrected_batch = corrected[:, 0].cpu().numpy().astype(np.float32)
            shift_batch = predicted_shift.cpu().numpy().astype(np.float32)

            for batch_index, (y, x) in enumerate(batch_coordinates):
                corrected_sum[y:y + tile_height, x:x + tile_width] += (
                    corrected_batch[batch_index] * blend_weight
                )
                # The MACNet field is one y-shift per x-column. Repeat it over
                # the tile height so overlapping 2-D tiles can be blended.
                tile_shift = np.broadcast_to(
                    shift_batch[batch_index][None, :],
                    (tile_height, tile_width),
                )
                shift_sum[y:y + tile_height, x:x + tile_width] += (
                    tile_shift * blend_weight
                )
                weight_sum[y:y + tile_height, x:x + tile_width] += blend_weight

        corrected_np = corrected_sum / np.maximum(weight_sum, 1e-6)
        correction_field = shift_sum / np.maximum(weight_sum, 1e-6)
        corrected_np = corrected_np[:height, :width]
        correction_field = correction_field[:height, :width]

        corrected_u16 = np.clip(
            corrected_np * 65535.0,
            0,
            65535,
        ).astype(np.uint16)

        return corrected_u16, correction_field.astype(np.float32)

    output_tiff.parent.mkdir(parents=True, exist_ok=True)

    if page_index >= 0:
        corrected_page, shift_page = correct_page(page_index)
        tifffile.imwrite(str(output_tiff), corrected_page)
        shift_path = output_tiff.with_suffix(".shift.npy")
        np.save(shift_path, shift_page)

        print(f"Saved corrected page: {output_tiff}")
        print(f"Saved predicted correction field: {shift_path}")
        return

    page_count = count_tiff_pages(input_tiff)
    corrected_pages: List[np.ndarray] = []
    shift_pages: List[np.ndarray] = []

    reference_shape: Tuple[int, int] | None = None

    for current_page in range(page_count):
        corrected_page, shift_page = correct_page(current_page)

        if reference_shape is None:
            reference_shape = corrected_page.shape
        elif corrected_page.shape != reference_shape:
            raise ValueError(
                "All pages must have the same H x W shape to save one "
                f"multi-page TIFF. First page={reference_shape}, "
                f"page {current_page}={corrected_page.shape}."
            )

        corrected_pages.append(corrected_page)
        shift_pages.append(shift_page)
        print(f"Corrected page {current_page + 1}/{page_count}")

    corrected_stack = np.stack(corrected_pages, axis=0)
    shift_stack = np.stack(shift_pages, axis=0)

    tifffile.imwrite(str(output_tiff), corrected_stack)
    shift_path = output_tiff.with_suffix(".shift.npy")
    np.save(shift_path, shift_stack.astype(np.float32))

    print(f"Saved corrected multi-page TIFF: {output_tiff}")
    print(f"Saved correction fields [N, H, W]: {shift_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="MAC-Net-inspired A-line artifact correction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--data-root",
        type=str,
        default=defaults.data_root,
        help=f"Clean MAP image folder (default: {defaults.data_root}).",
    )
    train_parser.add_argument("--phantom-root", default=defaults.phantom_root)
    train_parser.add_argument("--rsom-root", default=defaults.rsom_root)
    train_parser.add_argument("--pa-root", default=defaults.pa_root)
    train_parser.add_argument("--no-phantom", action="store_true")
    train_parser.add_argument("--no-rsom", action="store_true")
    train_parser.add_argument("--no-pa", action="store_true")
    train_parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    train_parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=defaults.resume_checkpoint,
        help="Optional checkpoint whose weights and optimizer states are resumed.",
    )
    train_parser.add_argument("--height", type=int, default=defaults.image_height)
    train_parser.add_argument("--width", type=int, default=defaults.image_width)
    train_parser.add_argument("--epochs", type=int, default=defaults.epochs)
    train_parser.add_argument(
        "--pretrain-epochs", type=int, default=defaults.pretrain_epochs
    )
    train_parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    train_parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    train_parser.add_argument(
        "--max-shift",
        type=float,
        default=defaults.max_distortion_px,
        help=f"Typical jitter cap in pixels (default: {defaults.max_distortion_px:g}).",
    )
    train_parser.add_argument(
        "--tail-max-shift",
        type=float,
        default=defaults.tail_max_distortion_px,
        help=(
            "Rare local tail cap in pixels "
            f"(default: {defaults.tail_max_distortion_px:g})."
        ),
    )
    train_parser.add_argument(
        "--tail-probability",
        type=float,
        default=defaults.tail_probability,
    )
    train_parser.add_argument(
        "--min-jitter-width", type=int, default=defaults.min_jitter_width_px
    )
    train_parser.add_argument(
        "--max-jitter-width", type=int, default=defaults.max_jitter_width_px
    )
    train_parser.add_argument(
        "--min-jitter-gap", type=int, default=defaults.min_jitter_gap_px
    )
    train_parser.add_argument(
        "--max-jitter-gap", type=int, default=defaults.max_jitter_gap_px
    )
    train_parser.add_argument(
        "--min-jitter-std",
        type=float,
        default=defaults.min_jitter_std_px,
    )
    train_parser.add_argument(
        "--max-jitter-std",
        type=float,
        default=defaults.max_jitter_std_px,
    )
    train_parser.add_argument(
        "--min-jitter-components",
        type=int,
        default=defaults.min_jitter_components,
        help="Minimum number of jitter bases superimposed at each x position.",
    )
    train_parser.add_argument(
        "--max-jitter-components",
        type=int,
        default=defaults.max_jitter_components,
        help="Maximum number of jitter bases superimposed at each x position.",
    )
    train_parser.add_argument(
        "--empirical-jitter-path",
        type=str,
        default=defaults.empirical_jitter_path,
        help="Optional high_frequency_jitter.npy path.",
    )
    train_parser.add_argument(
        "--empirical-jitter-probability",
        type=float,
        default=defaults.empirical_jitter_probability,
        help="Fraction of jitter samples based on the measured profile.",
    )
    train_parser.add_argument(
        "--identity-probability",
        type=float,
        default=defaults.identity_probability,
    )
    train_parser.add_argument(
        "--jitter-profile",
        choices=("smooth_v2", "block_v1"),
        default=defaults.jitter_profile,
    )
    train_parser.add_argument(
        "--augment-probability",
        type=float,
        default=defaults.augment_probability,
    )
    train_parser.add_argument(
        "--source-balance-power",
        type=float,
        default=defaults.source_balance_power,
        help="0 disables source balancing; 1 gives each source equal probability.",
    )
    train_parser.add_argument(
        "--val-fraction", type=float, default=defaults.val_fraction
    )
    train_parser.add_argument(
        "--save-every", type=int, default=defaults.save_every,
        help="Save a periodic checkpoint every N epochs (default: 50).",
    )
    train_parser.add_argument(
        "--jitter-preview-count",
        type=int,
        default=defaults.jitter_preview_count,
        help="Save this many synthetic jitter examples before training (0 disables).",
    )
    train_parser.add_argument(
        "--patience", type=int, default=defaults.patience,
        help="Stop after N epochs without validation macro-L1 improvement (default: 30).",
    )
    train_parser.add_argument("--device", type=str, default="auto")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--input-tiff", type=Path, required=True)
    infer_parser.add_argument("--output-tiff", type=Path, required=True)
    infer_parser.add_argument(
        "--page",
        type=int,
        default=-1,
        help="-1 corrects all TIFF pages; nonnegative value corrects one page.",
    )
    infer_parser.add_argument("--device", type=str, default="auto")
    infer_parser.add_argument("--tile-height", type=int, default=256)
    infer_parser.add_argument("--tile-width", type=int, default=512)
    infer_parser.add_argument(
        "--tile-stride-y",
        type=int,
        default=128,
        help="Vertical tile stride; 128 gives 50%% overlap for 256px height.",
    )
    infer_parser.add_argument(
        "--tile-stride-x",
        type=int,
        default=256,
        help="Horizontal tile stride; 256 gives 50%% overlap for 512px width.",
    )
    infer_parser.add_argument("--tile-batch-size", type=int, default=4)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            phantom_root="" if args.no_phantom else args.phantom_root,
            rsom_root="" if args.no_rsom else args.rsom_root,
            pa_root="" if args.no_pa else args.pa_root,
            output_dir=args.output_dir,
            resume_checkpoint=args.resume_checkpoint,
            image_height=args.height,
            image_width=args.width,
            epochs=args.epochs,
            pretrain_epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_distortion_px=args.max_shift,
            tail_max_distortion_px=args.tail_max_shift,
            tail_probability=args.tail_probability,
            min_jitter_width_px=args.min_jitter_width,
            max_jitter_width_px=args.max_jitter_width,
            min_jitter_gap_px=args.min_jitter_gap,
            max_jitter_gap_px=args.max_jitter_gap,
            min_jitter_std_px=args.min_jitter_std,
            max_jitter_std_px=args.max_jitter_std,
            min_jitter_components=args.min_jitter_components,
            max_jitter_components=args.max_jitter_components,
            empirical_jitter_path=args.empirical_jitter_path,
            empirical_jitter_probability=args.empirical_jitter_probability,
            identity_probability=args.identity_probability,
            jitter_profile=args.jitter_profile,
            augment_probability=args.augment_probability,
            source_balance_power=args.source_balance_power,
            val_fraction=args.val_fraction,
            save_every=args.save_every,
            jitter_preview_count=args.jitter_preview_count,
            patience=args.patience,
            device=args.device,
        )
        run_training(config)
    elif args.command == "infer":
        run_inference(
            checkpoint_path=args.checkpoint,
            input_tiff=args.input_tiff,
            output_tiff=args.output_tiff,
            page_index=args.page,
            device_name=args.device,
            tile_height=args.tile_height,
            tile_width=args.tile_width,
            tile_stride_y=args.tile_stride_y,
            tile_stride_x=args.tile_stride_x,
            tile_batch_size=args.tile_batch_size,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
