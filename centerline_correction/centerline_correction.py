import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from skimage.filters import frangi
from scipy.ndimage import gaussian_filter1d
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

from predict_vessel_mask.predict import (
    DEFAULT_MODEL as PREDICT_DEFAULT_MODEL,
    load_model,
    predict as predict_unet,
    select_device,
)


# ============================================================
# User settings
# ============================================================
SMOOTHING = 1500

BIN_WIDTH = 15
WINDOW_WIDTH = 60
STRIDE = 25
ORIGINAL_FILTER_THRESHOLD = 0.05
UNET_FILTER_THRESHOLD = 0

ORIGINAL_WEIGHT_POWER = 2.0
UNET_WEIGHT_POWER = 2.0

BASE_UNET_WEIGHT = 0.7
MIN_UNET_WEIGHT = 0.6
DISAGREEMENT_SCALE = 25.0
SPREAD_BASELINE_SIGMA = 10.0
SPREAD_ERROR_SCALE = 60.0

CENTERLINE_SOURCE = "original"
ORIGINAL_CENTERLINE_WEIGHT = 0.2
UNET_CENTERLINE_WEIGHT = 0.8

FILTER_THRESHOLD = 0.05  # Threshold for removing low-probability pixels before centerline extraction.

PREDICTION_IMAGE_PATH = (
    REPO_ROOT
    / "predict_vessel_mask"
    / "prediction_result"
    / "multicontrast"
    / "gray_mouse_ear_flattened_multicontrast_probability.npy"
)
UNET_INPUT_IMAGE_PATH = REPO_ROOT / "raw" / "gray_mouse_ear.png"
RECONSTRUCTION_IMAGE_PATH = REPO_ROOT / "raw" / "mouse_ear.png"

APPLY_INPUT_DISTORTION = False
# (x_fraction, y_shift_pixels). Positive shift moves content downward.
INPUT_DISTORTION_POINTS = [
    (0.0, 0.0),
    (0.25, -160.0),
    (0.55, 180.0),
    (1.0, 0.0),
]

# The selected centerline is used for flattening.
# Both original and U-Net centerlines are always calculated and displayed.


# Values below this threshold are removed before center-point extraction.


# These settings intentionally match the uploaded centerline method.
CENTERLINE_MIN_WEIGHT = 1.0
CENTERLINE_Q_LOW = 25
CENTERLINE_Q_HIGH = 75
CENTERLINE_WEIGHT_POWER = 2.0
CENTERLINE_PROB_THRESHOLD = 0.05

UNET_GAUSSIAN_SIGMA = 0

# Settings retained from the previously generated 2-channel U-Net code.
APPLY_FRANGI_POSTPROCESS = True
POST_FRANGI_WEIGHT = 0.2
POST_FRANGI_SIGMAS = (1, 2, 3, 4)

UNET_MODEL = REPO_ROOT / "models" / "unet_2ch_input.pt"
if not UNET_MODEL.is_file():
    UNET_MODEL = Path(PREDICT_DEFAULT_MODEL)

DEVICE = "auto"  # "auto", "cpu", or e.g. "cuda:0"
PREDICTION_OUTPUT_DIR = (
    REPO_ROOT / "centerline_correction" / "centerline_corrected_result"
)


# ============================================================
# Utility functions
# ============================================================

def interpolate_isolated_outliers(
    x,
    y,
    median_radius=2,
    abs_threshold=18.0,
):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 5:
        return y.copy(), np.zeros(len(y), dtype=bool)

    local_median = np.empty_like(y)

    for i in range(len(y)):
        start = max(0, i - median_radius)
        stop = min(len(y), i + median_radius + 1)

        neighbors = np.concatenate([
            y[start:i],
            y[i + 1:stop]
        ])

        local_median[i] = (
            np.median(neighbors)
            if len(neighbors) > 0
            else y[i]
        )

    candidate = np.abs(y - local_median) > abs_threshold

    isolated = np.zeros_like(candidate)

    for i in range(1, len(candidate) - 1):
        isolated[i] = (
            candidate[i]
            and not candidate[i - 1]
            and not candidate[i + 1]
        )

    corrected = y.copy()

    if np.any(isolated):
        valid = ~isolated

        corrected[isolated] = np.interp(
            x[isolated],
            x[valid],
            y[valid],
        )

    return corrected, isolated

def soft_threshold_probability(
    probability: np.ndarray,
    threshold: float
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float32)

    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must satisfy 0 <= threshold < 1.")

    return np.clip(
        (probability - threshold) / (1.0 - threshold),
        0.0,
        1.0
    ).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract original-image and U-Net centerlines with the same "
            "weighted-percentile method."
        )
    )
    parser.add_argument("--sigma-min", type=int, default=2)
    parser.add_argument("--sigma-max", type=int, default=20)
    return parser.parse_args()


def normalize_01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    minimum = float(array.min())
    maximum = float(array.max())

    if maximum - minimum < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    return (array - minimum) / (maximum - minimum)


def load_probability_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        probability = np.load(path).astype(np.float32)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to load prediction input: {path}")
        probability = image.astype(np.float32)

    probability = np.asarray(probability).squeeze()
    if probability.ndim != 2:
        raise ValueError(
            f"Prediction input must be a 2-D probability map, got {probability.shape}."
        )

    if not np.all(np.isfinite(probability)):
        raise ValueError("Prediction probability map contains NaN or inf values.")

    if probability.max() > 1.0:
        probability = probability / 255.0

    return probability.astype(np.float32)


def probability_map_to_bgr(probability: np.ndarray) -> np.ndarray:
    image = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def distort_probability_map_y(
    probability: np.ndarray,
    shift_points: list[tuple[float, float]],
) -> np.ndarray:
    """Warp each column vertically using normalized x positions and pixel shifts."""
    probability = np.asarray(probability, dtype=np.float32)
    if not shift_points:
        return probability.copy()

    height, width = probability.shape
    points = np.asarray(shift_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("INPUT_DISTORTION_POINTS must contain (x_fraction, y_shift) pairs.")

    order = np.argsort(points[:, 0])
    x_points = np.clip(points[order, 0], 0.0, 1.0) * (width - 1)
    y_shifts = points[order, 1]
    if len(np.unique(x_points)) != len(x_points):
        raise ValueError("Each distortion x_fraction must be unique.")

    x_coordinates = np.arange(width, dtype=np.float32)
    column_shifts = np.interp(
        x_coordinates,
        x_points,
        y_shifts,
        left=y_shifts[0],
        right=y_shifts[-1],
    ).astype(np.float32)

    map_x = np.tile(x_coordinates, (height, 1))
    map_y = (
        np.arange(height, dtype=np.float32)[:, None]
        - column_shifts[None, :]
    )
    return cv2.remap(
        probability,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def distort_image_y(
    image: np.ndarray,
    shift_points: list[tuple[float, float]],
    shift_scale: float = 1.0,
) -> np.ndarray:
    """Warp a grayscale or color image vertically with scaled pixel shifts."""
    image = np.asarray(image)
    if not shift_points:
        return image.copy()

    height, width = image.shape[:2]
    points = np.asarray(shift_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("INPUT_DISTORTION_POINTS must contain (x_fraction, y_shift) pairs.")

    order = np.argsort(points[:, 0])
    x_points = np.clip(points[order, 0], 0.0, 1.0) * (width - 1)
    y_shifts = points[order, 1] * shift_scale
    if len(np.unique(x_points)) != len(x_points):
        raise ValueError("Each distortion x_fraction must be unique.")

    x_coordinates = np.arange(width, dtype=np.float32)
    column_shifts = np.interp(
        x_coordinates,
        x_points,
        y_shifts,
        left=y_shifts[0],
        right=y_shifts[-1],
    ).astype(np.float32)

    map_x = np.tile(x_coordinates, (height, 1))
    map_y = (
        np.arange(height, dtype=np.float32)[:, None]
        - column_shifts[None, :]
    )
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def enhance_probability_with_frangi(
    probability: np.ndarray,
    sigmas=(1, 2, 3, 4),
    weight: float = 0.2,
) -> np.ndarray:
    enhanced = frangi(
        probability.astype(np.float32),
        sigmas=sigmas,
        black_ridges=False,
    )
    enhanced = normalize_01(enhanced)
    combined = probability + weight * enhanced
    return np.clip(combined, 0.0, 1.0)


def weighted_percentile(values, weights, percentile):
    """Return a weighted percentile exactly as in the uploaded code."""
    values = np.asarray(values)
    weights = np.asarray(weights)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumsum = np.cumsum(weights)
    cutoff = percentile / 100.0 * cumsum[-1]

    return values[np.searchsorted(cumsum, cutoff)]


def extract_center_points_from_probability(
    prob,
    window_width=25,
    stride=5,
    min_weight=1.0,
    q_low=25,
    q_high=75,
    weight_power=1.0,
    prob_threshold=0.0,
    profile_sigma=2.0,
):
    prob = np.asarray(prob, dtype=np.float32)

    if prob.ndim != 2:
        raise ValueError("prob must be a 2-D array.")

    height, width = prob.shape

    center_x = []
    center_y = []
    center_top = []
    center_bottom = []

    for x0 in range(0, width, stride):
        x1 = min(x0 + window_width, width)

        if x1 <= x0:
            continue

        patch = prob[:, x0:x1]

        # 임계값 이하 제거
        patch = np.where(
            patch > prob_threshold,
            patch,
            0.0,
        )

        # 기존 픽셀 가중 방식을 y별로 합산
        y_profile = np.sum(
            patch ** weight_power,
            axis=1,
        )

        # 세로 방향의 작은 검출 변동 완화
        if profile_sigma > 0:
            y_profile = gaussian_filter1d(
                y_profile,
                sigma=profile_sigma,
                mode="nearest",
            )

        if y_profile.sum() < min_weight:
            continue

        valid = y_profile > 0
        y_values = np.arange(height)[valid]
        y_weights = y_profile[valid]

        y_top = weighted_percentile(
            y_values,
            y_weights,
            q_low,
        )

        y_bottom = weighted_percentile(
            y_values,
            y_weights,
            q_high,
        )

        center_x.append((x0 + x1 - 1) / 2)
        center_y.append((y_top + y_bottom) / 2)
        center_top.append(y_top)
        center_bottom.append(y_bottom)

    return (
        np.asarray(center_x, dtype=np.float32),
        np.asarray(center_y, dtype=np.float32),
        np.asarray(center_top, dtype=np.float32),
        np.asarray(center_bottom, dtype=np.float32),
    )


def fit_spline(center_x, center_y, smoothing=80000):
    center_x = np.asarray(center_x, dtype=np.float64)
    center_y = np.asarray(center_y, dtype=np.float64)

    if len(center_x) < 2 or len(center_y) < 2:
        raise ValueError("At least two center points are required for spline fitting.")
    if len(center_x) != len(center_y):
        raise ValueError("center_x and center_y must have the same length.")

    order = np.argsort(center_x)
    x = center_x[order]
    y = center_y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]

    if len(x) < 2:
        raise ValueError("At least two unique x positions are required.")

    k = min(3, len(x) - 1)
    spline = UnivariateSpline(x, y, k=k, s=smoothing)

    x_curve = np.arange(int(x.min()), int(x.max()) + 1)
    y_curve = spline(x_curve)

    return x_curve, y_curve, spline


def threshold_source_map(source_map: np.ndarray, threshold: float) -> np.ndarray:
    """Match the uploaded preprocessing: preserve values above threshold."""
    return np.where(
        source_map >= threshold,
        source_map,
        0.0,
    ).astype(np.float32)


def save_float_map_as_png(path: Path, array: np.ndarray) -> None:
    image = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Unable to save image: {path}")


def flatten_probability_map_with_centerline(
    probability: np.ndarray,
    x_curve: np.ndarray,
    y_curve: np.ndarray,
) -> tuple[np.ndarray, int]:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError("probability must be a 2-D array.")

    height, width = probability.shape
    x_curve = np.asarray(x_curve, dtype=np.float32)
    y_curve = np.asarray(y_curve, dtype=np.float32)

    order = np.argsort(x_curve)
    x_curve = x_curve[order]
    y_curve = y_curve[order]

    full_x = np.arange(width, dtype=np.float32)
    full_y = np.interp(
        full_x,
        x_curve,
        y_curve,
        left=float(y_curve[0]),
        right=float(y_curve[-1]),
    )
    target_y = int(np.round(np.median(full_y)))
    shift = full_y - target_y

    map_x = np.tile(full_x, (height, 1)).astype(np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    source_y = np.arange(height, dtype=np.float32)

    for x in range(width):
        map_y[:, x] = source_y + shift[x]

    flattened = cv2.remap(
        probability,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(flattened, 0.0, 1.0).astype(np.float32), target_y
    
def interpolate_short_outlier_runs(
    x,
    y,
    median_radius=4,
    abs_threshold=18.0,
    max_run_length=5,
):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(y) < 7:
        return y.copy(), np.zeros(len(y), dtype=bool)

    local_median = np.empty_like(y)

    for i in range(len(y)):
        start = max(0, i - median_radius)
        stop = min(len(y), i + median_radius + 1)

        neighbors = np.concatenate([
            y[start:i],
            y[i + 1:stop],
        ])

        local_median[i] = (
            np.median(neighbors)
            if len(neighbors) > 0
            else y[i]
        )

    candidate = (
        np.abs(y - local_median)
        > abs_threshold
    )

    correct_mask = np.zeros_like(candidate)

    i = 0
    while i < len(candidate):
        if not candidate[i]:
            i += 1
            continue

        start = i

        while i < len(candidate) and candidate[i]:
            i += 1

        stop = i
        run_length = stop - start

        if run_length <= max_run_length:
            correct_mask[start:stop] = True

    corrected = y.copy()

    valid = ~correct_mask

    if (
        np.any(correct_mask)
        and np.count_nonzero(valid) >= 2
    ):
        corrected[correct_mask] = np.interp(
            x[correct_mask],
            x[valid],
            y[valid],
        )

    return corrected, correct_mask


# ============================================================
# Main processing
# ============================================================
def main():
    args = parse_args()

    if args.sigma_min < 1 or args.sigma_max < args.sigma_min:
        raise ValueError("Sigma values must satisfy 1 <= sigma-min <= sigma-max.")
    if CENTERLINE_SOURCE not in {"original", "unet", "combined"}:
        raise ValueError(
            'CENTERLINE_SOURCE must be "original", "unet", or "combined".'
        )
    if ORIGINAL_CENTERLINE_WEIGHT < 0 or UNET_CENTERLINE_WEIGHT < 0:
        raise ValueError("Centerline weights must be non-negative.")
    weight_sum = ORIGINAL_CENTERLINE_WEIGHT + UNET_CENTERLINE_WEIGHT
    if weight_sum <= 0:
        raise ValueError("At least one centerline weight must be positive.")
    if not 0.0 <= FILTER_THRESHOLD <= 1.0:
        raise ValueError("FILTER_THRESHOLD must be between 0 and 1.")
    if UNET_GAUSSIAN_SIGMA < 0:
        raise ValueError("UNET_GAUSSIAN_SIGMA must be 0 or greater.")

    original_source_map = load_probability_map(PREDICTION_IMAGE_PATH)
    unet_input_map = load_probability_map(UNET_INPUT_IMAGE_PATH)

    if APPLY_INPUT_DISTORTION:
        original_source_map = distort_probability_map_y(
            original_source_map,
            INPUT_DISTORTION_POINTS,
        )
        unet_input_map = distort_probability_map_y(
            unet_input_map,
            INPUT_DISTORTION_POINTS,
        )

    prediction_bgr = probability_map_to_bgr(original_source_map)
    unet_input_bgr = probability_map_to_bgr(unet_input_map)

    # Always run U-Net so both original and prediction centerlines can be compared.
    device = select_device(DEVICE)
    model, metadata = load_model(Path(UNET_MODEL), device)

    if metadata.get("in_channels", 1) == 2:
        frangi_channel = normalize_01(
            frangi(
                unet_input_map,
                sigmas=range(args.sigma_min, args.sigma_max + 1),
                black_ridges=False,
            )
        )
        model_input = np.stack(
            [unet_input_map, frangi_channel],
            axis=0,
        )
    else:
        model_input = unet_input_map

    raw_unet_source_map = predict_unet(model, model_input, device)
    unet_source_map = raw_unet_source_map.astype(np.float32).copy()

    if APPLY_FRANGI_POSTPROCESS:
        unet_source_map = enhance_probability_with_frangi(
            unet_source_map,
            sigmas=POST_FRANGI_SIGMAS,
            weight=POST_FRANGI_WEIGHT,
        )

    if UNET_GAUSSIAN_SIGMA > 0:
        unet_source_map = cv2.GaussianBlur(
            unet_source_map,
            (0, 0),
            sigmaX=UNET_GAUSSIAN_SIGMA,
            sigmaY=UNET_GAUSSIAN_SIGMA,
        )

    if unet_source_map.shape != original_source_map.shape:
        unet_source_map = cv2.resize(
            unet_source_map,
            (original_source_map.shape[1], original_source_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        unet_input_map = cv2.resize(
            unet_input_map,
            (original_source_map.shape[1], original_source_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        unet_input_bgr = probability_map_to_bgr(unet_input_map)

    # Save thresholded maps for inspection, but extract centerlines from the
    # probability maps directly, matching the reference code.
    original_prob = soft_threshold_probability(
        original_source_map,
        ORIGINAL_FILTER_THRESHOLD,
    )

    unet_prob = soft_threshold_probability(
        unet_source_map,
        UNET_FILTER_THRESHOLD,
    )

    original_center_x, original_center_y, original_top, original_bottom = (
        extract_center_points_from_probability(
            original_prob,
            window_width=WINDOW_WIDTH,
            stride=STRIDE,
            min_weight=CENTERLINE_MIN_WEIGHT,
            q_low=CENTERLINE_Q_LOW,
            q_high=CENTERLINE_Q_HIGH,
            weight_power=ORIGINAL_WEIGHT_POWER,
            prob_threshold=0.0,
        )
    )

    unet_center_x, unet_center_y, unet_top, unet_bottom = (
        extract_center_points_from_probability(
            unet_prob,
            window_width=WINDOW_WIDTH,
            stride=STRIDE,
            min_weight=CENTERLINE_MIN_WEIGHT,
            q_low=CENTERLINE_Q_LOW,
            q_high=CENTERLINE_Q_HIGH,
            weight_power=UNET_WEIGHT_POWER,
            prob_threshold=0.0,
            profile_sigma=2.0,
        )
    )

    unet_center_y_corrected, unet_outliers = (
        interpolate_short_outlier_runs(
            unet_center_x,
            unet_center_y,
            median_radius=4,
            abs_threshold=18.0,
            max_run_length=5,
        )
    )

    if len(original_center_x) < 2:
        raise RuntimeError(
            "Not enough original-image center points. "
            "Try lowering FILTER_THRESHOLD or CENTERLINE_PROB_THRESHOLD."
        )
    if len(unet_center_x) < 2:
        raise RuntimeError(
            "Not enough U-Net center points. "
            "Try lowering FILTER_THRESHOLD or CENTERLINE_PROB_THRESHOLD."
        )

    original_x_curve, original_y_curve, _ = fit_spline(
        original_center_x,
        original_center_y,
        smoothing=SMOOTHING,
    )
    unet_x_curve, unet_y_curve, _ = fit_spline(
        unet_center_x,
        unet_center_y_corrected,
        smoothing=SMOOTHING,
    )

    if CENTERLINE_SOURCE == "original":
        selected_prob = original_source_map
        center_x = original_center_x
        center_y = original_center_y
        x_curve = original_x_curve
        y_curve = original_y_curve
    elif CENTERLINE_SOURCE == "unet":
        selected_prob = unet_source_map
        center_x = unet_center_x
        center_y = unet_center_y
        x_curve = unet_x_curve
        y_curve = unet_y_curve
    else:
        common_start = max(original_center_x.min(), unet_center_x.min())
        common_stop = min(original_center_x.max(), unet_center_x.max())
        if common_stop <= common_start:
            raise RuntimeError(
                "Original and U-Net centerline points do not overlap in x."
            )

        center_x = np.arange(
            common_start,
            common_stop + 0.5 * STRIDE,
            STRIDE,
            dtype=np.float32,
        )
        original_y_interp = np.interp(
            center_x,
            original_center_x,
            original_center_y,
        )
        unet_y_interp = np.interp(
            center_x,
            unet_center_x,
            unet_center_y_corrected,
        )
        unet_top_interp = np.interp(
            center_x,
            unet_center_x,
            unet_top,
        )
        unet_bottom_interp = np.interp(
            center_x,
            unet_center_x,
            unet_bottom,
        )
        unet_spread = unet_bottom_interp - unet_top_interp
        baseline_spread = gaussian_filter1d(
            unet_spread,
            sigma=SPREAD_BASELINE_SIGMA,
            mode="nearest",
        )
        spread_error = np.abs(unet_spread - baseline_spread)
        spread_confidence = np.exp(
            -((spread_error / SPREAD_ERROR_SCALE) ** 2)
        )
        delta = np.abs(unet_y_interp - original_y_interp)
        center_agreement = np.exp(
            -((delta / DISAGREEMENT_SCALE) ** 2)
        )
        combined_confidence = center_agreement * spread_confidence
        unet_weight = (
            MIN_UNET_WEIGHT
            + (BASE_UNET_WEIGHT - MIN_UNET_WEIGHT) * combined_confidence
        )
        original_weight = 1.0 - unet_weight
        center_y = (
            original_weight * original_y_interp
            + unet_weight * unet_y_interp
        ).astype(np.float32)
        x_curve, y_curve, _ = fit_spline(
            center_x,
            center_y,
            smoothing=SMOOTHING,
        )
        selected_prob = (
            (ORIGINAL_CENTERLINE_WEIGHT / weight_sum) * original_source_map
            + (UNET_CENTERLINE_WEIGHT / weight_sum) * unet_source_map
        ).astype(np.float32)

    output_dir = Path(PREDICTION_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = Path(PREDICTION_IMAGE_PATH).stem
    if APPLY_INPUT_DISTORTION:
        output_stem = f"{output_stem}_distorted"

    np.save(
        output_dir / f"{output_stem}_original_source.npy",
        original_source_map,
    )
    np.save(
        output_dir / f"{output_stem}_unet_input_source.npy",
        unet_input_map,
    )
    np.save(
        output_dir / f"{output_stem}_unet_probability_raw.npy",
        raw_unet_source_map,
    )
    np.save(
        output_dir / f"{output_stem}_unet_source_processed.npy",
        unet_source_map,
    )
    np.save(
        output_dir / f"{output_stem}_original_thr_{FILTER_THRESHOLD:g}.npy",
        original_prob,
    )
    np.save(
        output_dir / f"{output_stem}_unet_thr_{FILTER_THRESHOLD:g}.npy",
        unet_prob,
    )
    if CENTERLINE_SOURCE == "combined":
        np.save(
            output_dir / f"{output_stem}_combined_source.npy",
            selected_prob,
        )

    flattened_unet_probability, flattened_unet_target_y = (
        flatten_probability_map_with_centerline(
            unet_source_map,
            x_curve,
            y_curve,
        )
    )
    flattened_unet_probability_path = (
        output_dir
        / f"{output_stem}_unet_probability_{CENTERLINE_SOURCE}_centerline_flattened.npy"
    )
    np.save(
        flattened_unet_probability_path,
        flattened_unet_probability,
    )

    cv2.imwrite(
        str(output_dir / f"{output_stem}_original_input.png"),
        prediction_bgr,
    )
    cv2.imwrite(
        str(output_dir / f"{output_stem}_unet_input.png"),
        unet_input_bgr,
    )
    save_float_map_as_png(
        output_dir / f"{output_stem}_original_thresholded.png",
        original_prob,
    )
    save_float_map_as_png(
        output_dir / f"{output_stem}_unet_probability.png",
        unet_source_map,
    )
    save_float_map_as_png(
        output_dir / f"{output_stem}_unet_thresholded.png",
        unet_prob,
    )
    save_float_map_as_png(
        output_dir
        / f"{output_stem}_unet_probability_{CENTERLINE_SOURCE}_centerline_flattened.png",
        flattened_unet_probability,
    )

    print(
        f"U-Net prediction: device={device}, "
        f"base_channels={metadata['base_channels']}, "
        f"input_channels={metadata.get('in_channels', 'unknown')}, "
        f"Frangi postprocess={APPLY_FRANGI_POSTPROCESS}"
    )
    print(
        f"Prediction input probability map: {PREDICTION_IMAGE_PATH}, "
        f"shape={original_source_map.shape}, "
        f"min={original_source_map.min():.6f}, "
        f"max={original_source_map.max():.6f}"
    )
    print(
        f"U-Net input image: {UNET_INPUT_IMAGE_PATH}, "
        f"shape={unet_input_map.shape}, "
        f"min={unet_input_map.min():.6f}, "
        f"max={unet_input_map.max():.6f}"
    )
    print(
        f"Input distortion: {APPLY_INPUT_DISTORTION}, "
        f"points={INPUT_DISTORTION_POINTS}"
    )
    print(
        "Centerline extraction for original and U-Net: "
        f"window_width={WINDOW_WIDTH}, stride={STRIDE}, "
        f"q={CENTERLINE_Q_LOW}/{CENTERLINE_Q_HIGH}, "
        f"original_weight_power={ORIGINAL_WEIGHT_POWER:g}, "
        f"unet_weight_power={UNET_WEIGHT_POWER:g}, "
        "sliding windows"
    )
    print(
        f"Center points: original={len(original_center_x)}, "
        f"U-Net={len(unet_center_x)}, selected={CENTERLINE_SOURCE}"
    )
    print(
        "Flattened U-Net probability map saved: "
        f"{flattened_unet_probability_path}, "
        f"shape={flattened_unet_probability.shape}, "
        f"target_y={flattened_unet_target_y}"
    )
    if CENTERLINE_SOURCE == "combined":
        print(
            "Combined centerline weights: "
            f"U-Net min={np.min(unet_weight):.3f}, "
            f"max={np.max(unet_weight):.3f}, "
            f"mean={np.mean(unet_weight):.3f}, "
            f"confidence mean={np.mean(combined_confidence):.3f}"
        )

    # Overlay U-Net prediction on the original image for visual inspection.
    overlay_alpha = (0.65 * unet_prob)[..., None]
    prediction_color = np.zeros_like(unet_input_bgr, dtype=np.float32)
    prediction_color[..., 2] = 255.0
    original_prediction_overlay = (
        unet_input_bgr.astype(np.float32) * (1.0 - overlay_alpha)
        + prediction_color * overlay_alpha
    )
    original_prediction_overlay = np.clip(
        original_prediction_overlay,
        0,
        255,
    ).astype(np.uint8)
    overlay_path = output_dir / f"{output_stem}_original_unet_overlay.png"
    cv2.imwrite(str(overlay_path), original_prediction_overlay)

    # Display centerlines calculated with the identical method.
    panel_count = 3 if CENTERLINE_SOURCE == "combined" else 2
    centerline_figure, centerline_axes = plt.subplots(
        1,
        panel_count,
        figsize=(8 * panel_count, 5),
    )

    centerline_axes[0].imshow(
        cv2.cvtColor(prediction_bgr, cv2.COLOR_BGR2RGB)
    )
    centerline_axes[0].scatter(
        original_center_x,
        original_center_y,
        s=12,
        c="yellow",
        alpha=0.65,
        label="center points",
    )
    centerline_axes[0].plot(
        original_x_curve,
        original_y_curve,
        "r-",
        linewidth=2,
        label="fitted centerline",
    )
    centerline_axes[0].set_title("Original image centerline")
    centerline_axes[0].legend(loc="upper right", fontsize=8)
    centerline_axes[0].axis("off")

    centerline_axes[1].imshow(unet_source_map, cmap="gray")
    centerline_axes[1].scatter(
        unet_center_x,
        unet_center_y_corrected,
        s=12,
        c="yellow",
        alpha=0.65,
        label="corrected center points",
    )
    centerline_axes[1].plot(
        unet_x_curve,
        unet_y_curve,
        "r-",
        linewidth=2,
        label="fitted centerline",
    )
    centerline_axes[1].set_title("U-Net result centerline")
    centerline_axes[1].legend(loc="upper right", fontsize=8)
    centerline_axes[1].axis("off")

    if CENTERLINE_SOURCE == "combined":
        centerline_axes[2].imshow(selected_prob, cmap="gray")
        centerline_axes[2].scatter(
            center_x,
            center_y,
            s=12,
            c="yellow",
            alpha=0.65,
            label="weighted center points",
        )
        centerline_axes[2].plot(
            x_curve,
            y_curve,
            "r-",
            linewidth=2,
            label="fitted centerline",
        )
        centerline_axes[2].set_title("Weighted combined centerline")
        centerline_axes[2].legend(loc="upper right", fontsize=8)
        centerline_axes[2].axis("off")

    centerline_figure.tight_layout()
    centerline_path = output_dir / f"{output_stem}_original_unet_centerlines.png"
    centerline_figure.savefig(centerline_path, dpi=160, bbox_inches="tight")
    plt.show()

    # Load reconstruction image and flatten it with the selected centerline.
    reconstruction_bgr = cv2.imread(
        RECONSTRUCTION_IMAGE_PATH,
        cv2.IMREAD_COLOR,
    )
    if reconstruction_bgr is None:
        raise FileNotFoundError(
            f"Unable to load reconstruction image: {RECONSTRUCTION_IMAGE_PATH}"
        )

    source_height, source_width = selected_prob.shape
    reconstruction_height = reconstruction_bgr.shape[0]
    if APPLY_INPUT_DISTORTION:
        reconstruction_bgr = distort_image_y(
            reconstruction_bgr,
            INPUT_DISTORTION_POINTS,
            shift_scale=reconstruction_height / source_height,
        )

    reconstruction_rgb = cv2.cvtColor(
        reconstruction_bgr,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)

    height, width = reconstruction_rgb.shape[:2]

    scale_x = width / source_width
    scale_y = height / source_height

    x_curve_orig = x_curve * scale_x
    y_curve_orig = y_curve * scale_y

    order = np.argsort(x_curve_orig)
    x_curve_orig = x_curve_orig[order]
    y_curve_orig = y_curve_orig[order]

    target_y = int(np.median(y_curve_orig))
    full_x = np.arange(width)
    full_y = np.interp(full_x, x_curve_orig, y_curve_orig)
    shift = full_y - target_y

    map_x = np.tile(np.arange(width), (height, 1)).astype(np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for x in range(width):
        map_y[:, x] = np.arange(height) + shift[x]

    flattened_rgb = cv2.remap(
        reconstruction_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    flattened_figure = plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(reconstruction_rgb, 0, 255).astype(np.uint8))
    plt.plot(
        x_curve_orig,
        y_curve_orig,
        "g",
        linewidth=2,
        label=f"{CENTERLINE_SOURCE} centerline",
    )
    plt.title(
        f"Original reconstruction with {CENTERLINE_SOURCE} centerline"
    )
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(flattened_rgb, 0, 255).astype(np.uint8))
    plt.axhline(
        target_y,
        color="g",
        linewidth=2,
        label="target straight line",
    )
    plt.title("Flattened image")
    plt.axis("off")

    plt.tight_layout()
    flattened_path = (
        output_dir
        / f"{output_stem}_{CENTERLINE_SOURCE}_centerline_flattened.png"
    )
    flattened_figure.savefig(flattened_path, dpi=160, bbox_inches="tight")

    cv2.imwrite(
        str(output_dir / f"{Path(RECONSTRUCTION_IMAGE_PATH).stem}_original.png"),
        reconstruction_bgr,
    )
    flattened_bgr = cv2.cvtColor(
        np.clip(flattened_rgb, 0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    cv2.imwrite(
        str(
            output_dir
            / f"{Path(RECONSTRUCTION_IMAGE_PATH).stem}_{CENTERLINE_SOURCE}_flattened.png"
        ),
        flattened_bgr,
    )

    print(f"Original/U-Net overlay saved: {overlay_path}")
    print(f"Centerline comparison saved: {centerline_path}")
    print(f"Flattened comparison saved: {flattened_path}")
    plt.show()


if __name__ == "__main__":
    main()
