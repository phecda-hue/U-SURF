import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wavelet_denoise import (
    DEFAULT_INPUT,
    atrous_decompose,
    normalize01,
    read_rgba_luma,
    save_gray,
    save_with_luma,
    structure_weight,
    white_annotation_mask,
)


@dataclass
class Component:
    x0: int
    x1: int
    center_x: float
    curvature: float
    sign: float
    period: float
    score: float
    y_shift: float = 0.0
    y_shifts: tuple[float, ...] = (0.0,)
    y_shift_weights: tuple[float, ...] = (1.0,)


def fill_missing_1d(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).copy()
    good = np.flatnonzero(ok)
    bad = np.flatnonzero(~ok)
    if len(good) == 0:
        return np.zeros_like(values)
    values[bad] = np.interp(bad, good, values[good], left=values[good[0]], right=values[good[-1]])
    return values


def binned_trimmed_profile(values: np.ndarray, bins: np.ndarray, bin_count: int, trim: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    profile = np.zeros(bin_count, np.float32)
    counts = np.bincount(bins, minlength=bin_count).astype(np.float32)
    ok = counts > 0
    order = np.argsort(bins)
    sorted_bins = bins[order]
    sorted_values = values[order].astype(np.float32)
    starts = np.searchsorted(sorted_bins, np.arange(bin_count), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(bin_count), side="right")

    for idx in np.flatnonzero(ok):
        vals = np.sort(sorted_values[starts[idx] : ends[idx]])
        if len(vals) >= 8:
            lo = int(len(vals) * trim)
            hi = max(lo + 1, int(len(vals) * (1 - trim)))
            profile[idx] = float(np.mean(vals[lo:hi]))
        else:
            profile[idx] = float(np.median(vals))
    profile = fill_missing_1d(profile, ok)
    return profile, ok


def binned_mean_profile(values: np.ndarray, bins: np.ndarray, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(bins, minlength=bin_count).astype(np.float32)
    sums = np.bincount(bins, weights=values.astype(np.float32), minlength=bin_count).astype(np.float32)
    ok = counts > 0
    profile = np.zeros(bin_count, np.float32)
    profile[ok] = sums[ok] / np.maximum(counts[ok], 1)
    return fill_missing_1d(profile, ok), ok


def harmonic_profile(values: np.ndarray, phase: np.ndarray, bg: np.ndarray, period: float, harmonics: int) -> np.ndarray:
    bins = np.floor(np.mod(phase, period) / period * 256).astype(np.int32)
    valid_bins = bins[bg].ravel()
    valid_values = values[bg].ravel().astype(np.float32)
    if len(valid_values) < 200:
        return np.zeros(256, np.float32)
    profile, ok = binned_trimmed_profile(valid_values, valid_bins, 256, trim=0.2)
    if ok.sum() < 24:
        return profile
    profile -= profile.mean()

    fft = np.fft.rfft(profile)
    keep = np.zeros_like(fft)
    for k in range(1, min(harmonics + 1, len(fft))):
        keep[k] = fft[k]
    return np.fft.irfft(keep, n=len(profile)).astype(np.float32)


def phase_to_field(profile: np.ndarray, phase: np.ndarray, period: float) -> np.ndarray:
    idx = np.mod(phase, period) / period * len(profile)
    lo = np.floor(idx).astype(np.int32) % len(profile)
    hi = (lo + 1) % len(profile)
    frac = idx - np.floor(idx)
    return profile[lo] * (1 - frac) + profile[hi] * frac


def parabola_phase(h: int, w: int, center_x: float, curvature: float, sign: float, y_shift: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    return yy - y_shift - sign * curvature * (xx - center_x) ** 2


def score_candidate(
    residual: np.ndarray,
    bg: np.ndarray,
    center_x: float,
    curvature: float,
    sign: float,
    period_range: tuple[float, float],
) -> tuple[float, float]:
    h, w = residual.shape
    phase = parabola_phase(h, w, center_x, curvature, sign)
    ph = phase[bg].astype(np.float32)
    vals = residual[bg].astype(np.float32)
    if len(vals) < 800:
        return -np.inf, 0.0

    bins = np.round(ph - ph.min()).astype(np.int32)
    size = int(bins.max()) + 1
    if size < 80 or size > h * 4:
        return -np.inf, 0.0
    counts = np.bincount(bins, minlength=size).astype(np.float32)
    ok = counts > max(8, np.percentile(counts[counts > 0], 20)) if np.any(counts > 0) else counts > 0
    if ok.sum() < 50:
        return -np.inf, 0.0

    prof, _ = binned_mean_profile(vals, bins, size)
    prof -= cv2.GaussianBlur(prof.reshape(-1, 1), (1, 61), 0).reshape(-1)
    prof *= np.hanning(len(prof)).astype(np.float32)

    spec = np.fft.rfft(prof)
    freqs = np.fft.rfftfreq(len(prof), d=1.0)
    band = (freqs >= 1.0 / period_range[1]) & (freqs <= 1.0 / period_range[0])
    if band.sum() < 3:
        return -np.inf, 0.0
    psd = np.abs(spec) ** 2
    idxs = np.flatnonzero(band)
    best = idxs[int(np.argmax(psd[band]))]
    period = 1.0 / max(freqs[best], 1e-6)
    local = psd[max(0, best - 1) : min(len(psd), best + 2)].sum()
    score = float(local / (psd[band].sum() + 1e-8)) * float(np.std(prof))
    return score, float(period)


def refine_period(values: np.ndarray, phase: np.ndarray, bg: np.ndarray, period0: float, harmonics: int) -> tuple[float, float]:
    best_period = period0
    best_score = -np.inf
    lo = max(6.0, period0 - 2.0)
    hi = period0 + 2.0
    for period in np.arange(lo, hi + 1e-6, 0.2):
        profile = harmonic_profile(values, phase, bg, float(period), harmonics)
        template = phase_to_field(profile, phase, float(period)).astype(np.float32)
        vals = values[bg]
        pred = template[bg]
        den = float(np.sum(pred * pred)) + 1e-6
        amp = float(np.sum(vals * pred) / den)
        recon = amp * pred
        before = float(np.sum(vals * vals)) + 1e-6
        after = float(np.sum((vals - recon) ** 2))
        score = 1.0 - after / before
        if score > best_score:
            best_score = score
            best_period = float(period)
    return best_period, best_score


def fit_component(
    residual: np.ndarray,
    bg: np.ndarray,
    comp: Component,
    harmonics: int,
) -> np.ndarray:
    crop = residual[:, comp.x0 : comp.x1]
    crop_bg = bg[:, comp.x0 : comp.x1]
    h, w = crop.shape
    total = np.zeros_like(crop, dtype=np.float32)
    total_weight = 0.0

    for y_shift, weight in zip(comp.y_shifts, comp.y_shift_weights):
        phase = parabola_phase(h, w, comp.center_x - comp.x0, comp.curvature, comp.sign, y_shift)
        period, _ = refine_period(crop, phase, crop_bg, comp.period, harmonics)
        profile = harmonic_profile(crop, phase, crop_bg, period, harmonics)
        template = phase_to_field(profile, phase, period).astype(np.float32)

        num = np.where(crop_bg, crop * template, 0).sum(axis=0)
        den = np.where(crop_bg, template * template, 0).sum(axis=0)
        amp_x = num / (den + 1e-5)
        valid_x = crop_bg.sum(axis=0) > max(8, 0.08 * h)
        amp_x = fill_missing_1d(amp_x.astype(np.float32), valid_x)
        amp_x = cv2.GaussianBlur(amp_x.reshape(1, -1), (0, 0), 20).reshape(-1)
        amp_x = np.clip(amp_x, -2.0, 2.0)
        field = cv2.GaussianBlur((template * amp_x[None, :]).astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.4)
        total += float(weight) * field
        total_weight += float(weight)

    return total / max(total_weight, 1e-6)


def find_best_component(
    residual: np.ndarray,
    bg: np.ndarray,
    x0: int,
    x1: int,
    curvature_range: tuple[float, float],
    period_range: tuple[float, float],
    coarse_center_steps: int,
    coarse_curvature_steps: int,
    fine_steps: int,
    signs: tuple[float, ...],
) -> Component | None:
    crop = residual[:, x0:x1]
    crop_bg = bg[:, x0:x1]
    if crop_bg.mean() < 0.12:
        return None

    width = x1 - x0
    best: Component | None = None
    centers = np.linspace(-0.3 * width, 1.3 * width, coarse_center_steps)
    curvatures = np.linspace(curvature_range[0], curvature_range[1], coarse_curvature_steps)
    for center in centers:
        for curvature in curvatures:
            for sign in signs:
                score, period = score_candidate(crop, crop_bg, float(center), float(curvature), sign, period_range)
                if best is None or score > best.score:
                    best = Component(x0, x1, x0 + float(center), float(curvature), sign, period, score)
    if best is None:
        return None

    coarse_center = best.center_x - x0
    coarse_curvature = best.curvature
    coarse_step = (curvature_range[1] - curvature_range[0]) / 12.0
    fine_centers = np.linspace(coarse_center - 0.12 * width, coarse_center + 0.12 * width, fine_steps)
    fine_curvatures = np.linspace(
        max(curvature_range[0], coarse_curvature - coarse_step),
        min(curvature_range[1], coarse_curvature + coarse_step),
        fine_steps,
    )
    for center in fine_centers:
        for curvature in fine_curvatures:
            score, period = score_candidate(crop, crop_bg, float(center), float(curvature), best.sign, period_range)
            if score > best.score:
                best = Component(x0, x1, x0 + float(center), float(curvature), best.sign, period, score)
    return best


def build_wavelet_residual(luma: np.ndarray, levels: int, band_start: int, band_end: int) -> np.ndarray:
    bands, _ = atrous_decompose(luma, levels)
    chosen = bands[band_start : band_end + 1]
    residual = np.sum(chosen, axis=0).astype(np.float32)
    residual -= cv2.GaussianBlur(residual, (0, 0), 45)
    return residual


def hanning(width: int) -> np.ndarray:
    win = np.hanning(width).astype(np.float32)
    return np.maximum(win, 0.08)


def robust_sigma(values: np.ndarray) -> float:
    values = values[np.isfinite(values)].astype(np.float32)
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def soften_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    soft = mask.astype(np.float32)
    if sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), sigma)
    return np.clip(soft, 0, 1)


def apply_noise_correction(
    luma: np.ndarray,
    noise: np.ndarray,
    support: np.ndarray,
    alpha: float,
    noise_clip_sigma: float,
    compensate_local_mean: bool,
    compensation_sigma: float,
    compensation_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    clipped_noise = noise.copy()
    valid_noise = clipped_noise[support]
    sigma = robust_sigma(valid_noise)
    if noise_clip_sigma > 0 and sigma > 1e-8:
        limit = noise_clip_sigma * sigma
        clipped_noise = np.clip(clipped_noise, -limit, limit)

    corrected_raw = luma - alpha * clipped_noise
    if not compensate_local_mean:
        return np.clip(corrected_raw, 0, 1), clipped_noise

    blur_sigma = max(float(compensation_sigma), 1.0)
    orig_mean = cv2.GaussianBlur(luma, (0, 0), blur_sigma)
    corr_mean = cv2.GaussianBlur(corrected_raw, (0, 0), blur_sigma)
    support_soft = soften_mask(support, max(blur_sigma * 0.5, 1.0))
    gain = np.clip(float(compensation_strength), 0, 1)
    corrected = corrected_raw + gain * support_soft * (orig_mean - corr_mean)
    return np.clip(corrected, 0, 1), clipped_noise


def parse_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if not text:
        return ranges
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("-", ":").split(":", 1)
        x0, x1 = int(left), int(right)
        if x1 < x0:
            x0, x1 = x1, x0
        ranges.append((x0, x1))
    return ranges


def range_allowed(x0: int, x1: int, include_ranges: list[tuple[int, int]]) -> bool:
    if not include_ranges:
        return True
    return any(max(x0, a) < min(x1, b) for a, b in include_ranges)


def draw_component_overlay(rgb: np.ndarray, components: list[Component], output_path: Path) -> None:
    overlay = rgb.copy()
    colors = [
        (0, 255, 255),
        (80, 220, 255),
        (40, 180, 255),
        (0, 255, 120),
        (255, 220, 40),
        (255, 120, 40),
    ]
    h, w = overlay.shape[:2]
    for idx, comp in enumerate(components):
        color = colors[idx % len(colors)]
        xs = np.arange(comp.x0, comp.x1, dtype=np.float32)
        local_center = comp.center_x
        for y_shift in comp.y_shifts:
            base = comp.sign * comp.curvature * (xs - local_center) ** 2 + y_shift
            # Draw several parallel ridges for this family using the fitted period.
            offsets = np.arange(-8, h + 8, max(comp.period, 1.0), dtype=np.float32)
            for off in offsets:
                ys = base + off
                pts = []
                for x, y in zip(xs, ys):
                    if 0 <= y < h:
                        pts.append([int(round(x)), int(round(y))])
                if len(pts) >= 2:
                    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), out)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb, luma, alpha = read_rgba_luma(args.input)
    annotation = white_annotation_mask(rgb)
    structure = structure_weight(luma, annotation) > args.structure_threshold
    bg = ~structure
    residual = build_wavelet_residual(luma, args.levels, args.band_start, args.band_end)

    h, w = luma.shape
    noise_sum = np.zeros_like(luma, dtype=np.float32)
    weight_sum = np.zeros_like(luma, dtype=np.float32)
    residual_work = residual.copy()
    components: list[Component] = []
    selected_per_window: dict[tuple[int, int], int] = {}
    include_ranges = parse_ranges(args.include_ranges)
    y_shifts = tuple(args.y_shifts if args.y_shifts is not None else [args.y_shift])
    if args.y_shift_weights is None:
        y_shift_weights = tuple([1.0] * len(y_shifts))
    else:
        y_shift_weights = tuple(args.y_shift_weights)
        if len(y_shift_weights) != len(y_shifts):
            raise ValueError("--y-shift-weights must have the same count as --y-shifts")

    starts = list(range(0, max(1, w - args.window_w + 1), args.stride))
    if not starts or starts[-1] != w - args.window_w:
        starts.append(max(0, w - args.window_w))

    for iteration in range(args.max_components):
        candidates: list[Component] = []
        for x0 in starts:
            x1 = min(w, x0 + args.window_w)
            if not range_allowed(x0, x1, include_ranges):
                continue
            if selected_per_window.get((x0, x1), 0) >= args.max_per_window:
                continue
            comp = find_best_component(
                residual_work,
                bg,
                x0,
                x1,
                tuple(args.curvature_range),
                tuple(args.period_range),
                args.coarse_center_steps,
                args.coarse_curvature_steps,
                args.fine_steps,
                tuple(args.signs),
            )
            if comp is not None:
                comp.y_shift = y_shifts[0]
                comp.y_shifts = y_shifts
                comp.y_shift_weights = y_shift_weights
                candidates.append(comp)
        if not candidates:
            break
        best = max(candidates, key=lambda c: c.score)
        if best.score < args.min_score:
            break

        field = fit_component(residual_work, bg, best, args.harmonics)
        width = best.x1 - best.x0
        win = hanning(width)
        noise_sum[:, best.x0 : best.x1] += field * win[None, :]
        weight_sum[:, best.x0 : best.x1] += win[None, :]
        residual_work[:, best.x0 : best.x1] -= args.beta * field
        components.append(best)
        selected_per_window[(best.x0, best.x1)] = selected_per_window.get((best.x0, best.x1), 0) + 1

        save_gray(
            args.output_dir / f"component_{iteration + 1:02d}_noise_x{best.x0}_{best.x1}.png",
            field,
            (0.2, 99.8),
        )

    noise = np.zeros_like(luma, dtype=np.float32)
    valid = weight_sum > 1e-5
    noise[valid] = noise_sum[valid] / weight_sum[valid]
    corrected_raw = np.clip(luma - args.alpha * noise, 0, 1)
    corrected, clipped_noise = apply_noise_correction(
        luma,
        noise,
        valid,
        args.alpha,
        args.noise_clip_sigma,
        args.compensate_local_mean,
        args.compensation_sigma,
        args.compensation_strength,
    )

    save_with_luma(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_corrected.png", rgb, corrected, alpha)
    save_with_luma(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_corrected_raw.png", rgb, corrected_raw, alpha)
    save_gray(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_noise.png", noise, (0.2, 99.8))
    save_gray(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_noise_clipped.png", clipped_noise, (0.2, 99.8))
    save_gray(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_residual.png", residual, (0.2, 99.8))
    save_gray(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_residual_after.png", residual_work, (0.2, 99.8))
    cv2.imwrite(str(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_background_mask.png"), bg.astype(np.uint8) * 255)
    draw_component_overlay(rgb, components, args.output_dir / "pig_bile_duct_wavelet_multi_parabola_overlay.png")

    with open(args.output_dir / "pig_bile_duct_wavelet_multi_parabola_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"input: {args.input}\n")
        f.write(f"window_w: {args.window_w}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"max_components: {args.max_components}\n")
        f.write(f"max_per_window: {args.max_per_window}\n")
        f.write(f"used_components: {len(components)}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"beta: {args.beta}\n")
        f.write(f"include_ranges: {args.include_ranges}\n")
        f.write(f"y_shifts: {y_shifts}\n")
        f.write(f"y_shift_weights: {y_shift_weights}\n")
        f.write(f"noise_clip_sigma: {args.noise_clip_sigma}\n")
        f.write(f"compensate_local_mean: {args.compensate_local_mean}\n")
        f.write(f"compensation_sigma: {args.compensation_sigma}\n")
        f.write(f"compensation_strength: {args.compensation_strength}\n")
        f.write("idx,x0,x1,center_x,curvature,sign,period,score,y_shifts,y_shift_weights\n")
        for idx, comp in enumerate(components, 1):
            f.write(
                f"{idx},{comp.x0},{comp.x1},{comp.center_x:.3f},{comp.curvature:.7f},"
                f"{comp.sign:.1f},{comp.period:.3f},{comp.score:.8f},"
                f"{';'.join(f'{v:.3f}' for v in comp.y_shifts)},"
                f"{';'.join(f'{v:.3f}' for v in comp.y_shift_weights)}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Wavelet + local multi-parabola matching pursuit denoising.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("multi_parabola_outputs"))
    parser.add_argument("--levels", type=int, default=6)
    parser.add_argument("--band-start", type=int, default=1)
    parser.add_argument("--band-end", type=int, default=4)
    parser.add_argument("--window-w", type=int, default=300)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--max-per-window", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.0008)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--beta", type=float, default=0.80)
    parser.add_argument("--harmonics", type=int, default=4)
    parser.add_argument("--structure-threshold", type=float, default=0.22)
    parser.add_argument("--curvature-range", type=float, nargs=2, default=(0.0005, 0.0060))
    parser.add_argument("--period-range", type=float, nargs=2, default=(10.0, 46.0))
    parser.add_argument("--coarse-center-steps", type=int, default=7)
    parser.add_argument("--coarse-curvature-steps", type=int, default=9)
    parser.add_argument("--fine-steps", type=int, default=7)
    parser.add_argument("--signs", type=float, nargs="+", default=(-1.0, 1.0))
    parser.add_argument("--include-ranges", type=str, default="")
    parser.add_argument("--y-shift", type=float, default=0.0)
    parser.add_argument("--y-shifts", type=float, nargs="+", default=None)
    parser.add_argument("--y-shift-weights", type=float, nargs="+", default=None)
    parser.add_argument(
        "--noise-clip-sigma",
        type=float,
        default=0.0,
        help="Clip estimated noise to +/- this many robust sigmas inside the correction support. 0 disables clipping.",
    )
    parser.add_argument(
        "--compensate-local-mean",
        action="store_true",
        help="Restore the low-frequency local mean after noise subtraction to reduce dark oversubtraction lines.",
    )
    parser.add_argument(
        "--compensation-sigma",
        type=float,
        default=18.0,
        help="Gaussian sigma for local mean compensation.",
    )
    parser.add_argument(
        "--compensation-strength",
        type=float,
        default=0.5,
        help="0..1 strength of local mean compensation.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
