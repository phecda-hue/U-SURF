import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, map_coordinates, median_filter


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bscan_analysis import resolve_output_path  # noqa: E402
from jitter_spike_analysis import apply_large_motion_demons, estimate_dx, parse_runs  # noqa: E402


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def fft_amp(values, period):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if valid.sum() < 8:
        return np.nan
    x = values[valid] - np.nanmean(values[valid])
    spectrum = np.abs(np.fft.rfft(x))
    freq = np.fft.rfftfreq(x.size, d=1.0)
    idx = int(np.argmin(np.abs(freq - 1.0 / period)))
    return float(spectrum[idx])


def shift_bscan_lateral(image, dx):
    z, x = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        indexing="ij",
    )
    coords = np.array([z, x + dx])
    return map_coordinates(image, coords, order=1, mode="nearest")


def apply_lateral_correction(volume, correction):
    corrected = volume.astype(np.float64, copy=True)
    for idx, dx in enumerate(correction):
        if abs(dx) > 1e-8:
            corrected[:, :, idx] = shift_bscan_lateral(volume[:, :, idx], dx)
    return corrected


def sliding_harmonic_component(values, periods, window, step, min_valid_fraction):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    x = np.arange(n, dtype=np.float64)
    estimate = np.zeros(n, dtype=np.float64)
    weights_sum = np.zeros(n, dtype=np.float64)
    half = window // 2

    centers = list(range(half, max(half + 1, n - half), step))
    if centers[-1] != n - half - 1:
        centers.append(max(half, n - half - 1))

    for center in centers:
        start = max(0, center - half)
        end = min(n, center + half + 1)
        idx = np.arange(start, end)
        y = values[idx]
        valid = np.isfinite(y)
        if valid.sum() < max(8, int(idx.size * min_valid_fraction)):
            continue

        cols = [np.ones(valid.sum(), dtype=np.float64)]
        xv = x[idx][valid]
        for period in periods:
            omega = 2.0 * np.pi / period
            cols.append(np.sin(omega * xv))
            cols.append(np.cos(omega * xv))
        design = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(design, y[valid], rcond=None)

        pred_cols = []
        for period in periods:
            omega = 2.0 * np.pi / period
            pred_cols.append(np.sin(omega * x[idx]))
            pred_cols.append(np.cos(omega * x[idx]))
        pred_design = np.column_stack(pred_cols)
        pred = pred_design @ coef[1:]

        taper = np.hanning(idx.size)
        if not np.any(taper > 0):
            taper = np.ones(idx.size, dtype=np.float64)
        estimate[idx] += pred * taper
        weights_sum[idx] += taper

    valid_weight = weights_sum > 1e-8
    out = np.zeros(n, dtype=np.float64)
    out[valid_weight] = estimate[valid_weight] / weights_sum[valid_weight]
    return out


def make_map(volume):
    return np.log1p(np.max(np.abs(volume), axis=0))


def select_ridge_seeds(map_img, count, min_y, max_y, min_distance):
    y0 = max(0, min_y)
    y1 = min(map_img.shape[0], max_y)
    score = np.percentile(map_img[y0:y1], 97, axis=1) - np.percentile(map_img[y0:y1], 50, axis=1)
    order = np.argsort(score)[::-1]
    seeds = []
    for idx in order:
        y = int(idx + y0)
        if all(abs(y - prev) >= min_distance for prev in seeds):
            seeds.append(y)
        if len(seeds) >= count:
            break
    return sorted(seeds)


def track_ridge(map_img, seed_y, search_radius, centroid_halfwidth):
    n_y, n_x = map_img.shape
    smooth = gaussian_filter1d(map_img, sigma=1.0, axis=0)
    path = np.full(n_x, np.nan, dtype=np.float64)
    mid = n_x // 2

    def locate(k, center):
        lo = max(0, int(round(center)) - search_radius)
        hi = min(n_y, int(round(center)) + search_radius + 1)
        segment = smooth[lo:hi, k]
        if segment.size == 0:
            return np.nan
        peak = int(np.argmax(segment)) + lo
        c0 = max(0, peak - centroid_halfwidth)
        c1 = min(n_y, peak + centroid_halfwidth + 1)
        ys = np.arange(c0, c1, dtype=np.float64)
        vals = smooth[c0:c1, k]
        vals = vals - np.percentile(vals, 20)
        vals = np.clip(vals, 0, None)
        if vals.sum() <= 1e-8:
            return float(peak)
        return float(np.sum(ys * vals) / np.sum(vals))

    path[mid] = locate(mid, seed_y)
    for k in range(mid + 1, n_x):
        path[k] = locate(k, path[k - 1])
    for k in range(mid - 1, -1, -1):
        path[k] = locate(k, path[k + 1])
    return path


def ridge_fft_rows(stage_maps, seeds, args):
    rows = []
    for stage, map_img in stage_maps.items():
        for seed in seeds:
            y = track_ridge(map_img, seed, args.ridge_search_radius, args.ridge_centroid_halfwidth)
            slow = median_filter(y, size=args.ridge_trend_window, mode="nearest")
            residual = y - slow
            row = {
                "stage": stage,
                "seed_y": seed,
                "residual_median_abs": float(np.nanmedian(np.abs(residual))),
            }
            for period in args.periods:
                key = str(period).replace(".", "p")
                row[f"fft_{key}"] = fft_amp(residual, period)
            rows.append(row)
    return rows


def save_outputs(
    original,
    baseline,
    harmonic,
    final_volume,
    dx_a,
    dx_b,
    dx_c,
    dx_d,
    baseline_correction,
    harmonic_component,
    final_correction,
    ridge_rows,
    args,
):
    prefix = resolve_output_path(args.output_prefix)
    original_map = make_map(original)
    baseline_map = make_map(baseline)
    harmonic_map = make_map(harmonic)
    final_map = make_map(final_volume)
    extent = [1, original.shape[2], original.shape[1] - 1, 0]

    fig, axes = plt.subplots(5, 1, figsize=(17, 19), sharex=True)
    axes[0].imshow(original_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[0].set_title("A: Original MAP")
    axes[1].imshow(baseline_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[1].set_title("B: baseline no-spike trajectory MAP")
    axes[2].imshow(harmonic_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[2].set_title("C: baseline + sliding harmonic MAP")
    axes[3].imshow(final_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[3].set_title("D: baseline + sliding harmonic + Demons MAP")
    im = axes[4].imshow(final_map - baseline_map, cmap="coolwarm", aspect="auto", origin="upper", extent=extent)
    axes[4].set_title("D - B")
    fig.colorbar(im, ax=axes[4], fraction=0.015, pad=0.01)
    for ax in axes:
        ax.set_ylabel("A-line index")
    axes[4].set_xlabel("B-scan index")
    plt.tight_layout()
    path = resolve_output_path(f"{args.output_prefix}_maps.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    x = np.arange(1, original.shape[2] + 1)
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    axes[0].plot(x, dx_a, linewidth=0.7, label="A original")
    axes[0].plot(x, dx_b, linewidth=0.7, label="B baseline")
    axes[0].plot(x, dx_c, linewidth=0.7, label="C harmonic")
    axes[0].plot(x, dx_d, linewidth=0.7, label="D harmonic + Demons")
    axes[0].legend()
    axes[0].set_ylabel("re-measured dx")
    axes[1].plot(x, baseline_correction, linewidth=0.7, label="baseline correction")
    axes[1].plot(x, -args.harmonic_gain * harmonic_component, linewidth=0.7, label="harmonic increment")
    axes[1].legend()
    axes[1].set_ylabel("px")
    axes[2].plot(x, final_correction, linewidth=0.7, label="final correction")
    axes[2].legend()
    axes[2].set_ylabel("px")
    for period in args.periods:
        axes[3].axhline(1.0 / period, linewidth=0.8, linestyle="--", label=f"1/{period:g}")
    axes[3].plot([], [])
    axes[3].set_ylabel("target freq")
    axes[3].set_xlabel("B-scan")
    axes[3].legend()
    plt.tight_layout()
    path = resolve_output_path(f"{args.output_prefix}_trajectory.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    summary = []
    for stage, dx in [
        ("A_original", dx_a),
        ("B_baseline", dx_b),
        ("C_harmonic", dx_c),
        ("D_harmonic_plus_demons", dx_d),
    ]:
        row = {
            "stage": stage,
            "median_abs_dx": float(np.nanmedian(np.abs(dx))),
            "p95_abs_dx": float(np.nanpercentile(np.abs(dx), 95)),
            "max_abs_dx": float(np.nanmax(np.abs(dx))),
        }
        for period in args.periods:
            key = str(period).replace(".", "p")
            row[f"fft_{key}"] = fft_amp(dx, period)
        summary.append(row)

    summary_path = resolve_output_path(f"{args.output_prefix}_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {summary_path}")
    for row in summary:
        print(row)

    ridge_path = resolve_output_path(f"{args.output_prefix}_ridge_fft.csv")
    with ridge_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ridge_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ridge_rows)
    print(f"Saved: {ridge_path}")

    npz_path = resolve_output_path(f"{args.output_prefix}_results.npz")
    np.savez_compressed(
        npz_path,
        baseline_correction=baseline_correction,
        harmonic_component=harmonic_component,
        final_correction=final_correction,
        dx_original=dx_a,
        dx_baseline=dx_b,
        dx_harmonic=dx_c,
        dx_harmonic_plus_demons=dx_d,
    )
    print(f"Saved: {npz_path}")


def run(args):
    baseline = np.load(args.baseline_results)
    baseline_correction = np.asarray(baseline["correction"], dtype=np.float64)
    dx_baseline = np.asarray(baseline["dx_after_b"], dtype=np.float64)

    harmonic_component = sliding_harmonic_component(
        dx_baseline,
        args.periods,
        args.harmonic_window,
        args.harmonic_step,
        args.min_valid_fraction,
    )
    harmonic_component = np.clip(harmonic_component, -args.max_harmonic_correction, args.max_harmonic_correction)
    final_correction = baseline_correction - args.harmonic_gain * harmonic_component

    mat = loadmat(args.input)
    volume = np.asarray(mat[args.variable])
    if final_correction.size != volume.shape[2]:
        raise ValueError(f"Correction length {final_correction.size} does not match volume B-scans {volume.shape[2]}")

    print("Applying baseline correction once from raw volume...")
    baseline_volume = apply_lateral_correction(volume, baseline_correction)
    print("Applying baseline + harmonic correction once from raw volume...")
    harmonic_volume = apply_lateral_correction(volume, final_correction)
    print("Applying large-motion Demons after harmonic correction...")
    large_runs = parse_runs(args.large_runs)
    final_volume = apply_large_motion_demons(harmonic_volume, large_runs, args.demons_spline_offsets, args)

    print("Re-measuring dx trajectories...")
    dx_original = estimate_dx(volume, args.spline_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    dx_harmonic = estimate_dx(harmonic_volume, args.spline_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    dx_final = estimate_dx(final_volume, args.spline_offsets, args.block_count, args.upsample_factor, args.max_block_shift)

    original_map = make_map(volume)
    baseline_map = make_map(baseline_volume)
    harmonic_map = make_map(harmonic_volume)
    final_map = make_map(final_volume)
    seeds = select_ridge_seeds(
        baseline_map,
        args.ridge_count,
        args.ridge_min_y,
        args.ridge_max_y,
        args.ridge_min_distance,
    )
    ridge_rows = ridge_fft_rows(
        {
            "A_original": original_map,
            "B_baseline": baseline_map,
            "C_harmonic": harmonic_map,
            "D_harmonic_plus_demons": final_map,
        },
        seeds,
        args,
    )
    print(f"Ridge seeds: {seeds}")

    save_outputs(
        volume,
        baseline_volume,
        harmonic_volume,
        final_volume,
        dx_original,
        dx_baseline,
        dx_harmonic,
        dx_final,
        baseline_correction,
        harmonic_component,
        final_correction,
        ridge_rows,
        args,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to IHD3DPA1.mat")
    parser.add_argument("--variable", default="IHD3DPA1")
    parser.add_argument("--baseline-results", default="archive_outputs_20260813_175331/step8_hybrid_no_spike_aligned_results.npz")
    parser.add_argument("--output-prefix", default="step9_periodic_harmonic")
    parser.add_argument("--large-runs", default="151-153,326,428-430,979")
    parser.add_argument("--periods", type=parse_float_list, default=(2.25,))
    parser.add_argument("--harmonic-window", type=int, default=96)
    parser.add_argument("--harmonic-step", type=int, default=8)
    parser.add_argument("--harmonic-gain", type=float, default=0.7)
    parser.add_argument("--max-harmonic-correction", type=float, default=0.8)
    parser.add_argument("--min-valid-fraction", type=float, default=0.7)
    parser.add_argument("--spline-offsets", type=parse_int_list, default=(-3, -2, -1, 1, 2, 3))
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--upsample-factor", type=int, default=20)
    parser.add_argument("--max-block-shift", type=float, default=8.0)
    parser.add_argument("--demons-spline-offsets", type=parse_int_list, default=(-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
    parser.add_argument("--scales", type=parse_int_list, default=(4, 2, 1))
    parser.add_argument("--iterations", type=parse_int_list, default=(80, 60, 40))
    parser.add_argument("--sigma-update", type=float, default=0.8)
    parser.add_argument("--sigma-field", type=float, default=1.2)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--ridge-count", type=int, default=5)
    parser.add_argument("--ridge-min-y", type=int, default=180)
    parser.add_argument("--ridge-max-y", type=int, default=1450)
    parser.add_argument("--ridge-min-distance", type=int, default=90)
    parser.add_argument("--ridge-search-radius", type=int, default=12)
    parser.add_argument("--ridge-centroid-halfwidth", type=int, default=4)
    parser.add_argument("--ridge-trend-window", type=int, default=31)
    run(parser.parse_args())
