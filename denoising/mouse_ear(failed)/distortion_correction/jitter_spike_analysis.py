import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.ndimage import map_coordinates, median_filter
from skimage.registration import phase_cross_correlation


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bscan_analysis import (  # noqa: E402
    modified_symmetric_demons,
    normalize_with_limits,
    parse_int_list,
    resolve_output_path,
    transform_image,
    warp_image,
)


def parse_runs(value):
    runs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            item = item.split(":", 1)[0]
        if "-" in item:
            start, end = item.split("-", 1)
            runs.append((int(start), int(end)))
        else:
            k = int(item)
            runs.append((k, k))
    return runs


def in_runs(bscan, runs):
    return any(start <= bscan <= end for start, end in runs)


def load_summary(path, large_runs):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bscan = int(float(row["bscan"]))
            rows.append(
                {
                    "bscan": bscan,
                    "dx": float(row["dx"]) if row["dx"] != "nan" else np.nan,
                    "dz": float(row["dz"]) if row["dz"] != "nan" else np.nan,
                    "valid_blocks": int(float(row["valid_blocks"])),
                    "sign_agreement": float(row["sign_agreement"]) if row["sign_agreement"] != "nan" else np.nan,
                    "block_mad_dx": float(row["block_mad_dx"]) if row["block_mad_dx"] != "nan" else np.nan,
                    "large_motion": in_runs(bscan, large_runs),
                }
            )
    return rows


def load_block_dx(path, max_block_shift):
    block_values = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["valid"])) != 1:
                continue
            if float(row["mag"]) > max_block_shift:
                continue
            bscan = int(float(row["bscan"]))
            block = int(float(row["block"]))
            block_values[block][bscan] = float(row["dx"])
    return block_values


def reliable_mask(rows, min_blocks, min_agreement, max_mad):
    mask = np.zeros(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        mask[i] = (
            np.isfinite(row["dx"])
            and row["valid_blocks"] >= min_blocks
            and np.isfinite(row["sign_agreement"])
            and row["sign_agreement"] >= min_agreement
            and np.isfinite(row["block_mad_dx"])
            and row["block_mad_dx"] <= max_mad
            and not row["large_motion"]
        )
    return mask


def interpolate_short_gaps(values, valid, max_gap):
    values = np.asarray(values, dtype=np.float64)
    result = np.full_like(values, np.nan)
    result[valid] = values[valid]
    good = np.flatnonzero(np.isfinite(result))
    if good.size < 2:
        return np.nan_to_num(result)
    interp = np.interp(np.arange(values.size), good, result[good])
    missing = ~np.isfinite(result)
    i = 0
    while i < values.size:
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < values.size and missing[i]:
            i += 1
        end = i - 1
        if end - start + 1 <= max_gap and start > 0 and end < values.size - 1:
            result[start : end + 1] = interp[start : end + 1]
        else:
            result[start : end + 1] = 0.0
    return np.nan_to_num(result)


def circular_stats(phases, weights):
    phases = np.asarray(phases)
    weights = np.asarray(weights)
    z = np.sum(weights * np.exp(1j * phases)) / (np.sum(weights) + 1e-8)
    mean_phase = float(np.angle(z))
    coherence = float(np.abs(z))
    return mean_phase, coherence


def block_frequency_phase(block_values, bscans, target_period):
    target_freq = 1.0 / target_period
    rows = []
    for block, values in sorted(block_values.items()):
        series = np.full(bscans.size, np.nan, dtype=np.float64)
        index_map = {int(b): i for i, b in enumerate(bscans)}
        for bscan, dx in values.items():
            if bscan in index_map:
                series[index_map[bscan]] = dx
        valid = np.isfinite(series)
        if valid.sum() < bscans.size * 0.3:
            continue
        good = np.flatnonzero(valid)
        filled = np.interp(np.arange(bscans.size), good, series[good])
        y = filled - np.mean(filled)
        spectrum = np.fft.rfft(y)
        freq = np.fft.rfftfreq(y.size, d=1.0)
        idx = int(np.argmin(np.abs(freq - target_freq)))
        rows.append(
            {
                "block": block,
                "amplitude": float(np.abs(spectrum[idx])),
                "phase": float(np.angle(spectrum[idx])),
                "freq": float(freq[idx]),
                "period": float(1.0 / freq[idx]) if freq[idx] > 0 else np.inf,
            }
        )
    return rows


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
    for i, dx in enumerate(correction):
        if abs(dx) > 1e-8:
            corrected[:, :, i] = shift_bscan_lateral(volume[:, :, i], dx)
    return corrected


def expand_to_volume(values, bscan, n_bscans, fill=0.0):
    full = np.full(n_bscans, fill, dtype=np.float64)
    full[np.asarray(bscan, dtype=int) - 1] = values
    return full


def make_spline_k1(volume, k, offsets, exclude_start, exclude_end):
    idx = k - 1
    excluded = set(range(exclude_start - 1, exclude_end))
    neighbors = []
    for offset in offsets:
        ni = idx + offset
        if ni in excluded:
            continue
        if 0 <= ni < volume.shape[2]:
            neighbors.append(ni)
    neighbors = np.asarray(sorted(set(neighbors)), dtype=int)
    spline = CubicSpline(
        neighbors.astype(np.float64),
        volume[:, :, neighbors].astype(np.float64),
        axis=2,
        bc_type="not-a-knot",
    )
    return spline(float(idx))


def common_limits(images):
    values = np.concatenate([transform_image(img).ravel() for img in images])
    return np.percentile(values, 1), np.percentile(values, 99.5)


def apply_large_motion_demons(volume, runs, offsets, args):
    corrected = volume.astype(np.float64, copy=True)
    for start, end in runs:
        for k in range(start, end + 1):
            print(f"Demons for non-coherent/large motion B-scan {k}")
            idx = k - 1
            curr = corrected[:, :, idx]
            k1 = make_spline_k1(corrected, k, offsets, start, end)
            lo, hi = common_limits([curr, k1])
            curr_norm = normalize_with_limits(curr, lo, hi)
            k1_norm = normalize_with_limits(k1, lo, hi)
            dz, dx, _ = modified_symmetric_demons(
                reference=k1_norm,
                floating=curr_norm,
                scales=args.scales,
                iterations=args.iterations,
                sigma_update=args.sigma_update,
                sigma_field=args.sigma_field,
                max_step=args.max_step,
            )
            corrected[:, :, idx] = warp_image(curr, dz, dx, order=1)
    return corrected


def estimate_dx(volume, offsets, block_count, upsample_factor, max_block_shift):
    transformed = transform_image(volume)
    lo = np.percentile(transformed, 1)
    hi = np.percentile(transformed, 99.5)
    offsets = np.asarray(offsets, dtype=int)
    basis = np.eye(offsets.size)
    weights = []
    for i in range(offsets.size):
        weights.append(float(CubicSpline(offsets, basis[:, i])(0.0)))
    weights = np.asarray(weights)
    dx = np.full(volume.shape[2], np.nan)
    valid_start = max(0, -int(np.min(offsets)))
    valid_end = min(volume.shape[2] - 1, volume.shape[2] - 1 - int(np.max(offsets)))
    edges = np.linspace(0, volume.shape[1], block_count + 1, dtype=int)
    for idx in range(valid_start, valid_end + 1):
        k1 = np.zeros(volume.shape[:2], dtype=np.float64)
        for offset, weight in zip(offsets, weights):
            k1 += weight * volume[:, :, idx + offset]
        k_norm = normalize_with_limits(volume[:, :, idx], lo, hi)
        k1_norm = normalize_with_limits(k1, lo, hi)
        vals = []
        for s, e in zip(edges[:-1], edges[1:]):
            ref = k1_norm[:, s:e]
            mov = k_norm[:, s:e]
            if np.std(ref) <= 1e-8 or np.std(mov) <= 1e-8:
                continue
            shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=upsample_factor, normalization=None)
            if np.hypot(float(shift[0]), float(shift[1])) <= max_block_shift:
                vals.append(float(shift[1]))
        if vals:
            dx[idx] = float(np.median(vals))
    return dx


def fft_amp(dx, period):
    valid = np.isfinite(dx)
    y = dx[valid] - np.nanmean(dx[valid])
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, d=1.0)
    idx = int(np.argmin(np.abs(freq - 1.0 / period)))
    return float(spec[idx])


def save_outputs(volume, b_volume, c_volume, rows, phase_rows, correction, spike_mask, dx_after_b, dx_after_c, args):
    original_map = np.log1p(np.max(np.abs(volume), axis=0))
    b_map = np.log1p(np.max(np.abs(b_volume), axis=0))
    c_map = np.log1p(np.max(np.abs(c_volume), axis=0))
    extent = [1, volume.shape[2], volume.shape[1] - 1, 0]

    fig, axes = plt.subplots(4, 1, figsize=(17, 16), sharex=True)
    axes[0].imshow(original_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[0].set_title("A: Original MAP")
    axes[1].imshow(b_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[1].set_title("B: hybrid translation correction MAP")
    axes[2].imshow(c_map, cmap="hot", aspect="auto", origin="upper", extent=extent)
    axes[2].set_title("C: hybrid translation + Demons MAP")
    im = axes[3].imshow(c_map - original_map, cmap="coolwarm", aspect="auto", origin="upper", extent=extent)
    axes[3].set_title("C - A")
    fig.colorbar(im, ax=axes[3], fraction=0.015, pad=0.01)
    for ax in axes:
        ax.set_ylabel("A-line index")
    axes[3].set_xlabel("B-scan index")
    plt.tight_layout()
    path = resolve_output_path(f"{args.output_prefix}_maps.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    bscan = np.array([r["bscan"] for r in rows])
    dx = np.array([r["dx"] for r in rows])
    dx_filled = np.array([r["dx_filled"] for r in rows])
    dx_slow = np.array([r["dx_slow"] for r in rows])
    dx_jitter = np.array([r["dx_jitter"] for r in rows])
    local_residual = np.array([r["local_residual"] for r in rows])

    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
    axes[0].plot(bscan, dx, linewidth=0.7, label="dx")
    axes[0].plot(bscan, dx_slow, linewidth=1.0, label="slow")
    axes[0].scatter(bscan[spike_mask], dx[spike_mask], s=12, color="red", label="coherent spikes")
    axes[0].legend()
    axes[0].set_ylabel("dx")

    axes[1].plot(bscan, dx_jitter, linewidth=0.7, label="continuous jitter")
    axes[1].plot(bscan, local_residual, linewidth=0.7, label="local residual")
    axes[1].legend()
    axes[1].set_ylabel("px")

    axes[2].plot(bscan, correction[bscan - 1], linewidth=0.7)
    axes[2].set_ylabel("applied correction")

    axes[3].plot(np.arange(1, dx_after_b.size + 1), dx_after_b, linewidth=0.7, label="after B")
    axes[3].plot(np.arange(1, dx_after_c.size + 1), dx_after_c, linewidth=0.7, label="after C")
    axes[3].legend()
    axes[3].set_ylabel("re-measured dx")
    axes[3].set_xlabel("B-scan")
    plt.tight_layout()
    path = resolve_output_path(f"{args.output_prefix}_classification_and_trajectory.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    phase_csv = resolve_output_path(f"{args.output_prefix}_block_phase_2p25.csv")
    with phase_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase_rows[0].keys()))
        writer.writeheader()
        writer.writerows(phase_rows)
    print(f"Saved: {phase_csv}")

    phase_values = np.array([r["phase"] for r in phase_rows])
    amps = np.array([r["amplitude"] for r in phase_rows])
    mean_phase, coherence = circular_stats(phase_values, amps)

    summary_csv = resolve_output_path(f"{args.output_prefix}_summary.csv")
    summary = [
        {
            "stage": "A_original",
            "fft_2p25": fft_amp(dx, 2.25),
            "fft_3p0": fft_amp(dx, 3.0),
            "median_abs_dx": float(np.nanmedian(np.abs(dx))),
            "p95_abs_dx": float(np.nanpercentile(np.abs(dx), 95)),
            "max_abs_dx": float(np.nanmax(np.abs(dx))),
            "spike_count": int(spike_mask.sum()),
            "block_phase_coherence_2p25": coherence,
            "block_mean_phase_2p25": mean_phase,
        },
        {
            "stage": "B_hybrid_translation",
            "fft_2p25": fft_amp(dx_after_b, 2.25),
            "fft_3p0": fft_amp(dx_after_b, 3.0),
            "median_abs_dx": float(np.nanmedian(np.abs(dx_after_b))),
            "p95_abs_dx": float(np.nanpercentile(np.abs(dx_after_b), 95)),
            "max_abs_dx": float(np.nanmax(np.abs(dx_after_b))),
            "spike_count": int(spike_mask.sum()),
            "block_phase_coherence_2p25": coherence,
            "block_mean_phase_2p25": mean_phase,
        },
        {
            "stage": "C_hybrid_plus_demons",
            "fft_2p25": fft_amp(dx_after_c, 2.25),
            "fft_3p0": fft_amp(dx_after_c, 3.0),
            "median_abs_dx": float(np.nanmedian(np.abs(dx_after_c))),
            "p95_abs_dx": float(np.nanpercentile(np.abs(dx_after_c), 95)),
            "max_abs_dx": float(np.nanmax(np.abs(dx_after_c))),
            "spike_count": int(spike_mask.sum()),
            "block_phase_coherence_2p25": coherence,
            "block_mean_phase_2p25": mean_phase,
        },
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved: {summary_csv}")
    for row in summary:
        print(row)

    npz_path = resolve_output_path(f"{args.output_prefix}_results.npz")
    np.savez_compressed(
        npz_path,
        correction=correction,
        spike_mask=spike_mask,
        dx=dx,
        bscan=bscan,
        dx_after_b=dx_after_b,
        dx_after_c=dx_after_c,
    )
    print(f"Saved: {npz_path}")


def run(args):
    large_runs = parse_runs(args.large_runs)
    rows = load_summary(args.coherence_summary, large_runs)
    bscan = np.array([r["bscan"] for r in rows])
    dx = np.array([r["dx"] for r in rows], dtype=np.float64)
    reliable = reliable_mask(rows, args.min_valid_blocks, args.min_sign_agreement, args.max_block_mad)
    dx_filled = interpolate_short_gaps(dx, reliable, args.max_gap)
    dx_slow = median_filter(dx_filled, size=args.trend_window, mode="nearest")
    dx_jitter = dx_filled - dx_slow
    local_baseline = median_filter(dx_filled, size=args.spike_window, mode="nearest")
    local_residual = dx_filled - local_baseline

    spike_mask = (
        reliable
        & (np.abs(local_residual) >= args.spike_threshold)
        & np.array([not r["large_motion"] for r in rows], dtype=bool)
    )

    for i, row in enumerate(rows):
        row["dx_filled"] = float(dx_filled[i])
        row["dx_slow"] = float(dx_slow[i])
        row["dx_jitter"] = float(dx_jitter[i])
        row["local_residual"] = float(local_residual[i])
        row["coherent_spike"] = bool(spike_mask[i])

    continuous = -np.clip(dx_jitter, -args.max_continuous_correction, args.max_continuous_correction)
    spike_correction = np.zeros_like(continuous)
    spike_correction[spike_mask] = -np.clip(
        local_residual[spike_mask],
        -args.max_spike_correction,
        args.max_spike_correction,
    )

    mat = loadmat(args.input)
    volume = np.asarray(mat[args.variable])

    correction_subset = continuous.copy()
    if args.spike_mode == "add":
        correction_subset += spike_correction
    elif args.spike_mode == "replace":
        correction_subset[spike_mask] = spike_correction[spike_mask]
    else:
        raise ValueError(f"Unknown spike mode: {args.spike_mode}")
    correction = expand_to_volume(correction_subset, bscan, volume.shape[2], fill=0.0)
    for start, end in large_runs:
        correction[start - 1 : end] = 0.0

    block_values = load_block_dx(args.block_shifts, args.max_block_shift)
    phase_rows = block_frequency_phase(block_values, bscan, args.phase_period)
    if phase_rows:
        phases = np.array([r["phase"] for r in phase_rows])
        amps = np.array([r["amplitude"] for r in phase_rows])
        _, coh = circular_stats(phases, amps)
        print(f"Block phase coherence at period {args.phase_period}: {coh:.4f}")

    b_volume = apply_lateral_correction(volume, correction)
    c_volume = apply_large_motion_demons(b_volume, large_runs, args.demons_spline_offsets, args)
    dx_after_b = estimate_dx(b_volume, args.spline_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    dx_after_c = estimate_dx(c_volume, args.spline_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    save_outputs(volume, b_volume, c_volume, rows, phase_rows, correction, spike_mask, dx_after_b, dx_after_c, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to IHD3DPA1.mat")
    parser.add_argument("--variable", default="IHD3DPA1")
    parser.add_argument("--coherence-summary", default="archive_outputs_20260813_175331/step6_block_coherence_summary.csv")
    parser.add_argument("--block-shifts", default="archive_outputs_20260813_175331/step6_shift_trajectory_block_shifts.csv")
    parser.add_argument("--output-prefix", default="step8_hybrid_jitter_spike")
    parser.add_argument("--large-runs", default="151-153,326,428-430,979")
    parser.add_argument("--min-valid-blocks", type=int, default=5)
    parser.add_argument("--min-sign-agreement", type=float, default=0.75)
    parser.add_argument("--max-block-mad", type=float, default=1.0)
    parser.add_argument("--trend-window", type=int, default=31)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--spike-window", type=int, default=7)
    parser.add_argument("--spike-threshold", type=float, default=1.5)
    parser.add_argument("--max-continuous-correction", type=float, default=1.5)
    parser.add_argument("--max-spike-correction", type=float, default=3.0)
    parser.add_argument("--spike-mode", choices=("replace", "add"), default="replace")
    parser.add_argument("--phase-period", type=float, default=2.25)
    parser.add_argument("--spline-offsets", type=parse_int_list, default=(-3, -2, -1, 1, 2, 3))
    parser.add_argument("--demons-spline-offsets", type=parse_int_list, default=(-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--upsample-factor", type=int, default=20)
    parser.add_argument("--max-block-shift", type=float, default=8.0)
    parser.add_argument("--scales", type=parse_int_list, default=(4, 2, 1))
    parser.add_argument("--iterations", type=parse_int_list, default=(80, 60, 40))
    parser.add_argument("--sigma-update", type=float, default=0.8)
    parser.add_argument("--sigma-field", type=float, default=1.4)
    parser.add_argument("--max-step", type=float, default=1.0)

    args = parser.parse_args()
    if len(args.scales) != len(args.iterations):
        raise ValueError("--scales and --iterations must have the same length")
    run(args)
