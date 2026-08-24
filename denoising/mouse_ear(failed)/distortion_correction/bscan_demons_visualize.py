import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from bscan_analysis import modified_symmetric_demons, normalize_with_limits, parse_int_list, warp_image  # noqa: E402
from jitter_spike_analysis import estimate_dx  # noqa: E402
from periodic_harmonic_correction import apply_lateral_correction, fft_amp, make_map  # noqa: E402


def common_limits(images):
    values = np.concatenate([img.ravel() for img in images])
    return np.percentile(values, 1), np.percentile(values, 99.5)


def make_reference_from_neighbors(volume, idx, offsets):
    neighbors = []
    for offset in offsets:
        ni = idx + offset
        if 0 <= ni < volume.shape[2]:
            neighbors.append(ni)
    neighbors = np.asarray(sorted(set(neighbors)), dtype=int)
    if neighbors.size < 2:
        return None
    if neighbors.size < 4:
        return np.mean(volume[:, :, neighbors], axis=2)
    spline = CubicSpline(
        neighbors.astype(np.float64),
        volume[:, :, neighbors].astype(np.float64),
        axis=2,
        bc_type="not-a-knot",
    )
    return spline(float(idx))


def summarize_dx(stages, periods):
    rows = []
    for stage, dx in stages:
        row = {
            "stage": stage,
            "median_abs_dx": float(np.nanmedian(np.abs(dx))),
            "p95_abs_dx": float(np.nanpercentile(np.abs(dx), 95)),
            "max_abs_dx": float(np.nanmax(np.abs(dx))),
        }
        for period in periods:
            key = str(period).replace(".", "p")
            row[f"fft_{key}"] = fft_amp(dx, period)
        rows.append(row)
    return rows


def save_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_visuals(output_dir, original, harmonic, all_demons):
    maps = {
        "A Original": make_map(original),
        "B Harmonic correction": make_map(harmonic),
        "C All B-scan Demons": make_map(all_demons),
    }
    samples = np.concatenate([img.ravel()[::20] for img in maps.values()])
    vmin, vmax = np.percentile(samples, [1, 99.5])
    extent = [1, original.shape[2], original.shape[1] - 1, 0]

    fig, axes = plt.subplots(4, 1, figsize=(17, 16), sharex=True)
    for ax, (title, img) in zip(axes[:3], maps.items()):
        ax.imshow(img, cmap="hot", aspect="auto", origin="upper", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("A-line index")
    diff = maps["C All B-scan Demons"] - maps["B Harmonic correction"]
    lim = np.percentile(np.abs(diff), 99)
    im = axes[3].imshow(diff, cmap="coolwarm", aspect="auto", origin="upper", extent=extent, vmin=-lim, vmax=lim)
    axes[3].set_title("C - B")
    axes[3].set_ylabel("A-line index")
    axes[3].set_xlabel("B-scan index")
    fig.colorbar(im, ax=axes[3], fraction=0.015, pad=0.01)
    plt.tight_layout()
    fig.savefig(output_dir / "all_bscan_demons_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat = loadmat(args.input)
    volume = np.asarray(mat[args.variable])
    step9 = np.load(args.step9_results)
    final_correction = np.asarray(step9["final_correction"], dtype=np.float64)
    harmonic_volume = apply_lateral_correction(volume, final_correction)

    offsets = np.asarray(args.reference_offsets, dtype=int)
    corrected = harmonic_volume.astype(np.float64, copy=True)
    source = harmonic_volume.astype(np.float64, copy=False)
    applied = []

    total = source.shape[2]
    for idx in range(total):
        ref = make_reference_from_neighbors(source, idx, offsets)
        if ref is None:
            continue
        curr = source[:, :, idx]
        lo, hi = common_limits([curr, ref])
        curr_norm = normalize_with_limits(curr, lo, hi)
        ref_norm = normalize_with_limits(ref, lo, hi)
        dz, dx, _ = modified_symmetric_demons(
            reference=ref_norm,
            floating=curr_norm,
            scales=args.scales,
            iterations=args.iterations,
            sigma_update=args.sigma_update,
            sigma_field=args.sigma_field,
            max_step=args.max_step,
        )
        corrected[:, :, idx] = warp_image(curr, dz, dx, order=1)
        applied.append(idx + 1)
        if (idx + 1) % args.progress_every == 0:
            print(f"Processed {idx + 1}/{total} B-scans")

    np.savez_compressed(
        output_dir / "all_bscan_demons_results.npz",
        applied_bscans=np.asarray(applied, dtype=np.int32),
    )
    save_visuals(output_dir, volume, harmonic_volume, corrected)

    dx_original = estimate_dx(volume, args.dx_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    dx_harmonic = estimate_dx(harmonic_volume, args.dx_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    dx_all_demons = estimate_dx(corrected, args.dx_offsets, args.block_count, args.upsample_factor, args.max_block_shift)
    rows = summarize_dx(
        [
            ("A_original", dx_original),
            ("B_harmonic", dx_harmonic),
            ("C_all_bscan_demons", dx_all_demons),
        ],
        args.periods,
    )
    save_summary(output_dir / "all_bscan_demons_summary.csv", rows)

    print(f"Saved outputs to: {output_dir}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to IHD3DPA1.mat")
    parser.add_argument("--variable", default="IHD3DPA1")
    parser.add_argument(
        "--step9-results",
        default="denoising/mouse_ear(failed)/distortion_correction/outputs/step9/step9_from_step8_gain0p4_demons_results.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="denoising/mouse_ear(failed)/distortion_correction/outputs/step9_all_bscan_demons",
    )
    parser.add_argument("--reference-offsets", type=parse_int_list, default=(-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
    parser.add_argument("--dx-offsets", type=parse_int_list, default=(-3, -2, -1, 1, 2, 3))
    parser.add_argument("--periods", type=lambda v: tuple(float(x.strip()) for x in v.split(",") if x.strip()), default=(2.25, 3.0))
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--upsample-factor", type=int, default=20)
    parser.add_argument("--max-block-shift", type=float, default=8.0)
    parser.add_argument("--scales", type=parse_int_list, default=(4, 2, 1))
    parser.add_argument("--iterations", type=parse_int_list, default=(20, 15, 10))
    parser.add_argument("--sigma-update", type=float, default=0.8)
    parser.add_argument("--sigma-field", type=float, default=1.2)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    run(parser.parse_args())
