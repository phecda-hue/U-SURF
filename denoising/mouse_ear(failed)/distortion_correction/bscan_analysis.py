import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


CURRENT_DIR = Path(__file__).resolve().parent


def resolve_output_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = CURRENT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def transform_image(img):
    img = np.abs(np.asarray(img, dtype=np.float64))
    return np.log1p(img)


def normalize_with_limits(img, lo, hi):
    img = transform_image(img)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def normalize_triplet_common(prev_img, curr_img, next_img, p_low=1, p_high=99.5):
    triplet = np.stack(
        [
            transform_image(prev_img),
            transform_image(curr_img),
            transform_image(next_img),
        ],
        axis=0,
    )

    lo = np.percentile(triplet, p_low)
    hi = np.percentile(triplet, p_high)

    return (
        np.clip((triplet[0] - lo) / (hi - lo + 1e-8), 0, 1),
        np.clip((triplet[1] - lo) / (hi - lo + 1e-8), 0, 1),
        np.clip((triplet[2] - lo) / (hi - lo + 1e-8), 0, 1),
        lo,
        hi,
    )


def mutual_information_2d(a, b, num_bins=64):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if a.size == 0:
        return np.nan

    hist, _, _ = np.histogram2d(
        a,
        b,
        bins=num_bins,
        range=[[0, 1], [0, 1]],
    )

    total = hist.sum()
    if total <= 0:
        return np.nan

    pxy = hist / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    pxpy = px @ py

    valid = (pxy > 0) & (pxpy > 0)
    return float(np.sum(pxy[valid] * np.log2(pxy[valid] / pxpy[valid])))


def load_triplet_from_mat(mat_path, variable, k):
    mat_path = Path(mat_path)
    idx = k - 1

    try:
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        volume = mat[variable]

        if idx <= 0 or idx >= volume.shape[2] - 1:
            raise ValueError(
                f"k must be between 2 and {volume.shape[2] - 1}, got {k}"
            )

        return (
            volume[:, :, idx - 1],
            volume[:, :, idx],
            volume[:, :, idx + 1],
        )

    except NotImplementedError:
        pass
    except ValueError as e:
        if "Unknown mat file" not in str(e) and "HDF" not in str(e):
            raise

    import h5py

    with h5py.File(mat_path, "r") as h5:
        volume = h5[variable]
        num_bscan = volume.shape[0]

        if idx <= 0 or idx >= num_bscan - 1:
            raise ValueError(
                f"k must be between 2 and {num_bscan - 1}, got {k}"
            )

        return (
            np.asarray(volume[idx - 1, :, :]).T,
            np.asarray(volume[idx, :, :]).T,
            np.asarray(volume[idx + 1, :, :]).T,
        )


def load_bscans_from_mat(mat_path, variable, indices):
    mat_path = Path(mat_path)
    indices = list(indices)

    try:
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        volume = mat[variable]

        if min(indices) < 0 or max(indices) >= volume.shape[2]:
            raise ValueError(
                f"indices must be between 0 and {volume.shape[2] - 1}, got {indices}"
            )

        return np.stack([np.asarray(volume[:, :, i]) for i in indices], axis=2)

    except NotImplementedError:
        pass
    except ValueError as e:
        if "Unknown mat file" not in str(e) and "HDF" not in str(e):
            raise

    import h5py

    with h5py.File(mat_path, "r") as h5:
        volume = h5[variable]
        if min(indices) < 0 or max(indices) >= volume.shape[0]:
            raise ValueError(
                f"indices must be between 0 and {volume.shape[0] - 1}, got {indices}"
            )

        return np.stack([np.asarray(volume[i, :, :]).T for i in indices], axis=2)


def get_volume_shape(mat_path, variable):
    mat_path = Path(mat_path)

    try:
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        shape = mat[variable].shape
        return shape[0], shape[1], shape[2], "scipy"

    except NotImplementedError:
        pass
    except ValueError as e:
        if "Unknown mat file" not in str(e) and "HDF" not in str(e):
            raise

    import h5py

    with h5py.File(mat_path, "r") as h5:
        bscan, aline, depth = h5[variable].shape
        return depth, aline, bscan, "h5py"


def iter_bscans(mat_path, variable):
    mat_path = Path(mat_path)

    try:
        from scipy.io import loadmat

        mat = loadmat(mat_path)
        volume = mat[variable]
        for i in range(volume.shape[2]):
            yield np.asarray(volume[:, :, i])
        return

    except NotImplementedError:
        pass
    except ValueError as e:
        if "Unknown mat file" not in str(e) and "HDF" not in str(e):
            raise

    import h5py

    with h5py.File(mat_path, "r") as h5:
        volume = h5[variable]
        for i in range(volume.shape[0]):
            yield np.asarray(volume[i, :, :]).T


def warp_image(image, dz, dx, order=1):
    z, x = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        indexing="ij",
    )
    coords = np.array([z + dz, x + dx])
    return map_coordinates(image, coords, order=order, mode="nearest")


def resize_image(image, scale):
    if scale == 1:
        return image.astype(np.float64, copy=True)
    factor = 1.0 / scale
    return zoom(image, zoom=(factor, factor), order=1)


def resize_field(field, target_shape, multiplier):
    zoom_factors = (
        target_shape[0] / field.shape[0],
        target_shape[1] / field.shape[1],
    )
    return zoom(field, zoom=zoom_factors, order=1) * multiplier


def modified_symmetric_demons(
    reference,
    floating,
    scales=(4, 2, 1),
    iterations=(80, 60, 40),
    sigma_update=0.8,
    sigma_field=1.2,
    max_step=1.0,
):
    dz = None
    dx = None
    previous_scale = None

    for scale, n_iter in zip(scales, iterations):
        ref_s = resize_image(reference, scale)
        flo_s = resize_image(floating, scale)

        if dz is None:
            dz = np.zeros_like(ref_s, dtype=np.float64)
            dx = np.zeros_like(ref_s, dtype=np.float64)
        else:
            multiplier = previous_scale / scale
            dz = resize_field(dz, ref_s.shape, multiplier)
            dx = resize_field(dx, ref_s.shape, multiplier)

        grad_ref_z, grad_ref_x = np.gradient(ref_s)

        for _ in range(n_iter):
            warped = warp_image(flo_s, dz, dx, order=1)
            diff = ref_s - warped

            grad_warp_z, grad_warp_x = np.gradient(warped)
            denom_ref = grad_ref_z**2 + grad_ref_x**2 + diff**2 + 1e-8
            denom_warp = grad_warp_z**2 + grad_warp_x**2 + diff**2 + 1e-8

            update_z = (
                diff * grad_ref_z / denom_ref
                + diff * grad_warp_z / denom_warp
            )
            update_x = (
                diff * grad_ref_x / denom_ref
                + diff * grad_warp_x / denom_warp
            )

            update_mag = np.sqrt(update_z**2 + update_x**2)
            limited = update_mag > max_step
            update_z[limited] *= max_step / (update_mag[limited] + 1e-8)
            update_x[limited] *= max_step / (update_mag[limited] + 1e-8)

            if sigma_update > 0:
                update_z = gaussian_filter(update_z, sigma_update)
                update_x = gaussian_filter(update_x, sigma_update)

            dz += update_z
            dx += update_x

            if sigma_field > 0:
                dz = gaussian_filter(dz, sigma_field)
                dx = gaussian_filter(dx, sigma_field)

        previous_scale = scale

    return dz, dx, warp_image(floating, dz, dx, order=1)


def displacement_stats(dz, dx):
    mag = np.sqrt(dz**2 + dx**2)
    dz_dz, dz_dx = np.gradient(dz)
    dx_dz, dx_dx = np.gradient(dx)
    jacobian = (1 + dz_dz) * (1 + dx_dx) - dz_dx * dx_dz

    return {
        "median": float(np.median(mag)),
        "p95": float(np.percentile(mag, 95)),
        "max": float(np.max(mag)),
        "fold_fraction": float(np.mean(jacobian <= 0)),
        "jacobian_min": float(np.min(jacobian)),
        "jacobian": jacobian,
        "magnitude": mag,
    }


def make_linear_k1(prev_norm, next_norm):
    return 0.5 * (prev_norm + next_norm)


def make_spline_k1(mat_path, variable, k, offsets, lo, hi):
    k_idx = k - 1
    neighbor_indices = np.array([k_idx + offset for offset in offsets], dtype=int)

    if np.any(neighbor_indices == k_idx):
        raise ValueError("spline offsets must not include 0")
    if neighbor_indices.size < 4:
        raise ValueError("cubic spline K1 needs at least 4 neighboring B-scans")

    order = np.argsort(neighbor_indices)
    neighbor_indices = neighbor_indices[order]

    neighbor_stack = load_bscans_from_mat(
        mat_path,
        variable,
        neighbor_indices,
    )

    spline = CubicSpline(
        neighbor_indices.astype(np.float64),
        neighbor_stack.astype(np.float64),
        axis=2,
        bc_type="not-a-knot",
    )
    k1_raw = spline(float(k_idx))
    k1_norm = normalize_with_limits(k1_raw, lo, hi)

    return k1_raw, k1_norm, neighbor_indices + 1


def save_rgb_before_after(prev_norm, curr_norm, next_norm, corrected_norm, k, output_path):
    before = np.stack([prev_norm, curr_norm, next_norm], axis=-1)
    after = np.stack([prev_norm, corrected_norm, next_norm], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(before, aspect="auto")
    axes[0].set_title(f"Before: {k-1}/{k}/{k+1}")
    axes[1].imshow(after, aspect="auto")
    axes[1].set_title(f"After: {k-1}/K2/{k+1}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path = resolve_output_path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_registration_panel(k1_norm, curr_norm, corrected_norm, k, output_path, k1_title):
    diff_before = curr_norm - k1_norm
    diff_after = corrected_norm - k1_norm

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(k1_norm, cmap="hot", aspect="auto")
    axes[0, 0].set_title(k1_title)
    axes[0, 1].imshow(curr_norm, cmap="hot", aspect="auto")
    axes[0, 1].set_title(f"Original K={k}")
    axes[0, 2].imshow(corrected_norm, cmap="hot", aspect="auto")
    axes[0, 2].set_title("K2 registered")

    im0 = axes[1, 0].imshow(diff_before, cmap="coolwarm", aspect="auto")
    axes[1, 0].set_title("Original K - K1")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 1].imshow(diff_after, cmap="coolwarm", aspect="auto")
    axes[1, 1].set_title("K2 - K1")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    abs_improve = np.abs(diff_before) - np.abs(diff_after)
    im2 = axes[1, 2].imshow(abs_improve, cmap="viridis", aspect="auto")
    axes[1, 2].set_title("|before| - |after|")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes[0, :]:
        ax.axis("off")

    plt.tight_layout()
    output_path = resolve_output_path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_displacement_panel(dz, dx, stats, output_path):
    mag = stats["magnitude"]
    jacobian = stats["jacobian"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im0 = axes[0, 0].imshow(dx, cmap="coolwarm", aspect="auto")
    axes[0, 0].set_title("Dx, A-line direction")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(dz, cmap="coolwarm", aspect="auto")
    axes[0, 1].set_title("Dz, depth direction")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(mag, cmap="magma", aspect="auto")
    axes[1, 0].set_title("|D|")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(jacobian, cmap="coolwarm", aspect="auto", vmin=0, vmax=2)
    axes[1, 1].set_title("Jacobian determinant")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = resolve_output_path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_map_panels(
    original_show,
    corrected_show,
    diff_show,
    k,
    output_path,
    zoom_output_path,
    zoom_start,
    zoom_end,
):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    axes[0].imshow(original_show, cmap="hot", aspect="auto", origin="upper")
    axes[0].axvline(k - 1, linewidth=1.0, color="cyan")
    axes[0].set_title("Original MAP")

    axes[1].imshow(corrected_show, cmap="hot", aspect="auto", origin="upper")
    axes[1].axvline(k - 1, linewidth=1.0, color="cyan")
    axes[1].set_title(f"MAP with only B-scan {k} corrected")

    im = axes[2].imshow(diff_show, cmap="coolwarm", aspect="auto", origin="upper")
    axes[2].axvline(k - 1, linewidth=1.0, color="black")
    axes[2].set_title("Corrected MAP - original MAP")
    fig.colorbar(im, ax=axes[2], fraction=0.018, pad=0.01)

    for ax in axes:
        ax.set_ylabel("A-line index")
    axes[2].set_xlabel("B-scan index")

    plt.tight_layout()
    output_path = resolve_output_path(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    x0 = max(0, zoom_start - 1)
    x1 = min(original_show.shape[1], zoom_end)
    extent = [x0 + 1, x1, original_show.shape[0] - 1, 0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].imshow(
        original_show[:, x0:x1],
        cmap="hot",
        aspect="auto",
        origin="upper",
        extent=extent,
    )
    axes[0].axvline(k, linewidth=1.0, color="cyan")
    axes[0].set_title(f"Original MAP zoom: B-scans {x0 + 1}-{x1}")

    axes[1].imshow(
        corrected_show[:, x0:x1],
        cmap="hot",
        aspect="auto",
        origin="upper",
        extent=extent,
    )
    axes[1].axvline(k, linewidth=1.0, color="cyan")
    axes[1].set_title("Corrected MAP zoom")

    im = axes[2].imshow(
        diff_show[:, x0:x1],
        cmap="coolwarm",
        aspect="auto",
        origin="upper",
        extent=extent,
    )
    axes[2].axvline(k, linewidth=1.0, color="black")
    axes[2].set_title("Zoomed difference")
    fig.colorbar(im, ax=axes[2], fraction=0.018, pad=0.01)

    for ax in axes:
        ax.set_ylabel("A-line index")
    axes[2].set_xlabel("B-scan index")

    plt.tight_layout()
    zoom_output_path = resolve_output_path(zoom_output_path)
    plt.savefig(zoom_output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {zoom_output_path}")


def save_map_comparison(
    mat_path,
    variable,
    k,
    corrected_raw,
    output_path,
    zoom_output_path,
    zoom_start,
    zoom_end,
):
    _, num_alines, num_bscan, _ = get_volume_shape(mat_path, variable)
    original_map = np.zeros((num_alines, num_bscan), dtype=np.float32)

    for i, bscan in enumerate(iter_bscans(mat_path, variable)):
        original_map[:, i] = np.max(np.abs(bscan), axis=0)

    corrected_map = original_map.copy()
    corrected_map[:, k - 1] = np.max(np.abs(corrected_raw), axis=0)

    original_show = np.log1p(original_map)
    corrected_show = np.log1p(corrected_map)
    diff_show = corrected_show - original_show

    save_map_panels(
        original_show,
        corrected_show,
        diff_show,
        k,
        output_path,
        zoom_output_path,
        zoom_start,
        zoom_end,
    )


def parse_int_list(value):
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def run_step2(args):
    prev_raw, curr_raw, next_raw = load_triplet_from_mat(
        args.input,
        args.variable,
        args.k,
    )

    prev_norm, curr_norm, next_norm, lo, hi = normalize_triplet_common(
        prev_raw,
        curr_raw,
        next_raw,
    )

    k1_raw = None
    spline_neighbors = np.array([], dtype=int)

    if args.k1_method == "linear":
        k1_norm = make_linear_k1(prev_norm, next_norm)
        k1_description = f"linear average of B-scans {args.k-1} and {args.k+1}"
        k1_title = f"K1 linear from {args.k-1}/{args.k+1}"
    else:
        k1_raw, k1_norm, spline_neighbors = make_spline_k1(
            args.input,
            args.variable,
            args.k,
            args.spline_offsets,
            lo,
            hi,
        )
        neighbor_text = ",".join(str(int(n)) for n in spline_neighbors)
        k1_description = f"cubic spline from B-scans {neighbor_text}"
        k1_title = "K1 cubic spline"

    dz, dx, k2_norm_from_norm = modified_symmetric_demons(
        reference=k1_norm,
        floating=curr_norm,
        scales=args.scales,
        iterations=args.iterations,
        sigma_update=args.sigma_update,
        sigma_field=args.sigma_field,
        max_step=args.max_step,
    )

    corrected_raw = warp_image(np.asarray(curr_raw, dtype=np.float64), dz, dx, order=1)
    corrected_norm = normalize_with_limits(corrected_raw, lo, hi)

    stats = displacement_stats(dz, dx)

    mi_original_prev = mutual_information_2d(curr_norm, prev_norm, args.bins)
    mi_original_next = mutual_information_2d(curr_norm, next_norm, args.bins)
    mi_k1_prev = mutual_information_2d(k1_norm, prev_norm, args.bins)
    mi_k1_next = mutual_information_2d(k1_norm, next_norm, args.bins)
    mi_k2_prev = mutual_information_2d(corrected_norm, prev_norm, args.bins)
    mi_k2_next = mutual_information_2d(corrected_norm, next_norm, args.bins)
    mi_k_prev_next = mutual_information_2d(prev_norm, next_norm, args.bins)
    mi_k_vs_k1 = mutual_information_2d(curr_norm, k1_norm, args.bins)
    mi_k2_vs_k1 = mutual_information_2d(corrected_norm, k1_norm, args.bins)

    prefix = args.output_prefix or f"step2_Bscan_{args.k}"

    save_rgb_before_after(
        prev_norm,
        curr_norm,
        next_norm,
        corrected_norm,
        args.k,
        f"{prefix}_rgb_before_after.png",
    )
    save_registration_panel(
        k1_norm,
        curr_norm,
        corrected_norm,
        args.k,
        f"{prefix}_registration_panel.png",
        k1_title,
    )
    save_displacement_panel(
        dz,
        dx,
        stats,
        f"{prefix}_displacement_field.png",
    )

    if not args.skip_map:
        save_map_comparison(
            args.input,
            args.variable,
            args.k,
            corrected_raw,
            f"{prefix}_map_original_vs_corrected.png",
            f"{prefix}_map_zoom_{args.map_zoom_start}_{args.map_zoom_end}.png",
            args.map_zoom_start,
            args.map_zoom_end,
        )

    npz_path = resolve_output_path(f"{prefix}_registered_bscan_and_field.npz")
    np.savez_compressed(
        npz_path,
        k=args.k,
        prev_raw=prev_raw,
        curr_raw=curr_raw,
        next_raw=next_raw,
        corrected_raw=corrected_raw,
        prev_norm=prev_norm,
        curr_norm=curr_norm,
        next_norm=next_norm,
        k1_norm=k1_norm,
        k2_norm=corrected_norm,
        k2_norm_from_norm=k2_norm_from_norm,
        k1_raw=np.array([]) if k1_raw is None else k1_raw,
        spline_neighbors=spline_neighbors,
        dz=dz,
        dx=dx,
    )
    print(f"Saved: {npz_path}")

    k2_improved = (mi_k2_prev > mi_original_prev) and (mi_k2_next > mi_original_next)
    k2_better_than_k1 = (mi_k2_prev > mi_k1_prev) and (mi_k2_next > mi_k1_next)
    field_plausible = (
        stats["p95"] <= args.max_plausible_p95
        and stats["max"] <= args.max_plausible_max
        and stats["fold_fraction"] <= args.max_fold_fraction
    )

    if k2_better_than_k1 and field_plausible:
        recommendation = "K2 passes the strict K1-vs-K2 neighbor MI check; inspect overlays before replacing K."
    elif not field_plausible:
        recommendation = "Do not replace K yet: displacement field is not physically plausible enough."
    elif k2_improved:
        if args.k1_method == "spline":
            recommendation = "K2 improves the original K, but strict neighbor MI still favors spline K1; inspect morphology and MAP zoom before replacing K."
        else:
            recommendation = "K2 improves the original K, but strict neighbor MI still favors K1; inspect overlays and consider spline K1."
    else:
        recommendation = "Do not replace K yet: K2 did not improve both neighbor MI scores."

    summary_lines = [
        f"B-scan candidate K: {args.k}",
        f"Input: {args.input}",
        f"Variable: {args.variable}",
        f"K1 method: {k1_description}",
        f"Common normalization limits after log1p(abs(.)): lo={lo:.6g}, hi={hi:.6g}",
        "",
        "Original candidate MI:",
        f"MI({args.k-1},{args.k}) = {mi_original_prev:.6f}",
        f"MI({args.k},{args.k+1}) = {mi_original_next:.6f}",
        f"MI({args.k-1},{args.k+1}) = {mi_k_prev_next:.6f}",
        "",
        "Paper-style acceptance MI:",
        f"MI11 = MI(K1,{args.k-1}) = {mi_k1_prev:.6f}",
        f"MI12 = MI(K1,{args.k+1}) = {mi_k1_next:.6f}",
        f"MI21 = MI(K2,{args.k-1}) = {mi_k2_prev:.6f}",
        f"MI22 = MI(K2,{args.k+1}) = {mi_k2_next:.6f}",
        "",
        "Reference similarity:",
        f"MI(original K,K1) = {mi_k_vs_k1:.6f}",
        f"MI(K2,K1) = {mi_k2_vs_k1:.6f}",
        "",
        "Displacement statistics, pixels:",
        f"median |D| = {stats['median']:.6f}",
        f"95th percentile |D| = {stats['p95']:.6f}",
        f"max |D| = {stats['max']:.6f}",
        f"fold fraction, Jacobian <= 0 = {stats['fold_fraction']:.8f}",
        f"min Jacobian = {stats['jacobian_min']:.6f}",
        "",
        f"K2 improves both neighbor MI scores: {k2_improved}",
        f"K2 beats K1 on both neighbor MI scores: {k2_better_than_k1}",
        f"Displacement field plausible: {field_plausible}",
        f"Recommendation: {recommendation}",
    ]

    summary_path = resolve_output_path(f"{prefix}_summary.txt")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved: {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to IHD3DPA1.mat")
    parser.add_argument("--k", type=int, default=152, help="1-based B-scan index")
    parser.add_argument("--variable", default="IHD3DPA1")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument(
        "--k1-method",
        choices=("spline", "linear"),
        default="spline",
        help="K1 reference generation method",
    )
    parser.add_argument(
        "--spline-offsets",
        type=parse_int_list,
        default=(-3, -2, -1, 1, 2, 3),
        help="Offsets around K used for cubic spline K1",
    )
    parser.add_argument("--scales", type=parse_int_list, default=(4, 2, 1))
    parser.add_argument("--iterations", type=parse_int_list, default=(80, 60, 40))
    parser.add_argument("--sigma-update", type=float, default=0.8)
    parser.add_argument("--sigma-field", type=float, default=1.2)
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--max-plausible-p95", type=float, default=15.0)
    parser.add_argument("--max-plausible-max", type=float, default=30.0)
    parser.add_argument("--max-fold-fraction", type=float, default=0.001)
    parser.add_argument("--map-zoom-start", type=int, default=140)
    parser.add_argument("--map-zoom-end", type=int, default=165)
    parser.add_argument("--skip-map", action="store_true")

    args = parser.parse_args()

    if len(args.scales) != len(args.iterations):
        raise ValueError("--scales and --iterations must have the same length")

    run_step2(args)
